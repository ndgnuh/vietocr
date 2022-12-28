from vietocr.optim.optim import ScheduledOptim
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD, AdamW
from torch import nn
from vietocr.tool.translate import build_model
from vietocr.tool.translate import translate, batch_translate_beam_search
from vietocr.tool.utils import download_weights
from vietocr.tool.logger import Logger
from vietocr.loader.aug import ImgAugTransform

import yaml
import torch
from vietocr.tool.stats import AverageStatistic
from vietocr.loader.dataloader_v1 import DataGen
from vietocr.loader.dataloader import (
    OCRDataset,
    ClusterRandomSampler,
    Collator,
    collate_dlen_labels
)
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, OneCycleLR

import torchvision

from vietocr.tool.utils import compute_accuracy
from PIL import Image
from functools import partial
from tqdm import tqdm as std_tqdm
from os import path
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random

tqdm = partial(std_tqdm, dynamic_ncols=True)
print = std_tqdm.write


class CTCLoss(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.ctc = nn.CTCLoss(*a, **k)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        # outputs: [time, batch, class]
        # targets: [batch, max_length]
        logits = self.log_softmax(outputs)

        # target_lengths: [batch]
        target_lengths = torch.count_nonzero(targets != self.ctc.blank, dim=1)

        # input_lengths: [batch]
        # use time * batch for now
        input_lengths = torch.tensor(
            [logits.shape[0]] * logits.shape[1],
            device=logits.device
        )

        # ctc loss

        loss = self.ctc(logits, targets, input_lengths, target_lengths)
        return loss


class Trainer():
    def __init__(self, config, augmentor=ImgAugTransform()):

        self.config = config
        self.model, self.vocab = build_model(config)

        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.dataset_name = config['dataset']['name']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']

        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']

        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']
        self.metrics = config['trainer']['metrics']
        logger = config['trainer']['log']

        if logger:
            self.logger = Logger(logger)

        if config.get('pretrained', None) is not None:
            weight_file = download_weights(
                **config['pretrain'], quiet=config['quiet'])
            self.load_weights(weight_file)

        self.iter = 0

        self.optimizer = AdamW(self.model.parameters(),
                               betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(
            self.optimizer, total_steps=self.num_iters, **config['optimizer'])
#        self.optimizer = ScheduledOptim(
#            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#            #config['transformer']['d_model'],
#            512,
#            **config['optimizer'])

        # self.criterion = nn.CrossEntropyLoss(
        #     ignore_index=self.vocab.pad_id,
        #     label_smoothing=0.1
        # )
        self.criterion = CTCLoss(blank=self.vocab.blank_id, zero_infinity=True)
        # self.criterion = LabelSmoothingLoss(
        #     len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        transforms = None
        if self.image_aug:
            transforms = augmentor

        self.lmdb_cache_path = config.get("lmdb_cache_path", "data-cache")
        os.makedirs(self.lmdb_cache_path, exist_ok=True)

        self.train_gen = self.data_gen(
            path.join(self.lmdb_cache_path, f'train_{self.dataset_name}'),
            self.data_root,
            self.train_annotation,
            self.masked_language_model,
            transform=transforms
        )

        if self.valid_annotation:
            self.valid_gen = self.data_gen(
                path.join(self.lmdb_cache_path, f'valid_{self.dataset_name}'),
                self.data_root,
                self.valid_annotation,
                masked_language_model=False
            )

        self.train_losses = []

    def train(self):
        total_loss = 0

        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0

        data_iter = iter(self.train_gen)
        for i in tqdm(range(self.num_iters), "training"):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter,
                                                                                                                      total_loss /
                                                                                                                      self.print_every, self.optimizer.param_groups[
                                                                                                                          0]['lr'],
                                                                                                                      total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info)
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                metrics = self.validate()
                val_loss = metrics['val_loss']
                acc_full_seq = metrics['acc_full_seq']
                acc_per_char = metrics['acc_per_char']
                info = ' '.join([
                    f'iter: {self.iter:06d}',
                    f'valid loss: {val_loss:.4f}',
                    f'acc full seq: {acc_full_seq:.4f}',
                    f'acc per char: {acc_per_char:.4f}',
                ])
                self.print(info)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights)
                    best_acc = acc_full_seq

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        metrics = dict(
            val_loss=AverageStatistic(),
            acc_per_char=AverageStatistic(),
            acc_full_seq=AverageStatistic()
        )
        all_pr_sents = []
        all_gt_sents = []

        pbar = tqdm(
            enumerate(self.valid_gen),
            "Validating",
            total=len(self.valid_gen)
        )
        for step, batch in pbar:
            batch = self.batch_to_device(batch)
            img = batch['img'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)

            # Predict, no teacher forcing
            # the tgt output is only for seq length
            outputs = self.model(img, tgt_output)
            probs, translated = outputs.topk(k=1, dim=-1)
            # TODO: add confidence to prediction
            # perferably to the predictor, so the the outputs
            # are united, and there's no mismatch between codes
            # probs = probs.squeeze(-1)
            translated = translated.squeeze(-1)

            # TODO: condition for this transpose, only CTC need transpose
            translated = translated.transpose(0, 1)

            # Validation loss
            # CE Loss requires (batch, class, ...)
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                loss = self.criterion(
                    outputs.transpose(-1, 1), tgt_output).item()
            else:
                loss = self.criterion(outputs, tgt_output).item()

            # Validation accuracy
            # ic(translated.shape, tgt_output.shape)
            pr_sents = self.vocab.batch_decode(translated.tolist())
            gt_sents = self.vocab.batch_decode(tgt_output.tolist())
            acc_pc = compute_accuracy(pr_sents, gt_sents, 'per_char')
            acc_fs = compute_accuracy(pr_sents, gt_sents, 'full_sequence')

            # Append results
            all_pr_sents.extend(pr_sents)
            all_gt_sents.extend(gt_sents)

            metrics['val_loss'].append(loss)
            metrics['acc_per_char'].append(acc_pc)
            metrics['acc_full_seq'].append(acc_fs)

        # Tbp = To be printed
        # Print some random samples results
        # TODO: num printing samples (k)
        ic(len(all_gt_sents), len(all_pr_sents))
        tbp = random.choices(range(len(all_pr_sents)), k=5)
        tbp = [
            f"GT:   {all_gt_sents[i]}\nPR:   {all_pr_sents[i]}"
            for i in tbp
        ]
        self.print(("\n~~~~~~~~~\n").join(tbp))

        # Return average metrics
        means = {k: s.summarize() for k, s in metrics.items()}
        return means

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []
        prob = None
        pbar = tqdm(self.valid_gen, "Calculating accuracy")
        for batch in pbar:
            batch = self.batch_to_device(batch)

            translated_sentence, prob = translate(batch['img'], self.model)

            pred_sent = self.vocab.decode(translated_sentence.tolist())
            actual_sent = self.vocab.decode(batch['tgt_output'].tolist())
            for (p, g) in zip(pred_sent, actual_sent):
                print(f"= {p}")
                print(f"< {g}")
                print("~" * 10)
            # img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)

            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob

    def precision(self, sample=None):

        pred_sents, actual_sents, _, _ = self.predict(sample=sample)

        acc_full_seq = compute_accuracy(
            actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(
            actual_sents, pred_sents, mode='per_char')

        return acc_full_seq, acc_per_char

    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16):

        pred_sents, actual_sents, img_files, probs = self.predict(sample)

        if errorcase:
            wrongs = []
            for i in range(len(img_files)):
                if pred_sents[i] != actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {
            'family': fontname,
            'size': fontsize
        }

        for vis_idx in range(0, len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx]

            img = Image.open(open(img_path, 'rb'))
            plt.figure()
            plt.imshow(img)
            plt.title('prob: {:.3f} - pred: {} - actual: {}'.format(prob,
                                                                    pred_sent, actual_sent), loc='left', fontdict=fontdict)
            plt.axis('off')

        plt.show()

    def visualize_dataset(self, sample=16, fontname='serif'):
        n = 0
        for batch in self.train_gen:
            for i in range(self.batch_size):
                img = batch['img'][i].numpy().transpose(1, 2, 0)
                sent = self.vocab.decode(batch['tgt_input'].T[i].tolist())

                plt.figure()
                plt.title('sent: {}'.format(sent),
                          loc='center', fontname=fontname)
                plt.imshow(img)
                plt.axis('off')

                n += 1
                if n >= sample:
                    plt.show()
                    return

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        optim = ScheduledOptim(
            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            self.config['transformer']['d_model'], **self.config['optimizer'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter': self.iter, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}

        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(
            filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(
                    name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        # tgt_padding_mask = batch['tgt_padding_mask'].to(
        #     self.device, non_blocking=True)

        batch = {
            'img': img,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            # 'tgt_padding_mask': tgt_padding_mask,
            # 'filenames': batch['filenames']
        }

        return batch

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path,
                             root_dir=data_root, annotation_path=annotation,
                             vocab=self.vocab, transform=transform,
                             image_height=self.config['dataset']['image_height'],
                             image_min_width=self.config['dataset']['image_min_width'],
                             image_max_width=self.config['dataset']['image_max_width'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)

        gen = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_dlen_labels,
            shuffle=False,
            drop_last=False,
            **self.config['dataloader'])

        return gen

    def data_gen_v1(self, lmdb_path, data_root, annotation):
        data_gen = DataGen(data_root, annotation, self.vocab, 'cpu',
                           image_height=self.config['dataset']['image_height'],
                           image_min_width=self.config['dataset']['image_min_width'],
                           image_max_width=self.config['dataset']['image_max_width'])

        return data_gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output = batch['img'], batch['tgt_input'], batch['tgt_output']

        outputs = self.model(img, tgt_input)

        # CE Loss requires (batch, class, ...)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            loss = self.criterion(outputs.transpose(-1, 1), tgt_output)
        else:
            loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def print(self, *args, delim=", "):
        info = delim.join(args)
        std_tqdm.write(info)
