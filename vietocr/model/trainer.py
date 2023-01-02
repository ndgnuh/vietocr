from torchvision import transforms as T
from vietocr.optim.optim import ScheduledOptim
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD, AdamW
from torch import nn
from vietocr.tool.translate import build_model
from vietocr.tool.translate import translate, batch_translate_beam_search
from vietocr.tool.utils import download_weights
from ..tool import utils
from vietocr.tool.logger import Logger
from .stn import SpatialTransformer
from ..loader import aug
from . import losses

import yaml
import torch
from vietocr.tool.stats import AverageStatistic
from vietocr.loader.dataloader_v1 import DataGen
from vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, OneCycleLR

import torchvision

from vietocr.tool.utils import compute_accuracy
from PIL import Image
from functools import partial, cached_property
from tqdm import tqdm as std_tqdm
from os import path
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random
from dataclasses import dataclass
from .. import const

tqdm = partial(std_tqdm, dynamic_ncols=True)
print = std_tqdm.write


@dataclass
class TeacherForcingScheduler:
    start_step: int = 0
    end_step: int = 30000
    p0: float = 1
    _step: int = 0

    def current_ratio(self):
        p = 1 - (self._step - self.start_step) / \
            (self.end_step - self.start_step)
        p = self.p0 * p
        return p

    def step(self):
        p = self.current_ratio()
        self._step = self._step + 1
        return random.random() <= p


class Trainer():
    def __init__(self, config):

        self.config = config

        self.device = utils.get_device(config.get('device', None))
        self.num_iters = config['training']['iters']

        self.train_annotation = config['training']['train_annotation']
        self.valid_annotation = config['training']['valid_annotation']
        self.dataset_name = config['name']

        self.batch_size = config['training']['dl_batch_size']
        self.print_every = config['training']['print_every']
        self.valid_every = config['training']['valid_every']
        self.image_aug = config['training']['augment']

        self.name = config['name']

        self.model, self.vocab = build_model(config)
        self.support = DomAvs(
            cnn=self.model.cnn,
            aug=aug.default_augment
        )
        self.support.to(self.device)

        if 'weights' in config:
            weights = self.load_weights(config['weights'])
            errors = self.model.load_state_dict(weights, strict=False)
            errors = '\n'.join([
                f'\t{k}' for k in
                (errors.missing_keys + errors.unexpected_keys)
            ])
            self.print(f"Mismatch keys:\n{errors}")

        # if config.get('pretrained', None) is not None:
        #     weight_file = download_weights(
        #         **config['pretrain'], quiet=config['quiet'])
        #     self.load_weights(weight_file)

        self.iter = 0

        self.optimizer = AdamW(self.model.parameters(),
                               lr=config['training']['max_lr'],
                               betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(
            self.optimizer,
            total_steps=self.num_iters,
            max_lr=config['training']['max_lr'],
            pct_start=config['training']['pct_start'],
        )

        self.model_type = config['type']
        if config['type'] == 'ctc':
            self.criterion = losses.CTCLoss(
                vocab=self.vocab,
                **config['training'].get('loss_options', {})
            )
        elif config['type'] == 'seq2seq':
            self.criterion = losses.CrossEntropyLoss(
                vocab=self.vocab,
                **config['training'].get('loss_options', {})
            )

        transforms = None
        if self.image_aug:
            transforms = aug.T.Compose([
                aug.default_augment,
                aug.T.ToPILImage()
            ])

        self.train_gen = self.data_gen(
            self.train_annotation,
            transform=transforms,
            curriculum=True
        )

        if self.valid_annotation:
            self.valid_gen = self.data_gen(
                self.valid_annotation,
                curriculum=False
            )

        self.train_losses = []

        # Teacher forcing scheduler
        # TODO: convert to the name-options format
        self.tfs = TeacherForcingScheduler(
            start_step=config['training'].get('teacher_forcing_start', 0),
            end_step=config['training'].get(
                'teacher_forcing_end',
                config['training']['iters']
            ),
            p0=config['training'].get('teacher_forcing_max_prob', 1)
        )

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
                # Status line
                total_loss = total_loss / self.print_every
                lr = self.optimizer.param_groups[0]['lr']

                info = {
                    "iter": f'{self.iter: 06d}',
                    "train loss": f'{self.iter: 06d}',
                    "train loss": f'{total_loss:.4f}',
                    "lr": f'{lr:.3e}',
                    "tfr": f'{self.tfs.current_ratio():.3e}',
                    "load time": f'{total_loader_time:.4e}',
                    "gpu time": f'{total_gpu_time:.4e}',
                }
                info = ' - '.join(f'{k}: {v}' for k, v in info.items())

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                self.print(info)

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

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights_path)
                    best_acc = acc_full_seq

    @ torch.no_grad()
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

            # Validation loss
            # CE Loss requires (batch, class, ...)
            loss = self.criterion(outputs, tgt_output).item()

            # Validation accuracy
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

    def load_weights(self, path):
        if path.startswith('http'):
            weights = torch.load(download_weights(path))
        else:
            weights = torch.load(path)
        return weights

    # def load_weights(self, filename):
    #     state_dict = torch.load(
    #         filename, map_location=torch.device(self.device))

    #     for name, param in self.model.named_parameters():
    #         if name not in state_dict:
    #             print('{} not found'.format(name))
    #         elif state_dict[name].shape != param.shape:
    #             print('{} missmatching shape, required {} but found {}'.format(
    #                 name, param.shape, state_dict[name].shape))
    #             del state_dict[name]

    #     self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        dirname = path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.model.state_dict(), filename)
        self.print(f"Model weights saved to {filename}")

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(
            self.device, non_blocking=True)

        batch = {
            'img': img, 'tgt_input': tgt_input,
            'tgt_output': tgt_output, 'tgt_padding_mask': tgt_padding_mask,
            'filenames': batch['filenames']
        }

        return batch

    def data_gen(self, annotation_path, transform=None, curriculum=False):
        lmdb_path = path.join(
            const.lmdb_dir,
            utils.annotation_uuid(annotation_path)
        )
        data_root = path.dirname(annotation_path)
        annotation_path = path.basename(annotation_path)
        dataset = OCRDataset(lmdb_path=lmdb_path,
                             root_dir=data_root,
                             annotation_path=annotation_path,
                             vocab=self.vocab, transform=transform,
                             image_height=self.config['image_height'],
                             image_min_width=self.config['image_min_width'],
                             image_max_width=self.config['image_max_width'])

        sampler = ClusterRandomSampler(
            dataset,
            self.batch_size,
            shuffle=True,
            curriculum=curriculum
        )
        collate_fn = Collator()

        gen = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=self.config['training']['dl_num_workers'],
            pin_memory=self.config['training']['dl_pin_memory']
        )

        return gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        self.optimizer.zero_grad()
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch[
            'tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

        outputs = self.model(
            img,
            tgt_output,
            tgt_key_padding_mask=tgt_padding_mask,
            teacher_forcing=self.tfs.step()
        )

        # CE Loss requires (batch, class, ...)
        loss = self.criterion(outputs, tgt_output)

        # Extra supervision
        sp_loss = self.support(img)

        # Total
        total_loss = loss + sp_loss
        # total_loss = loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def print(self, *args, delim=", "):
        info = delim.join(args)
        std_tqdm.write(info)
        with open(self.export_logs_path, "a") as f:
            f.write(info.rstrip() + "\n")

    @ cached_property
    def export_weights_path(self):
        return path.join(const.weight_dir, self.name + ".pth")

    @ cached_property
    def export_checkpoints_path(self):
        return path.join(const.checkpoint_dir, self.name + ".pth")

    @ cached_property
    def export_logs_path(self):
        return path.join(const.log_dir, self.name + ".log")


class DomAvs(nn.Module):
    def __init__(self, cnn, aug):
        super().__init__()
        self.cnn = cnn
        # TODO: better augmentation
        with torch.no_grad():
            device = next(cnn.parameters()).device
            x = torch.rand(1, 3, 112, 112, device=device)
            x = self.cnn(x)
            hidden_size = x.shape[-1]

        # Generator
        self.stn = SpatialTransformer(3)
        self.g_net = aug
        self.g_loss = nn.SmoothL1Loss()

        # Discriminator
        self.d_net = nn.Linear(hidden_size, 2)
        self.d_loss = nn.CrossEntropyLoss()

    def forward(self, image):
        batch_size = image.shape[0]

        # Generate loss
        gen = self.g_net(image)
        while (image == gen).all():
            gen = self.g_net(image)
        gen = self.cnn(gen)
        image = self.cnn(image)
        g_loss = self.g_loss(gen, image)

        # Discriminate loss
        # d_input = torch.cat([gen, image], dim=1)
        # d_input = d_input.mean(dim=0)
        # d_output = self.d_net(d_input)
        # d_label = torch.tensor(
        #     [0] * batch_size + [1] * batch_size,
        #     device=d_output.device)
        # d_loss = self.d_loss(d_output, d_label)

        # Total loss
        loss = g_loss
        return loss
