from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from vietocr.tool.translate import build_model
from vietocr.tool.utils import download_weights
from vietocr.tool.logger import Logger
from vietocr.loader.aug import ImgAugTransform
from vietocr.loader.dataloader import OCRDataset

import torch
import random
from itertools import cycle
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model, self.vocab = build_model(config)
        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.val_annotation = config['dataset']['valid_annotation']
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

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.pad,
            label_smoothing=0.1
        )
        # self.criterion = LabelSmoothingLoss(
        #     len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        transform = None
        if self.image_aug:
            transform = ImgAugTransform()

        self.train_loader = self.setup_dataloader(
            self.train_annotation,
            transform
        )
        if self.val_annotation:
            self.val_loader = self.setup_dataloader(self.val_annotation)

        self.train_losses = []

    def setup_dataloader(self, index_file_path, transform=None):
        dataset = OCRDataset(
            index=index_file_path,
            vocab=self.vocab,
            transform=transform,
            image_height=self.config['dataset']['image_height'],
            image_min_width=self.config['dataset']['image_min_width'],
            image_max_width=self.config['dataset']['image_max_width'],
            sequence_max_length=self.config['dataset']['sequence_max_length']
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            ** self.config['dataloader']
        )
        # sampler=sampler,
        # collate_fn=collate_fn,
        # shuffle=False,
        # drop_last=False,
        # **self.config['dataloader'])
        return loader

    def train(self):
        # Statistics
        total_loss = 0

        train_loader = cycle(self.train_loader)
        for step in tqdm(range(self.num_iters), "Training"):
            batch = next(train_loader)
            total_loss = total_loss + self.train_step(batch)
            # loss.backward()
            # clip_grad_norm_(self.model.parameters(), 1)
            # self.optimizer.step()
            # self.scheduler.step()

            if step > 0 and step % self.valid_every == 0:
                metrics = self.valid()
                self.print(
                    f"Train loss: {total_loss / step:.2f}, Val loss: {metrics['val_loss']:.2f}, Char prec: {metrics['char_precision']:.2f}, Seq prec: {metrics['seq_precision']:.2f}"
                )
            #     print(f"accuracy {acc}")
            # loss.backward()

    def precision(self, predict, target, full_seq=False):
        if full_seq:
            return 1 if predict == target else 0

        total = 0
        n = len(target)
        for p, t in zip(predict, target):
            if p == t:
                total += 1

        return total / n

    def batch_precision(self, prediction, target, full_seq=False):
        precision = [
            self.precision(p, t, full_seq=full_seq)
            for p, t in zip(prediction, target)
        ]
        precision = sum(precision) / len(precision)
        return precision

    @torch.no_grad()
    def valid(self):
        char_precision = 0
        seq_precision = 0
        val_loss = 0
        model = self.model.eval()
        n = len(self.val_loader)
        step_to_print = random.choice(range(n))
        for step, batch in tqdm(enumerate(self.val_loader), "Validating", total=n):
            batch = batch.to(self.device)
            target = batch['target']
            # target_mask = batch['target_mask']
            output = model(**batch)

            loss = self.criterion(
                output.view(-1, output.shape[-1]),
                target.flatten()
            )
            val_loss += loss.cpu().item()

            prediction = output.cpu().argmax(dim=-1)

            # String prediction
            prediction = self.vocab.batch_decode(
                prediction.cpu().detach().tolist()
            )
            target = self.vocab.batch_decode(
                target.cpu().detach().tolist()
            )
            if step == step_to_print:
                for p, t in zip(prediction, target):
                    self.print(f"~~~~~\n= {t}\n< {p}")

            # Calculate precision
            # ic(prediction, target)
            # ic(precision, count)
            char_precision += self.batch_precision(
                prediction,
                target,
                full_seq=False
            )
            seq_precision += self.batch_precision(
                prediction,
                target,
                full_seq=True
            )

        char_precision = char_precision / n
        seq_precision = seq_precision / n
        val_loss = val_loss / n
        return dict(
            seq_precision=seq_precision,
            char_precision=char_precision,
            val_loss=val_loss
        )

    def train_step(self, batch):
        model = self.model.train()
        batch = batch.to(self.device)
        # Forward
        self.optimizer.zero_grad()
        output = model(**batch)

        # Calculate loss
        target = batch['target']
        loss = self.criterion(
            output.view(-1, output.shape[-1]),
            target.flatten()
        )

        # Backward and update
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        batch.to('cpu')

        return loss.cpu().item()
        # ic(output)

    def print(self, *args):
        return tqdm.write(" ".join(args))
