from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from vietocr.tool.translate import build_model
from vietocr.tool.utils import download_weights
from vietocr.tool.logger import Logger
from vietocr.loader.aug import ImgAugTransform
from vietocr.loader.dataloader import OCRDataset
from vietocr.model.vocab import VocabS2S
from vietocr.model.transformerocr import VietOCR

import torch
import random
from itertools import cycle
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict
from dataclasses import dataclass, field
from os import path


@dataclass
class AutoExport:
    weight_path: str
    checkpoint_path: str
    name: str
    threshold: float = 0.6
    current_bests: Dict = field(default_factory=dict)

    def run(self, weight, checkpoint, **metrics):
        for metric, value in metrics.items():
            current_best = self.current_bests.get(metric, -1)
            if value > current_best and value > self.threshold:
                self.current_best = value
                self.save(weight, checkpoint)

    # def self1

    def get_path(self, root, name, metric):
        return path.join(root, name + "_best_" + metric + ".pth")


@dataclass
class AutoExport:
    export_path: str
    threshold: float = 0.8
    current_best: float = -1

    def run(self, state_dict, metric):
        if metric < self.threshold:
            return
        if metric > self.current_best:
            self.current_best = metric
            torch.save(state_dict, self.export_path)


class Trainer:
    def __init__(self, config):
        # Add all the config key to this object
        self.__dict__.update(config)

        if 'vocab' in config:
            self.vocab = VocabS2S(config['vocab'])
        elif 'vocab_file':
            self.vocab = VocabS2S.from_file(config['vocab_file'])
        else:
            raise ValueError(
                "Either vocab or vocab file must be in the config")
        self.model = VietOCR(
            backbone=self.model_backbone,
            head_size=self.model_head_size,
            image_size=(self.image_height, self.image_width),
            vocab_size=len(self.vocab),
            sos_token_id=self.vocab.sos_id,
            max_sequence_length=self.max_sequence_length
        )
        self.model.to(config['device'])

        self.iter = 0

        # Optimization stuffs
        self.optimizer = AdamW(self.model.parameters(),
                               betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(
            self.optimizer,
            total_steps=self.training_num_iters,
            max_lr=self.training_lr
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.pad_id,
            label_smoothing=0.1
        )

        # DATA STUFFS
        if getattr(self, "data_augmentation", False):
            transform = ImgAugTransform()
        else:
            transform = None

        self.train_loader = self.setup_dataloader(
            self.data_train_annotation,
            transform
        )
        if hasattr(self, "data_val_annotation"):
            self.val_loader = self.setup_dataloader(
                self.data_val_annotation
            )

        # AUTO SAVE
        self.export = AutoExport(
            export_path=path.join(
                "weights",
                self.experiment_name + ".pth"
            ),
            threshold=0
        )

    def setup_dataloader(self, index_file_path, transform=None):
        dataset = OCRDataset(
            index=index_file_path,
            vocab=self.vocab,
            transform=transform,
            image_height=self.image_height,
            image_width=self.image_width,
            max_sequence_length=self.max_sequence_length
        )
        loader = DataLoader(
            dataset,
            batch_size=self.data_batch_size,
        )
        return loader

    def train(self):
        # Statistics
        total_loss = 0

        train_loader = cycle(self.train_loader)
        for step in tqdm(range(self.training_num_iters), "Training"):
            batch = next(train_loader)
            loss = self.train_step(batch)
            total_loss = total_loss + loss
            # loss.backward()
            # clip_grad_norm_(self.model.parameters(), 1)
            # self.optimizer.step()
            # self.scheduler.step()

            if step > 0 and step % self.training_validate_every == 0:
                metrics = self.valid()
                self.export.run(
                    self.model.state_dict(),
                    metrics['seq_precision']
                )
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

    def compute_loss(self, output, target):
        # output: (batch, sequence, vocab_size)
        # target: (batch, sequence)
        # ic(output.shape, target.shape)
        return self.criterion(
            output[:, 1:].reshape(-1, output.shape[-1]),
            target[:, 1:].flatten()
        )

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

            loss = self.compute_loss(output, target)

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
        loss = self.compute_loss(output, target)

        # Backward and update
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        return loss.cpu().item()
        # ic(output)

    def print(self, *args):
        return tqdm.write(" ".join(args))
