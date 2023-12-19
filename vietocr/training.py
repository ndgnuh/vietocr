"""
Module for training models.

In case you are wondering, no I did not write those comment boxes.
Vim did it.

# +---------------+
# | Nice isn't it |
# +---------------+
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from lightning import Fabric
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentations import get_augmentation
from .dataloaders import Sample, get_dataloader
from .metrics import Avg, acc_full_sequence, acc_per_char
from .models import CosineWWRD, CTCLoss, OCRModel
from .tools import resize_image
from .vocabs import Vocab, get_vocab


@dataclass
class ModelConfig:
    language: str
    vocab_type: str
    backbone: Union[Dict, str]
    head: Union[Dict, str]
    image_height: int = None
    image_min_width: int = None
    image_max_width: int = None
    weight_path: Optional[str] = None
    onnx_path: Optional[str] = None


@torch.no_grad()
def normalize_grad_(parameters):
    nn.utils.clip_grad_norm_(parameters, 10)
    # from copy import copy
    # scale = 1.
    for p in parameters:
        if p.grad is None:
            print(f"Parameter {id(p)} has no gradient, skipping")
            continue

        grad = p.grad
        dim = min(1, grad.dim)
        p.grad.data = F.normalize(p.grad.data, dim=dim)


@dataclass
class DataIter:
    dataloader: DataLoader
    total_steps: int

    def __len__(self):
        # Len is needed for tqdm to produce the progress bar
        return self.total_steps

    def __iter__(self):
        step = 1
        batch_idx = 0
        while True:
            for i, data in enumerate(self.dataloader):
                target_lengths = data[2]
                masks = target_lengths > 0
                if torch.count_nonzero(masks) == 0:
                    continue
                yield step, batch_idx, data
                step = step + 1
                if step > self.total_steps:
                    return
            batch_idx = batch_idx + 1


class Trainer:
    def __init__(
        self,
        # Modelling
        lang: str,
        vocab_type: str = "ctc",
        image_height: int = 32,
        image_min_width: int = 28,
        image_max_width: int = 768,
        # Data
        train_data: Optional[str] = None,
        val_data: Optional[str] = None,
        test_data: Optional[str] = None,
        # Training
        aug_prob: float = 0.5,
        max_steps: int = 100_000,
        validate_every: int = 2_000,
        lr: float = 1e-3,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        # +--------------------+
        # | TODO: Model config |
        # +--------------------+
        backbone_config = {
            "name": "tr_resnet18",
            "image_height": image_height,
        }
        # backbone_config = "tr_resnet18"
        head_config = "linear"

        # +-----------+
        # | Modelling |
        # +-----------+
        self.vocab: Vocab = get_vocab(lang=lang)
        self.model = OCRModel(len(self.vocab), backbone_config, head_config)
        self.fabric = Fabric()

        # +---------------------+
        # | TODO: model loading |
        # +---------------------+
        try:
            weights = torch.load("./model.pt", map_location="cpu")
            self.model.load_state_dict(weights)
        except Exception:
            pass

        # +--------------+
        # | Optimization |
        # +--------------+
        n = int(max_steps / 10_000)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            momentum=0.9,
            nesterov=True,
        )
        self.lr_scheduler = CosineWWRD(
            self.optimizer,
            total_steps=max_steps,
            num_warmup_steps=1000,
            cycle_length=max_steps,
            start_ratio=1,
            peak_ratio=1,
            decay=math.pow(1e-6 / lr, 1 / n),
        )
        self.criterion = CTCLoss(self.vocab)

        # +-------------------------------+
        # | Pre-process data for training |
        # +-------------------------------+
        self.augment = get_augmentation(aug_prob)

        def transform(sample: Sample) -> Sample:
            image = sample.image
            if self.augment is not None:
                image = self.augment(image=image)["image"]
            image = resize_image(image, image_height, image_min_width, image_max_width)
            target = self.vocab.encode(sample.target)
            new_sample = Sample(image, target)
            return new_sample

        # +------------------------------+
        # | Train/validation dataloaders |
        # +------------------------------+
        self.batch_size = batch_size
        kwargs = {
            "transform": transform,
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "shuffle": shuffle,
        }
        if train_data is not None:
            self.train_loader = get_dataloader(train_data, **kwargs)
        if val_data is not None:
            self.val_loader = get_dataloader(val_data, **kwargs)

        # ==============
        # | Scheduling |
        # ==============
        self.max_epochs = 100
        self.max_steps = max_steps or self.max_epochs * self.batch_size
        self.validate_every = validate_every

        # ===========
        # | Logging |
        # ===========
        self.logger = SummaryWriter(flush_secs=1)
        self.bests = dict(acc_fs=0, acc_pc=0)

    @torch.no_grad()
    def run_validation(self, step=0):
        """Run validation loop. Compute validation loss,
        full sequence and per-character accuracy.

        Args:
            step (int):
                The current global step, this is used for
                logging to tensorboard. Default: `0`.
        """
        # +-------+
        # | Setup |
        # +-------+
        model = self.fabric.setup(self.model)
        val_loader = self.fabric.setup_dataloaders(self.val_loader)
        model = model.eval()

        # +-----------------+
        # | Average metrics |
        # +-----------------+
        acc_fs = Avg()
        acc_pc = Avg()
        loss_avg = Avg()

        # +-----------------+
        # | Validation loop |
        # +-----------------+
        sample_predictions = []
        pbar = tqdm(val_loader, "Validate", dynamic_ncols=True, leave=False)
        for _, batch in enumerate(pbar):
            # +---------+
            # | Forward |
            # +---------+
            (images, targets, target_lengths) = batch
            outputs = model(images)
            loss = self.criterion(outputs, targets, target_lengths)
            loss = loss.item()

            # +-----------------------------+
            # | Decode and compute accuracy |
            # +-----------------------------+
            predicts = outputs.argmax(dim=-1)
            for pr, gt in zip(predicts, targets):
                pr = self.vocab.decode(pr.tolist())
                gt = self.vocab.decode(gt.tolist())
                sample_predictions.append([pr, gt])

                acc_fs.append(acc_full_sequence(pr, gt))
                acc_pc.append(acc_per_char(pr, gt))

            # +---------+
            # | Logging |
            # +---------+
            pbar.set_postfix({"loss": loss})
            loss_avg.append(loss)

        # +-----------------+
        # | Show prediction |
        # +-----------------+
        shown_predictions = random.choices(sample_predictions, k=5)
        n = 0
        for pr, gt in shown_predictions:
            n = max(n, len(pr), len(gt))
        for pr, gt in shown_predictions:
            tqdm.write("PR: " + pr)
            tqdm.write("GT: " + gt)
            tqdm.write("=" * (n + 4))

        # +--------------------+
        # | Log to tensorboard |
        # +--------------------+
        self.logger.add_scalar(
            "val/loss",
            loss_avg.get(),
            step,
            display_name="Validation loss",
        )
        self.logger.add_scalar(
            "val/acc-fs",
            acc_fs.get(),
            step,
            display_name="Accuracy full sequence",
        )
        self.logger.add_scalar(
            "val/acc-pc",
            acc_pc.get(),
            step,
            display_name="Accuracy per-characters",
        )

        # +---------------+
        # | Log to stdout |
        # +---------------+
        fmt_data = (loss_avg.get(), acc_fs.get(), acc_pc.get())
        self.bests["acc_fs"] = max(fmt_data[1], self.bests["acc_fs"])
        self.bests["acc_pc"] = max(fmt_data[2], self.bests["acc_pc"])
        fmt = "[Validation] loss: %.5e - Full sequence: %.4f - Per-char: %.4f"
        tqdm.write(fmt % fmt_data)

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pt")
        # tqdm.write("model saved to model.pt")

    def fit(self):
        """Run training"""
        # +------------------------------------------+
        # | Setup model, optimizers, and data loader |
        # +------------------------------------------+
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        lr_scheduler = self.lr_scheduler
        train_loader = self.fabric.setup_dataloaders(self.train_loader)
        train_loader = DataIter(train_loader, self.max_steps)

        # +-------------------------------+
        # | Setup logging/initial logging |
        # +-------------------------------+
        tqdm.write("Num training batches: %d" % len(self.train_loader))
        tqdm.write("Num validation batches: %d" % len(self.val_loader))
        logger: SummaryWriter = self.logger
        logger.add_text("Model", repr(model), 0)

        # +---------------+
        # | Training loop |
        # +---------------+
        pbar = tqdm(train_loader, "Train", dynamic_ncols=True)
        for step, batch_idx, (images, targets, target_lengths) in pbar:
            # +-------------------------+
            # | Forward/backward/update |
            # +-------------------------+
            optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, targets, target_lengths)
            loss.backward()
            normalize_grad_(model.parameters())
            optimizer.step()
            lr_scheduler.step()

            # +------------------+
            # | Model checkpoint |
            # +------------------+
            if step % 1000 == 0:
                self.save_model()

            # +---------------------------------------+
            # | Show sample predictions every N steps |
            # +---------------------------------------+
            if step % 200 == 0:
                scores, predicts = model.post_process(outputs)
                predicts = predicts.detach().cpu()
                targets = targets.detach().cpu()
                best_fmt = ", ".join(f"{k}={v}" for k, v in self.bests.items())
                tqdm.write(f"== step {step} == best: {best_fmt}")
                for pr, gt in zip(predicts, targets):
                    pr = self.vocab.decode(pr.tolist())
                    gt = self.vocab.decode(gt.tolist())
                    tqdm.write(f"> PR: {pr}")
                    tqdm.write(f"  GT: {gt}")

            # +---------------------+
            # | Run validation loop |
            # +---------------------+
            if step % self.validate_every == 0:
                model.eval()
                self.run_validation(step)
                model.train()

            # +---------+
            # | Logging |
            # +---------+
            width = images.shape[-1]
            lr = lr_scheduler.get_last_lr()[0]
            self.logger.add_scalar("train/loss", loss, step)
            self.logger.add_scalar("train/width", width, step)
            self.logger.add_scalar("train/lr", lr, step)
            self.logger.add_scalar("train/batch", batch_idx, step)
            pbar.set_postfix({"loss": loss.item(), "lr": f"{lr:.4e}"})
