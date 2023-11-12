import random
from dataclasses import dataclass
from typing import Optional

import torch
from lightning import Fabric
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataloaders import Sample, get_dataloader
from .models import OCRModel
from .models.losses import CrossEntropyLoss, CTCLoss
from .tools import resize_image
from .vocabs import Vocab, get_vocab


class NormalizedSGD(optim.SGD):
    def step(self, *args, **kwargs):
        # Normalize gradient
        for pg in self.param_groups:
            for p in pg["params"]:
                # No gradient
                if p.grad is None:
                    continue

                try:
                    p.grad = F.normalize(p.grad)
                except IndexError:
                    # Ignore dim errors
                    continue

        # SGD step
        super().step(*args, **kwargs)


@dataclass
class DataIter:
    dataloader: DataLoader
    total_steps: int

    def __len__(self):
        # Needed for tqdm to produce the progress bar
        return self.total_steps

    def __iter__(self):
        step = 1
        while True:
            for i, data in enumerate(self.dataloader):
                yield step, data
                step = step + 1
                if step > self.total_steps:
                    return


class Trainer:
    def __init__(
        self,
        # Modelling
        lang: str,
        vocab_type: str = "ctc",
        image_height: int = 32,
        image_min_width: int = 32,
        image_max_width: int = 512,
        # Data
        train_data: Optional[str] = None,
        val_data: Optional[str] = None,
        test_data: Optional[str] = None,
        # Training
        max_steps: int = 100_000,
        validate_every: int = 2_000,
        lr: float = 1e-3,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        backbone_config = {
            "name": "mlp_mixer_tiny",
            "image_height": image_height,
        }
        head_config = "linear"
        optim_config = {"lr": 1e-3, "momentum": 0.09, "weight_decay": 1e-5}

        # =============
        # | Modelling |
        # =============
        self.vocab: Vocab = get_vocab(lang=lang)
        self.model = OCRModel(len(self.vocab), backbone_config, head_config)
        self.fabric = Fabric()
        self.optimizer = NormalizedSGD(self.model.parameters(), **optim_config)
        self.criterion = CTCLoss(self.vocab)

        # ========
        # | Data |
        # ========
        def transform(sample: Sample) -> Sample:
            image = sample.image
            image = image.convert("RGB")
            image = resize_image(image, image_height, 32, 500)
            target = self.vocab.encode(sample.target)
            new_sample = Sample(image, target)
            return new_sample

        self.batch_size = 32
        kwargs = {
            "transform": transform,
            "batch_size": self.batch_size,
            "num_workers": 12,
            "pin_memory": True,
        }
        if train_data is not None:
            self.train_loader = get_dataloader(train_data, **kwargs)
        if val_data is not None:
            self.val_loader = get_dataloader(val_data, **kwargs)

        # Scheduling
        self.max_epochs = 100
        self.max_steps = max_steps or self.max_epochs * self.batch_size
        self.validate_every = validate_every

    @torch.no_grad()
    def run_validation(self):
        # Setup
        model = self.fabric.setup(self.model)
        val_loader = self.fabric.setup_dataloaders(self.val_loader)
        model = model.eval()

        # Validate
        sample_predictions = []
        losses = []
        pbar = tqdm(val_loader, "[V]", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(pbar):
            (images, targets, target_lengths) = batch
            outputs = model(images)
            loss = self.criterion(outputs, targets, target_lengths)
            loss = loss.item()

            predicts = outputs.argmax(dim=-1)
            for pr, gt in zip(predicts, targets):
                pr = self.vocab.decode(pr.tolist())
                gt = self.vocab.decode(gt.tolist())
                sample_predictions.append([pr, gt])

            pbar.set_postfix({"loss": loss})
            losses.append(loss)

        shown_predictions = random.choices(sample_predictions, k=5)
        mean_loss = sum(losses) / len(losses)
        n = 0
        for pr, gt in shown_predictions:
            n = max(n, len(pr), len(gt))
        for pr, gt in shown_predictions:
            tqdm.write("PR: " + pr)
            tqdm.write("GT: " + gt)
            tqdm.write("=" * (n + 4))
        tqdm.write("Mean validation loss: %.4e" % mean_loss)
        return mean_loss

    def fit(self):
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        train_loader = self.fabric.setup_dataloaders(self.train_loader)
        train_loader = DataIter(train_loader, self.max_steps)
        # train_loader = EchoDataLoader(train_loader)

        pbar = tqdm(train_loader, "Train", dynamic_ncols=True)
        for step, (images, targets, target_lengths) in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, targets, target_lengths)
            loss.backward()
            optimizer.step()

            # Validation
            if step % self.validate_every == 0:
                model.eval()
                self.run_validation()
                model.train()

            # Logging
            pbar.set_postfix({"loss": loss.item()})
