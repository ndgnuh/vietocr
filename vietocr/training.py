import random
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from lightning import Fabric
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataloaders import Sample, get_dataloader
from .models import OCRModel
from .models.losses import CrossEntropyLoss, CTCLoss
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


def normalize_grad_(parameters):
    for p in parameters:
        if p.grad is None:
            continue

        grad = p.grad
        dim = min(1, grad.dim)
        p.grad = F.normalize(p.grad, dim=dim)


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
        optim_config = {"lr": lr, "momentum": 0.09, "weight_decay": 1e-5}

        # =============
        # | Modelling |
        # =============
        self.vocab: Vocab = get_vocab(lang=lang)
        self.model = OCRModel(len(self.vocab), backbone_config, head_config)
        self.fabric = Fabric()
        # self.optimizer = NormalizedSGD(self.model.parameters(), **optim_config)
        self.optimizer = optim.SGD(self.model.parameters(), **optim_config)
        self.criterion = CTCLoss(self.vocab)

        # ========
        # | Data |
        # ========
        def transform(sample: Sample) -> Sample:
            image = sample.image
            image = image.convert("RGB")
            image = resize_image(image, image_height, image_min_width, image_max_width)
            target = self.vocab.encode(sample.target)
            new_sample = Sample(image, target)
            return new_sample

        self.batch_size = batch_size
        kwargs = {
            "transform": transform,
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
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

    @torch.no_grad()
    def run_validation(self, step=0):
        # Setup
        model = self.fabric.setup(self.model)
        val_loader = self.fabric.setup_dataloaders(self.val_loader)
        model = model.eval()

        # Validate
        sample_predictions = []
        losses = []
        pbar = tqdm(val_loader, "Validate", dynamic_ncols=True, leave=False)
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
        self.logger.add_scalar("val/loss", mean_loss, step)
        tqdm.write("Mean validation loss: %.6e" % mean_loss)
        return mean_loss

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pt")
        # tqdm.write("model saved to model.pt")

    def fit(self):
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        train_loader = self.fabric.setup_dataloaders(self.train_loader)
        train_loader = DataIter(train_loader, self.max_steps)
        tqdm.write("Num training batches: %d" % len(self.train_loader))
        tqdm.write("Num validation batches: %d" % len(self.val_loader))
        # train_loader = EchoDataLoader(train_loader)

        pbar = tqdm(train_loader, "Train", dynamic_ncols=True)
        for step, (images, targets, target_lengths) in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, targets, target_lengths)
            normalize_grad_(model.parameters())
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                self.save_model()


            # Show sample predictions
            if step % 2000 == 0:
                predicts, _ = model.post_process(outputs)
                predicts = predicts.detach().cpu()
                targets = targets.detach().cpu()
                tqdm.write("=" * 10 + f" STEP {step} " + "=" * 10)
                for pr, gt in zip(predicts, targets):
                    pr = self.vocab.decode(pr.tolist())
                    gt = self.vocab.decode(gt.tolist())
                    tqdm.write(f"PR: {pr}")
                    tqdm.write(f"GT: {gt}")
                    tqdm.write("=" * 10)
            # Validation
            if step % self.validate_every == 0:
                model.eval()
                self.run_validation(step)
                model.train()

            # Logging
            self.logger.add_scalar("train/loss", loss, step)
            pbar.set_postfix({"loss": loss.item()})
