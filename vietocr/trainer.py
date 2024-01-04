import random
from copy import copy
from os import makedirs, path
from pprint import pformat
from typing import Dict, Optional

import torch
from lightning_fabric import Fabric
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from .configs import OcrConfig
from .data import build_dataloader
from .images import prepare_input
from .metrics import acc_full_sequence, acc_per_char
from .models import OcrModel
from .vocabs import get_vocab


class _StatList(list):
    def mean(self):
        return sum(self) / len(self)

    def reset(self):
        while len(self) > 0:
            self.pop(0)


def mean(lst):
    return sum(lst) / len(lst)


def loop_for_steps(loader: DataLoader, total_steps: int):
    """It's easier to control the training this way.

    Yields:
        step (int): The current training step
        epoch (int): The current training epoch
        batch: The data batch of samples
    """
    step = 1
    epoch = 0
    while step < total_steps:
        for i, data in enumerate(loader):
            yield step, epoch, data
            step = step + 1
            if step > total_steps:
                return
        epoch = epoch + 1


def _train_step(model: OcrModel, batch, optimizer, lr_scheduler) -> float:
    image, targets, target_lengths = batch

    # Forward
    optimizer.zero_grad()
    outputs = model(image)

    # Backward
    loss = model.compute_loss(outputs, targets, target_lengths)
    loss.backward()

    # Scheduling LR
    optimizer.step()
    lr_scheduler.step()
    return loss.item()


@torch.no_grad()
def _validate_step(model: OcrModel, batch, vocab) -> float:
    image, targets, target_lengths = batch
    outputs = model(image)
    loss = model.compute_loss(outputs, targets, target_lengths)

    # Decode and compute scores
    pr_probits = torch.softmax(outputs, dim=-1)
    batch_pr = pr_probits.argmax(dim=-1)
    batch_gt = targets
    acc_fs, acc_pc, predictions = [], [], []
    for pr, gt in zip(batch_pr, batch_gt):
        pr_s = vocab.decode(pr.cpu().numpy())
        gt_s = vocab.decode(gt.cpu().numpy())
        acc_fs.append(acc_full_sequence(pr_s, gt_s))
        acc_pc.append(acc_per_char(pr_s, gt_s))
        predictions.append({"pr": pr_s, "gt": gt_s})
    return dict(loss=loss.item(), acc_fs=acc_fs, acc_pc=acc_pc, predictions=predictions)


class OcrTrainer:
    """Trainer for OcrModel.

    Args:
        config (OcrConfig): The configuration
        name (Optional[str]): The experiement name, will be used for
            logging directory.
    """

    def __init__(self, config: OcrConfig, name: str = None):
        # Training specific validations
        self.validate_config(config)

        # +---------------------+
        # | Training scheduling |
        # +---------------------+
        self.print_every = config.print_every
        self.validate_every = config.validate_every
        self.total_steps = config.total_steps
        if self.print_every is None:
            self.print_every = max(1, self.validate_every // 5)

        # +------------------------------+
        # | Initialize model and encoder |
        # +------------------------------+
        vocab = get_vocab(config.vocab, config.type)
        vocab_size = len(vocab)
        model = OcrModel(
            vocab_size,
            backbone=config.backbone,
            head=config.head,
            image_height=config.image_height,
            image_min_width=config.image_min_width,
            image_max_width=config.image_max_width,
        )
        try:  # Try to load weights, if not notify and ignore
            weights = torch.load(config.weights, map_location="cpu")
            model.load_state_dict(weights)
            tqdm.write(f"Loaded weights from {config.weights}")
        except Exception as e:
            tqdm.write(f"Can't load weights, error: {e}")
        self.vocab = vocab
        self.model = model
        self.save_weights_path = config.save_weights

        # +------------------------+
        # | Initialize dataloaders |
        # +------------------------+
        augment = None
        preprocess = prepare_input(
            min_width=config.image_min_width,
            max_width=config.image_max_width,
            height=config.image_height,
            normalize=False,  # Collate fn will normalize the image
            transpose=False,  # Collate fn will transposes the image
        )
        train_loader = build_dataloader(
            config.train_data,
            preprocess=preprocess,
            augment=augment,
            vocab=vocab,
            **config.dataloader_options,
        )
        val_loader = build_dataloader(
            config.val_data,
            vocab=vocab,
            preprocess=preprocess,
            augment=None,
            **config.dataloader_options,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        # +----------------------+
        # | Initialize optimizer |
        # +----------------------+
        optimizer = copy(config.optimizer)
        optimizer_type = optimizer.pop("type")
        Optimizer = getattr(optim, optimizer_type)
        self.optimizer = Optimizer(self.model.parameters(), **optimizer)

        # +----------------------------+
        # | Lr scheduler, if available |
        # +----------------------------+
        if config.lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
        else:
            lr_scheduler = copy(config.lr_scheduler)
            lr_scheduler_type = lr_scheduler.pop("type")
            LrScheduler = getattr(optim.lr_scheduler, lr_scheduler_type)
            self.lr_scheduler = LrScheduler(self.optimizer, **lr_scheduler)

        # Logging
        hparams = {
            "num_training_batches": len(self.train_loader),
            "num_validation_batches": len(self.val_loader),
            "vocab_size": len(self.vocab),
        }
        self.logger = SummaryWriter()
        self.logger.add_hparams(hparams, {})

        # Load checkpoints if available
        # TODO: load_checkpoint
        # TODO: save_checkpoint

    def save_weights(self):
        if self.save_weights_path is None:
            return
        weights = self.model.state_dict()

        # Check for NaN values
        for k, v in weights.items():
            if v.isnan().any():
                tqdm.write("There is NaN value in weights, won't save")
                return

        # Store weights
        weight_dir = path.dirname(self.save_weights_path)
        if weight_dir != "":
            makedirs(weight_dir, exist_ok=True)
        torch.save(weights, self.save_weights_path)
        tqdm.write(f"Model weights saved to {self.save_weights_path}")

    def fit(self):
        # Unpack
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        model = self.model
        vocab = self.vocab

        # Scheduling
        total_steps = self.total_steps
        print_every = self.print_every
        validate_every = self.validate_every

        # Setup models
        fabric = Fabric()
        model, optimizer = fabric.setup(model, optimizer)
        loaders = fabric.setup_dataloaders(self.train_loader, self.val_loader)
        train_loader, val_loader = loaders

        # Stats
        train_losses = _StatList()

        # Data looping
        data_iter = loop_for_steps(train_loader, total_steps)
        data_iter = tqdm(data_iter, total=total_steps)
        tqdm.write(f"Number of training batches {len(train_loader)}")
        tqdm.write(f"Number of validation batches {len(val_loader)}")
        tqdm.write(f"Vocabulary size {len(vocab)}")

        # Start training loop
        for step, epoch, batch in data_iter:
            # Training step and logging
            loss = _train_step(model, batch, optimizer, lr_scheduler)
            self.logger.add_scalar("loss/train", loss, step)

            # Average logging
            train_losses.append(loss)
            train_loss = train_losses.mean()
            if step % print_every == 0:
                images = batch[0].cpu()
                self.logger.add_scalar("loss/train-avg", train_loss, step)
                self.logger.add_image("sample/inputs", make_grid(images, 2), 0)
                train_losses.reset()
                self.save_weights()

            # Validate and log results
            if step % validate_every == 0:
                # Collect metrics
                val_losses, acc_fs, acc_pc, predictions = [], [], [], []
                for batch in tqdm(val_loader, "Validation"):
                    metrics = _validate_step(model, batch, vocab=vocab)
                    val_losses.append(metrics["loss"])
                    acc_fs.extend(metrics["acc_fs"])
                    acc_pc.extend(metrics["acc_pc"])
                    predictions.extend(metrics["predictions"])

                # Only write out 5 examples max
                num_samples = min(len(predictions), 5)
                samples = random.choices(predictions, k=num_samples)

                # Logging
                tqdm.write(pformat(samples))
                self.logger.add_scalar("loss/val-avg", mean(val_losses), step)
                self.logger.add_scalar("other/acc-fs", mean(acc_fs), step)
                self.logger.add_scalar("other/acc-pc", mean(acc_pc), step)
                self.logger.add_scalar("loss/train-avg", train_loss, step)

            # More logging
            lr = lr_scheduler.get_last_lr()[0]
            image_width = batch[0].shape[-1]
            log_dict = dict(loss=loss, lr=lr)
            data_iter.set_description(f"Train #{step}/{total_steps}")
            data_iter.set_postfix(log_dict)
            self.logger.add_scalar("other/lr", lr, step)
            self.logger.add_scalar("other/image-width", image_width, step)

        # Save weights for the final time
        self.save_weights()

    def validate_config(self, config: OcrConfig):
        """Validate the configuration specifically for training purpose.

        The conditions are:
        - The path to save weights is not empty,
        - If the load weights path is specified, it must be an existing file.
        - The list of training datasets is not empty (if it is specified as
            a list in the configuration), or the training data is not None.

        Notes: update the docs when adding conditions

        Args:
            config (OcrConfig): The model configuration.
        """
        msg = "Path to save weights is empty, training is meaning less"
        assert config.save_weights is not None, msg

        msg = f"The weights {config.weights} does not exists"
        assert config.weights is None or path.isfile(config.weights)

        msg = "No training data, please check the configuration"
        if isinstance(config.train_data, list):
            assert len(config.train_data) > 0, msg
        else:
            assert config.train_data is not None
