from pytorch_lightning.lite import LightningLite
from functools import partial
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
from os import path
import torch

from . import losses
from .. import const
from ..tool.translate import build_model
from ..tool.stats import (
    AverageStatistic,
    MaxStatistic,
    TotalTimer,
    AverageTimer
)
from ..tool.utils import compute_accuracy
from ..loader.aug import default_augment
from ..loader.dataloader import build_dataloader


def cycle(dataloader, total_steps: int):
    step = 0
    previous_shape = None
    while step < total_steps:
        for batch in dataloader:
            step = step + 1
            yield step, batch

            # TODO: how to solve
            # empty cache because tensors with
            # different sizes will stack up
            # in the GPU, causing CUDA OOM


def basic_train_step(lite, model, batch, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    images, labels = batch
    ic(images.shape, labels.shape)
    outputs = model(images, labels)
    loss = criterion(outputs, labels)
    lite.backward(loss)
    optimizer.step()
    return loss


def adversarial_train_step(lite, model, batch, criterion, optimizer, epsilon=0.3):
    model.train()

    # Generating gradient on the input images
    # Using requires grad here because backward doesn't work in eval mode
    for p in model.parameters():
        p.requires_grad = False
    optimizer.zero_grad()
    images, labels = batch
    images.requires_grad = True
    outputs = model(images, labels)
    loss = criterion(outputs, labels)
    loss.backward()
    data_grad = images.grad
    for p in model.parameters():
        p.requires_grad = True

    # Generating adversarial examples
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Train on perturbed image
    optimizer.zero_grad()
    outputs = model(perturbed_images, labels)
    loss = criterion(outputs, labels)
    lite.backward(loss)
    optimizer.step()
    return loss


class Trainer(LightningLite):
    def __init__(self, config):
        if config.get('device', 'cuda'):
            accelerator = 'gpu'
        else:
            accelerator = None
        super().__init__(accelerator=accelerator)
        training_config = config['training']

        # Checkpoint and saving
        self.name = config['name']
        self.output_weights_path = path.join(
            const.weight_dir, self.name + ".pth")

        # Scheduling stuffs
        self.total_steps = training_config['total_steps']
        self.validate_every = training_config['validate_every']
        if isinstance(self.validate_every, float):
            self.validate_every = int(self.total_steps * self.validate_every)
        self.print_every = self.validate_every // 5

        # Models
        # Leave the device stuff to lightning
        self.model, self.vocab = build_model(config, move_to_device=False)
        self.criterion = losses.CrossEntropyLoss(vocab=self.vocab)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-09
        )
        self.lr_scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            total_steps=training_config['total_steps'],
            max_lr=training_config['learning_rate']
        )

        # Dataloaders
        build_dataloader_ = partial(
            build_dataloader,
            vocab=self.vocab,
            image_height=config['image_height'],
            image_min_width=config['image_min_width'],
            image_max_width=config['image_max_width'],
            batch_size=training_config.get('batch_size', 1),
            num_workers=training_config.get('num_workers', 1),
        )
        self.train_data = build_dataloader_(
            annotation_path=training_config['train_annotation'],
            transform=default_augment
        )
        self.validate_data = build_dataloader_(
            annotation_path=training_config['validate_annotation'],
            transform=None
        )

    def run(self):
        train_loader = self.setup_dataloaders(self.train_data)
        model, optimizer = self.setup(self.model, self.optimizer)

        metrics = {}

        train_loss = AverageStatistic()
        best_full_seq = MaxStatistic()
        gpu_time = TotalTimer()
        load_time = AverageTimer()

        train_iter = cycle(train_loader)

        # Avoid getattr calls in the loop
        print_every = self.print_every
        validate_every = self.validate_every
        lr_scheduler = self.lr_scheduler
        criterion = self.criterion

        # 1 indexing in this case is better
        # - don't have to check for step > 0
        # - don't have to align the "validate every" config for the last step
        for step in trange(1, self.total_steps + 1, desc="Training", dynamic_ncols=True):
            with load_time:
                batch = next(train_data)

            # Training step
            with gpu_time:
                loss = basic_train_step(
                    self,
                    model,
                    batch,
                    optimizer=optimizer,
                    criterion=criterion
                )
                adversarial_train_step(
                    self,
                    model,
                    batch,
                    optimizer=optimizer,
                    criterion=criterion
                )
            train_loss.append(loss.item())

            lr_scheduler.step()

            if step % validate_every == 0:
                metrics = self.validate()
                info = (
                    f"Validating",
                    f"Loss: {metrics['val_loss']:.3f}",
                    f"Full seq: {metrics['full_seq']:.3f}",
                    f"Per char: {metrics['per_char']:.3f}",
                )

                tqdm.write(" - ".join(info))

                # Check if new best
                new_best = best_full_seq.append(metrics['full_seq'])
                if new_best:
                    torch.save(model.state_dict(), self.output_weights_path)
                    tqdm.write(
                        f"Model weights saved to {self.output_weights_path}")

            if step % print_every == 0:
                mean_train_loss = train_loss.summarize()
                lr = optimizer.param_groups[0]['lr']

                info = (
                    f"Training: {step}/{self.total_steps}",
                    f"Loss: {mean_train_loss:.3f}",
                    f"Lr: {lr:.1e}",
                    f"Best full seq: {best_full_seq.summarize():.2f}",
                    f"Load time: {load_time.summarize() * 1000:.2f}ms",
                    f"GPU time: {gpu_time.summarize():.2f}s",
                )
                tqdm.write(" - ".join(info))

    @torch.no_grad()
    def validate(self):
        data = self.setup_dataloaders(self.validate_data)
        model = self.setup(self.model)
        model.eval()

        val_loss = AverageStatistic()
        full_seq = AverageStatistic()
        per_char = AverageStatistic()
        criterion = self.criterion

        for images, labels in tqdm(data, desc="Validating", dynamic_ncols=True):
            outputs = model(images, labels)

            # Validation loss
            loss = criterion(outputs, labels).item()
            val_loss.append(loss)

            # Validation accuracies
            confidences, predictions = outputs.topk(k=1, dim=-1)
            predictions = predictions.squeeze(-1)
            pr_sents = self.vocab.batch_decode(predictions.tolist())
            gt_sents = self.vocab.batch_decode(labels.tolist())

            full_seq.append(compute_accuracy(
                pr_sents, gt_sents, 'full_sequence'))
            per_char.append(compute_accuracy(pr_sents, gt_sents, 'per_char'))

        metrics = dict(
            val_loss=val_loss.summarize(),
            full_seq=full_seq.summarize(),
            per_char=per_char.summarize(),
        )
        return metrics

    def train(self):
        self.run()
