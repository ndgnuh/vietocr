from ..loader.dataloader import build_dataloader
from ..loader.aug import default_augment
from ..tool.utils import compute_accuracy
from ..tool.stats import (
    AverageStatistic,
    MaxStatistic,
    TotalTimer,
    AverageTimer
)
from ..tool.translate import build_model
from .. import const
from . import losses
from pytorch_lightning.lite import LightningLite
from functools import partial
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
from os import path
from dataclasses import dataclass
import torch
import random
import os

# fix: https://github.com/ndgnuh/vietocr/issues/2
# ref: https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'


def get_logger(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        print("Install tensorboard to log")
        return None


def cycle(total_steps, dataloader):
    step = 0
    while True:
        for batch in dataloader:
            step = step + 1
            yield step, batch

            if step == total_steps:
                return


def basic_train_step(lite, model, batch, criterion, optimizer, teacher_forcing: bool = False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images, labels, target_lengths = batch
    outputs = model(images, labels, teacher_forcing=teacher_forcing)
    loss = criterion(outputs, labels, target_lengths)
    lite.backward(loss)
    optimizer.step()
    return loss


def adversarial_train_step(lite, model, batch, criterion, optimizer, epsilon=0.05, teacher_forcing: bool = False):
    model.train()

    # Generating gradient on the input images
    # Using requires grad here because backward doesn't work in eval mode
    for p in model.parameters():
        p.requires_grad = False
    optimizer.zero_grad()
    images, labels, target_lengths = batch
    images.requires_grad = True
    outputs = model(images, labels)
    loss = criterion(outputs, labels, target_lengths)
    loss.backward()
    data_grad = images.grad
    for p in model.parameters():
        p.requires_grad = True
    images.requires_grad = False

    # Generating adversarial examples
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Train on perturbed image
    optimizer.zero_grad()
    outputs = model(perturbed_images, labels, teacher_forcing=teacher_forcing)
    loss = criterion(outputs, labels, target_lengths) * 0.05
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
            const.weight_dir,
            f"{self.name}.pth"
        )
        self.logger = get_logger(
            path.join(const.log_dir, self.name)
        )

        # Scheduling stuffs
        self.total_steps = training_config['total_steps']
        self.validate_every = training_config['validate_every']
        if isinstance(self.validate_every, float):
            self.validate_every = int(self.total_steps * self.validate_every)
        self.print_every = training_config.get(
            'print_every',
            max(self.validate_every // 5, 1))
        self.tfs = TeacherForcingScheduler(
            start_step=training_config.get('teacher_forcing_start', 0),
            end_step=training_config.get(
                'teacher_forcing_end',
                50000
            ),
            p0=training_config.get('teacher_forcing_max_prob', 1)
        )

        # Models
        # Leave the device stuff to lightning
        self.model, self.vocab = build_model(config, move_to_device=False)
        if config['type'] == 's2s':
            self.criterion = losses.CrossEntropyLoss(vocab=self.vocab)
        else:
            self.criterion = losses.CTCLoss(vocab=self.vocab)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-09
        )
        self.lr_scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            total_steps=training_config['total_steps'],
            max_lr=training_config['learning_rate'],
            pct_start=0.1,
        )

        # Freezing
        frozen = training_config.get("freeze", [])
        if len(frozen) > 0:
            print("=" * 10 + " Freezing params " + "=" * 10)
            for name, param in self.model.named_parameters():
                for fname in frozen:
                    if name.startswith(fname):
                        param.requires_grad = False
                        print(f"\t* Freezing {name}")

        # Dataloaders
        build_dataloader_ = partial(
            build_dataloader,
            vocab=self.vocab,
            image_height=config['image_height'],
            image_min_width=config['image_min_width'],
            image_max_width=config['image_max_width'],
            batch_size=training_config.get('batch_size', 1),
            num_workers=training_config.get('num_workers', 1),
            letterbox=config['image_letterbox'],
            shift_target=True if config['type'] == 's2s' else False
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
        train_data = self.setup_dataloaders(self.train_data)
        print("Number of training batches:", len(train_data))
        print("Number of validation batches:", len(self.validate_data))
        model, optimizer = self.setup(self.model, self.optimizer)

        metrics = {}

        train_loss = AverageStatistic()
        best_full_seq = MaxStatistic()
        gpu_time = TotalTimer()

        # Avoid getattr calls in the loop
        print_every = self.print_every
        validate_every = self.validate_every
        lr_scheduler = self.lr_scheduler
        criterion = self.criterion
        tf_scheduler = self.tfs

        # 1 indexing in this case is better
        # - don't have to check for step > 0
        # - don't have to align the "validate every" config for the last step
        pbar = trange(1, self.total_steps + 1,
                      desc="Training",
                      dynamic_ncols=True)
        data_gen = cycle(self.total_steps, train_data)
        previouse_w = None
        step = 0
        # Use two loops to release CUDA cache
        # when the image width is changed
        while step < self.total_steps:
            while True:
                step, batch = next(data_gen)
                w = batch[0].shape[-1]  # image width

                # Training step
                with gpu_time:
                    teacher_forcing = tf_scheduler.step()
                    loss = basic_train_step(
                        self,
                        model,
                        batch,
                        optimizer=optimizer,
                        criterion=criterion,
                        teacher_forcing=teacher_forcing,
                    )
                    # adversarial_train_step(
                    #     self,
                    #     model,
                    #     batch,
                    #     optimizer=optimizer,
                    #     criterion=criterion,
                    #     teacher_forcing=teacher_forcing,
                    # )
                train_loss.append(loss.item())

                lr_scheduler.step()

                if step % validate_every == 0:
                    metrics = self.validate()
                    for k, v in metrics.items():
                        self.log(f"val/{k}", v, step)
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
                        torch.save(model.state_dict(),
                                   self.output_weights_path)
                        tqdm.write(
                            f"Model weights saved to {self.output_weights_path}")

                lr = optimizer.param_groups[0]['lr']
                self.log("lr", lr, step)
                self.log("loss/train", loss.item(), step)
                self.log("teacher-forcing", tf_scheduler.current_ratio(), step)
                self.log("image width", w, step)

                if step % print_every == 0:
                    mean_train_loss = train_loss.summarize()
                    lr = optimizer.param_groups[0]['lr']

                    info = (
                        f"Training: {step}/{self.total_steps}",
                        f"Loss: {mean_train_loss:.3f}",
                        f"LR: {lr:.1e}",
                        f"TFR: {tf_scheduler.current_ratio():.2f}",
                        f"Best full seq: {best_full_seq.summarize():.2f}",
                        f"GPU time: {gpu_time.summarize():.2f}s",
                        f"Width: {batch[0].shape[-1]}",
                    )
                    tqdm.write(" - ".join(info))
                pbar.update(1)

                # Break from inner loop
                if previouse_w != w:
                    mean_train_loss = train_loss.summarize()
                    lr = optimizer.param_groups[0]['lr']

                    info = (
                        f"Training: {step}/{self.total_steps}",
                        f"Loss: {mean_train_loss:.3f}",
                        f"LR: {lr:.1e}",
                        f"TFR: {tf_scheduler.current_ratio():.2f}",
                        f"Best full seq: {best_full_seq.summarize():.2f}",
                        f"GPU time: {gpu_time.summarize():.2f}s",
                        f"Width: {batch[0].shape[-1]}",
                    )
                    tqdm.write(" - ".join(info))
                    torch.cuda.empty_cache()
                    previouse_w = w
                    break

    def log(self, tag, value, step):
        logger = self.logger
        if logger is None:
            return
        logger.add_scalar(tag, value, step)

    @torch.no_grad()
    def validate(self):
        data = self.setup_dataloaders(self.validate_data)
        model = self.setup(self.model)
        model.eval()

        val_loss = AverageStatistic()
        full_seq = AverageStatistic()
        per_char = AverageStatistic()
        criterion = self.criterion

        all_gts = []
        all_prs = []

        torch.cuda.empty_cache()
        for batch in tqdm(data, desc="Validating", dynamic_ncols=True):
            images, labels, target_lengths = batch

            outputs = model(images, labels)

            # Validation loss
            loss = criterion(outputs, labels, target_lengths).item()
            val_loss.append(loss)

            # Validation accuracies
            confidences, predictions = outputs.topk(k=1, dim=-1)
            predictions = predictions.squeeze(-1)
            pr_sents = self.vocab.batch_decode(predictions.tolist())
            gt_sents = self.vocab.batch_decode(labels.tolist())
            full_seq.append(compute_accuracy(
                pr_sents, gt_sents, 'full_sequence'))
            per_char.append(compute_accuracy(pr_sents, gt_sents, 'per_char'))

            # Predictions
            all_gts.extend(gt_sents)
            all_prs.extend(pr_sents)

        # Randomly print 5 of the PR-GT
        n = len(all_gts)
        idx = random.choices(range(n), k=5)
        info = [
            f"PR: {all_prs[i]}\nGT: {all_gts[i]}"
            for i in idx
        ]
        tqdm.write("\n~~~~~~\n".join(info))

        metrics = dict(
            val_loss=val_loss.summarize(),
            full_seq=full_seq.summarize(),
            per_char=per_char.summarize(),
        )
        torch.cuda.empty_cache()
        return metrics

    def train(self):
        self.run()


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
        return min(max(p, 0), 1)

    def step(self):
        p = self.current_ratio()
        self._step = self._step + 1
        return random.random() <= p
