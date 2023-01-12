from pytorch_lightning.lite import LightningLite
from functools import partial
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
from itertools import cycle
import torch

from ..tool.translate import build_model
from ..loader.aug import default_augment
from ..loader.dataloader import build_dataloader


class Trainer(LightningLite):
    def __init__(self, config):
        if config.get('device', 'cuda'):
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'
        super().__init__()
        training_config = config['training']

        # Scheduling stuffs
        self.total_steps = training_config['total_steps']
        self.validate_every = training_config['validate_every']
        if isinstance(self.validate_every, float):
            self.validate_every = int(self.total_steps * self.validate_every)
        self.print_every = self.validate_every // 5


        self.model, self.vocab = build_model(config)
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
            vocab = self.vocab,
            image_height = config['image_height'],
            image_min_width = config['image_min_width'],
            image_max_width = config['image_max_width'],
            batch_size = training_config.get('batch_size', 1),
            num_workers = training_config.get('num_workers', 1),
        )

        self.train_data = build_dataloader_(
            annotation_path=training_config['train_annotation'],
            transform = default_augment
        )
        self.validate_data = build_dataloader_(
            annotation_path=training_config['train_annotation'],
            transform = None
        )

    def run(self):
        train_data = self.setup_dataloaders(self.train_data)
        model, optimizer = self.setup(self.model, self.optimizer)

        train_data = cycle(train_data)
        for step in trange(self.total_steps, desc="Training", dynamic_ncols=True):
            images, labels = next(train_data)
            outputs = model(images)
            
            if step % self.validate_every == 0 and step > 0:
                self.validate()


    @torch.no_grad()
    def validate(self):
        data = self.setup_dataloaders(self.validate_data)
        for batch in tqdm(data, desc="Validating", dynamic_ncols=True):
            pass

    def train(self):
        self.run()
