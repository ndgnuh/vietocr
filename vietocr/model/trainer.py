from pytorch_lightning.lite import LightningLite
from functools import partial

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


        self.model, self.vocab = build_model(config)

        build_dataloader_ = partial(
            build_dataloader,
            vocab = self.vocab,
            image_height = config['image_height'],
            image_min_width = config['image_min_width'],
            image_max_width = config['image_max_width'],
            batch_size = training_config.get('batch_size', 1),
            num_workers = training_config.get('num_workers', 1)
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
        pass

    def train(self):
        self.run()
