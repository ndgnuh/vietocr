from copy import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class Config:
    # Modelling specs
    lang: str
    type: str
    backbone: Union[str, Dict]
    head: Union[str, Dict]

    # Load weights
    weights: Optional[str] = None

    # Input specs
    image_height: int = 32
    image_min_width: int = 32
    image_max_width: int = 512

    # Optimization
    lr: float = 7e-4
    optimizer: Union[str, Dict] = "adam"
    lr_scheduler: Optional[Union[str, Dict]] = None

    # Training and validation data
    train_data: List = field(default_factory=list)
    val_data: List = field(default_factory=list)

    @classmethod
    def from_yaml(Config, file: str):
        import yaml

        with open(file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return Config(**config)


def initialize(namespace, config, attr=True, **extra_options):
    """Initialize something from configuration.

    TODO: explain `config`

    If config is a string...

    Args:
        namespace: The namespace to look up, can be a module, object or dictionary.
        config (Union[str, dict]): Config according to specs
        attr (bool): If True, use attribute to index the namespace,
            otherwise use indexing. default: True.
    """
    if isinstance(config, str):
        Class = getattr(namespace, config) if attr else namespace[config]
        return Class(**extra_options)
    else:
        config = copy(config)
        type_name = config.pop("type")
        Class = getattr(namespace, type_name) if attr else namespace[type_name]
        config.update(extra_options)
        return Class(**config)
