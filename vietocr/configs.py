"""This module provides configuration schema for the package.

The module also acts as a glue module to provide init configurations
for other functions and classes, e.g. DataModule, DataLoader, etc.

The ideal usage would allow the config to be either:

- a string: this string will be used to look up a namespace and the result
  will be the callee function/class, the function will be called with its
  default arguments.
- a dictionary with "type" key, the "type" key will be used to search the
  namespace for the correct callee function/class, other keys in the dictionary
  will be used as the keyword arguments for the callee function.

The package should also support other types of initialization, not just from
configurations. So all the classes should receive meaningful arguments instead
of just "config", this is a WIP.
"""
from copy import copy
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Union


@dataclass
class OcrConfig:
    """OCR configuration, this configuration file might be reworked in the later version."""

    # Modelling specs
    vocab: str
    """Vocabulary language."""

    type: str
    """Vocabulary type."""

    backbone: Union[str, Dict]
    """The backbone configuration."""

    head: Union[str, Dict]
    """The prediction head configuration, must be appropriate for the vocabulary type."""

    # Weights saving, loading, checkpoint is for trainer
    weights: Optional[str] = None
    """Path to load weights."""

    checkpoints: Optional[str] = None
    """This option is unused."""

    save_weights: Optional[str] = None
    """Path to save model weights."""

    save_checkpoints: Optional[str] = None
    """This option is unused."""

    save_onnx: Optional[str] = None
    """Path to save ONNX weights."""

    # Input specs
    image_height: int = 32
    """Image height."""

    image_min_width: int = 32
    """Minimum image width when resizing."""

    image_max_width: int = 512
    """Maximum image width when resizing."""

    # Optimization
    lr: float = 7e-4
    optimizer: Union[str, Dict] = "adam"
    lr_scheduler: Optional[Union[str, Dict]] = None

    # Training and validation data
    train_data: List = field(default_factory=list)
    """Training data specification."""

    val_data: List = field(default_factory=list)
    """Validation data specification."""

    test_data: List = field(default_factory=list)
    """Test data specification."""

    batch_size: int = 1
    """Training batch size."""

    num_workers: int = 0
    """Number of data loader workers."""

    shuffle: bool = False
    """If true, the data should be shuffled."""

    # Trainin schedulings
    total_steps: int = 100_000
    """Total training steps."""

    validate_every: int = 4000
    """Validate interval steps."""

    print_every: Optional[int] = None
    """Logging interval steps."""

    @classmethod
    def from_yaml(Config, file: str):
        """Load a config from yaml file.

        Args:
            file: path to config file, the file is a yaml file.

        Returns:
            A :class:`OcrConfig` object with data loaded from the input file.
        """
        import yaml

        with open(file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return Config(**config)

    @property
    def dataloader_options(self) -> dict:
        """Keyword arguments for torch's DataLoader."""
        return dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def to_dict(self) -> dict:
        """Convert the config object to a dictionary."""
        return asdict(self)


def ez_init(
    namespace: Dict[str, Callable],
    config: Union[Dict, str],
    **extra_options,
):
    """Initialize something from configuration.

    If config is a string:

    - a string: this string will be used to look up a namespace and the result
      will be the callee function/class, the function will be called with its
      default arguments.
    - a dictionary with "type" key, the "type" key will be used to search the
      namespace for the correct callee function/class, other keys in the dictionary
      will be used as the keyword arguments for the callee function.

    Parameters:
        namespace (Dict[str, Callable]): The namespace dictionary to look up.
        config (Union[str, dict]): The config str or dictionary.
        **extra_options (bool): Extra keyword arguments to pass to the callee.
    """
    if isinstance(config, str):
        Callee = namespace[config]
        return Callee(**extra_options)
    else:
        config = copy(config)
        type_name = config.pop("type")
        Callee = namespace[type_name]
        config.update(extra_options)
        return Callee(**config)
