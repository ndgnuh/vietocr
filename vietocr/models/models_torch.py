from copy import deepcopy
from typing import Dict, Union

from torch import Tensor, nn

from . import backbone_mlp_mixer, heads

BACKBONES = {}
BACKBONES.update(backbone_mlp_mixer.MODULES)

HEADS = {}
HEADS.update(heads.MODULES)


def _initialize(namespace: Dict, config: Union[Dict, str], **extra_kwargs):
    if isinstance(config, dict):
        config = deepcopy(config)
        name = config.pop("name")
        config.update(extra_kwargs)
        return namespace[name](**config)
    else:
        return namespace[config](**extra_kwargs)


class OCRModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        backbone_config: Union[Dict, str],
        head_config: Union[Dict, str],
    ):
        super().__init__()
        # Initialize backbone
        # All backbone must have `get_hidden_size` method
        self.backbone = _initialize(BACKBONES, backbone_config)
        hidden_size = self.backbone.get_hidden_size()

        # Prediction head
        # All prediction head must accept `vocab_size` and `hidden_size` kwargs
        self.head = _initialize(
            HEADS,
            head_config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        # Store stuffs
        self.backbone_config = backbone_config
        self.head_config = head_config
        self.vocab_size = vocab_size

    def forward(self, images: Tensor) -> Tensor:
        # Forward
        x = self.backbone(images)
        x = self.head(x)

        # Output shape agreement: [batch size, length, vocab size]
        B, L, C = x.shape
        assert B == images.shape[0]
        assert C == self.vocab_size

        # Output
        return x


# Conveniences
def get_available_backbones():
    return BACKBONES


def get_available_heads():
    return HEADS
