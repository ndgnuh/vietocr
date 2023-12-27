from copy import deepcopy
from typing import Dict, Union

import torch
from torch import Tensor, nn

from . import backbone_fvtr, backbone_mlp_mixer, backbone_resnet, heads
from .losses import CrossEntropyLoss, CTCLoss
from .optim import CosineWWRD

# +-------------------------------------------+
# | Assert so that the LSP does not complaint |
# +-------------------------------------------+
assert CosineWWRD
assert CrossEntropyLoss
assert CTCLoss

# +-------------------------------------+
# | Dictionary of all backbones by name |
# +-------------------------------------+
BACKBONES = {}
BACKBONES.update(backbone_mlp_mixer.MODULES)
BACKBONES.update(backbone_fvtr.MODULES)
BACKBONES.update(backbone_resnet.MODULES)

# +--------------------------------------------+
# | Dictionary of all prediction heads by name |
# +--------------------------------------------+
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


class OcrModel(nn.Module):
    """Entrypoint for the initializing OcrModel."""

    def __init__(
        self,
        vocab_size: int,
        backbone: Union[Dict, str],
        head: Union[Dict, str],
        image_height: int,
        image_min_width: int,
        image_max_width: int,
        model_type: str = "ctc",
    ):
        super().__init__()
        backbone_config = backbone
        head_config = head

        # Initialize backbone
        # All backbone must have `get_hidden_size` method
        self.backbone = _initialize(
            BACKBONES,
            backbone_config,
            image_height=image_height,
            image_min_width=image_min_width,
            image_max_width=image_max_width,
        )
        with torch.no_grad():
            image = torch.rand(1, 3, 32, 128)
            hidden_size = self.backbone(image).shape[-1]
            del image

        # Prediction head
        # All prediction head must accept `vocab_size` and `hidden_size` kwargs
        self.head = _initialize(
            HEADS,
            head_config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.compute_loss = CTCLoss() if model_type == "ctc" else CrossEntropyLoss()

        # Store stuffs
        self.backbone_config = backbone_config
        self.head_config = head_config
        self.vocab_size = vocab_size
        self.exporting = False

    def forward(self, images: Tensor, post_process: bool = False) -> Tensor:
        # Forward
        x = self.backbone(images)
        x = self.head(x)

        # Output shape agreement: [batch size, length, vocab size]
        B, L, C = x.shape
        assert B == images.shape[0]
        assert C == self.vocab_size

        # Output immidiately, training case
        if not post_process and not self.exporting:
            return x

        scores, indices = self.post_process(x)
        if self.exporting:
            # ONNX Export case, x is not needed
            return scores, indices
        else:
            # Validation, both hidden features and the outputs are needed
            return scores, indices, x

    def post_process(self, x: Tensor):
        probs = torch.softmax(x, dim=-1)
        scores, indices = torch.max(probs, dim=-1)
        return scores, indices

    def export_onnx(self, example_inputs: Tensor, output_file: str, **options):
        self.exporting = True
        kwargs = {}
        kwargs["input_names"] = ["images"]
        kwargs["dynamic_axes"] = {"images": [0, 3]}
        kwargs["do_constant_folding"] = True
        kwargs.update(options)
        torch.onnx.export(self, example_inputs, output_file, **kwargs)
        self.exporting = False


# +--------------+
# | Conveniences |
# +--------------+


def get_available_backbones():
    return BACKBONES


def get_available_heads():
    return HEADS
