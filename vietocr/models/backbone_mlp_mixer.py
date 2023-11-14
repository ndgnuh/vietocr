import math
from typing import List

import torch
from torch import nn

from .reversible import Reversible


class PositionEncoding(nn.Module):
    def __init__(self, hidden_size: int, height: int):
        super().__init__()
        self.row_embs = nn.Parameter(torch.zeros(1, hidden_size, height, 1))

        # shape: [D / 2]
        invfreq = torch.arange(0, hidden_size, 2)
        invfreq = invfreq * (-math.log(10000) / hidden_size)
        self.register_buffer("iF", invfreq)

    def forward(self, image):
        # Notation:
        # - W: width
        # - H: height
        # - D: hidden size (channels)

        # [W]
        device = image.device
        dtype = image.dtype
        pos = torch.arange(image.shape[-1], device=device, dtype=dtype)

        # [W, 1] * [1, D / 2] -> [W, D / 2]
        pos = pos[:, None] * self.iF
        sin = torch.sin(pos)
        cos = torch.cos(pos)

        # [W, D] -> [1, D, 1, W]
        col_embs = torch.cat([sin, cos], dim=1)
        col_embs = col_embs.transpose(1, 0)[None, :, None, :]

        # [1, D, H, 1] + [1, D, 1, W] -> [1, D, H, W]
        row_embs = self.row_embs
        pos_embs = row_embs + col_embs
        return pos_embs


class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        image_height: int,
        image_channel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        kwargs = dict(kernel_size=(4, 3), stride=(3, 2))
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(image_channel, hidden_size, **kwargs),
            nn.SELU(True),
        )
        with torch.no_grad():
            img = torch.rand(1, 3, image_height, 10)
            num_hpatch = self.patch_embedding(img).shape[-2]

        self.position_encodings = PositionEncoding(hidden_size, num_hpatch)
        self.dropout = nn.Dropout(dropout)
        self.num_vertical_patches = num_hpatch

    def forward(self, image):
        embeddings = self.patch_embedding(image)
        embeddings = self.position_encodings(embeddings) + embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class PermuteDim(nn.Module):
    def __init__(self, src: str, dst: str):
        super().__init__()
        self.perm = [src.index(s) for s in dst]
        self.extra_repr = lambda: f"from='{src}', to='{dst}', perm={self.perm}"

    def forward(self, x):
        x = x.permute(self.perm)
        return x


class MLPMixerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_vertical_patches: int, expansion: int = 4):
        super().__init__()
        kwargs = dict(kernel_size=(1, 3), padding=(0, 1), groups=hidden_size)
        self.h_mixer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * expansion, **kwargs),
            nn.SELU(True),
            nn.Conv2d(hidden_size * expansion, hidden_size, **kwargs),
        )

        self.v_mixer = nn.Sequential(
            PermuteDim("bchw", "bcwh"),
            nn.Linear(num_vertical_patches, num_vertical_patches * expansion),
            nn.SELU(True),
            nn.Linear(num_vertical_patches * expansion, num_vertical_patches),
            PermuteDim("bcwh", "bchw"),
        )

        self.c_mixer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.SELU(True),
            nn.Linear(hidden_size * expansion, hidden_size),
        )
        self.to_vec = PermuteDim("bchw", "bhwc")
        self.to_img = PermuteDim("bhwc", "bchw")
        self.norm_vh = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # Spatial norm
        residual = x
        x = self.to_vec(x)
        x = self.norm_vh(x)
        x = self.to_img(x)

        # Spatial mixers
        x = self.h_mixer(x)
        x = self.v_mixer(x) + residual

        residual = x
        # Normal MLP
        x = self.to_vec(x)
        x = self.norm_c(x)
        x = self.c_mixer(x)
        x = self.to_img(x) + residual
        return x


class MiddleProjection(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kwargs = dict(kernel_size=(3, 1), padding=(1, 0), stride=(2, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.to_vec = PermuteDim("bchw", "bhwc")
        self.norm = nn.LayerNorm(out_channels)
        self.to_img = PermuteDim("bhwc", "bchw")


class FinalProjection(nn.Module):
    def __init__(self, in_channels, out_channels, dropout: float = 0.1):
        super().__init__()
        self.output = nn.Sequential(
            PermuteDim("bcw", "bwc"),
            nn.Linear(in_channels, out_channels),
            nn.SELU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.mean(dim=-2)
        x = self.output(x)
        return x


class MLPMixerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_vertical_patches: int,
        num_layers: int,
        final: bool,
    ):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            block = MLPMixerBlock(in_channels, num_vertical_patches)
            self.blocks.append(block)

        if final:
            self.projection = FinalProjection(in_channels, out_channels)
        else:
            self.projection = MiddleProjection(in_channels, out_channels)

        # Calculate next stats
        with torch.no_grad():
            img = torch.rand(1, in_channels, num_vertical_patches, 10)
            self.num_vertical_patches = self(img).shape[-2]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        num_layers: List[int],
        image_height: int,
    ):
        super().__init__()
        patch_embeddings = PatchEmbeddings(hidden_sizes[0], image_height)
        self.stages = nn.ModuleList([patch_embeddings])

        num_stages = len(num_layers)
        num_vertical_patches = patch_embeddings.num_vertical_patches
        finals = [False] * (num_stages - 1) + [True]
        for i in range(num_stages):
            stage = MLPMixerStage(
                in_channels=hidden_sizes[i],
                out_channels=hidden_sizes[i + 1],
                num_vertical_patches=num_vertical_patches,
                num_layers=num_layers[i],
                final=finals[i],
            )
            num_vertical_patches = stage.num_vertical_patches
            self.stages.append(stage)
        self.last_hidden_size = hidden_sizes[-1]

    def get_hidden_size(self):
        return self.last_hidden_size

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


def mlp_mixer_tiny(image_height: int, **opts):
    num_layers = [3, 6, 3]
    hidden_sizes = [64, 128, 256, 192]
    return MLPMixer(
        hidden_sizes=hidden_sizes,
        num_layers=num_layers,
        image_height=image_height,
        **opts,
    )


MODULES = {
    "mlp_mixer_tiny": mlp_mixer_tiny,
}
