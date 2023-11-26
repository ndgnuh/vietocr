import math
from functools import partial
from typing import List

import torch
from torch import nn


class PositionEncoding2D(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # shape: [D / 2]
        invfreq = torch.arange(0, hidden_size, 2)
        invfreq = invfreq * (-math.log(10000) / hidden_size)
        self.register_buffer("iF", invfreq.reshape(1, -1, 1, 1))
        self.cache = None

    def sincos(self, x):
        return torch.cat([x.sin(), x.cos()], dim=1)

    def forward(self, image):
        # == cached embedding ==
        if self.cache is not None:
            return self.cache

        # == Image: [B, C, H, W] ==
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype

        # == Positions ==
        pos_h = torch.arange(H, device=device, dtype=dtype).reshape(1, 1, H, 1)
        pos_w = torch.arange(W, device=device, dtype=dtype).reshape(1, 1, 1, W)

        # == Position embeddings ==
        embed_h = self.sincos(pos_h * self.iF)
        embed_w = self.sincos(pos_w * self.iF)
        self.cache = embed = embed_h + embed_w
        return embed


class PermuteDim(nn.Module):
    def __init__(self, src: str, dst: str):
        super().__init__()
        self.perm = [src.index(s) for s in dst]
        self.extra_repr = lambda: f"from='{src}', to='{dst}', perm={self.perm}"

    def forward(self, x):
        x = x.permute(self.perm)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        # +-------------+
        # | Convolution |
        # +-------------+
        kernel_size = (3, 3) if stride == 1 else (3, 1)
        padding = (1, 1) if stride == 1 else (1, 0)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=(stride, 1),
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

        # +---------------------+
        # | Residual connection |
        # +---------------------+
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1))
        else:
            self.residual = nn.Identity()

        # +---------------------+
        # | Residual activation |
        # +---------------------+
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.residual(x) + self.conv(x)
        x = self.relu(x)
        return x


class ResnetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        hconv: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                block = ResidualBlock(
                    in_channels,
                    out_channels,
                    stride=2,
                    hconv=hconv,
                )
            else:
                block = ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    hconv=True,
                )

            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Resnet(nn.Module):
    def __init__(
        self,
        channels: List[int],
        num_layers: List[int],
        hconv: List[bool],
    ):
        super().__init__()
        # +------------+
        # | Stem layer |
        # +------------+
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, padding=1),
            nn.InstanceNorm2d(channels[0]),
            nn.ReLU(True),
            nn.Conv2d(channels[0], channels[0], (3, 1), (2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(channels[0]),
        )

        # +---------------+
        # | Resnet stages |
        # +---------------+
        self.stages = nn.ModuleList()
        for i in range(len(num_layers)):
            h1 = channels[i]
            h2 = channels[i + 1]
            n = num_layers[i]
            hc = hconv[i]
            stage = ResnetStage(h1, h2, n, hc)
            self.stages.append(stage)

        # +--------------------------+
        # | Output size API function |
        # +--------------------------+
        self.final_hidden_size = channels[-1]
        print(self)

    def get_hidden_size(self):
        return self.final_hidden_size

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        b, c, h, w = 0, 1, 2, 3
        x = x.permute(b, w, h, c).flatten(1, 2)
        return x


class ScaleDotAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.qkv = nn.Linear(H, H * 3, bias=False)
        self.out = nn.Linear(H, H, bias=False)
        self.scale = 1 / math.sqrt(H)

    def forward(self, x):
        # == QKV forward ==
        q, k, v = self.qkv(x).reshape(*x.shape, 3).unbind(-1)

        # == Attention matrix ==
        qk = q.matmul(k.transpose(-1, -2))
        qk = torch.softmax(qk * self.scale, dim=-1)

        # == Output context ==
        ctx = qk.matmul(v)
        out = self.out(ctx)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, channels, *args):
        super().__init__()
        self.norm_atn_1 = nn.LayerNorm(channels)
        self.atn_1 = ScaleDotAttention(channels)
        self.norm_atn_2 = nn.LayerNorm(channels)
        self.atn_2 = ScaleDotAttention(channels)
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(True),
            nn.Linear(channels * 4, channels),
        )

        self.permute_1 = PermuteDim("bchw", "bwhc")
        self.permute_2 = PermuteDim("bwhc", "bhwc")
        self.permute_3 = PermuteDim("bhwc", "bchw")

    def forward(self, x):
        # Width token attention
        x = self.permute_1(x)
        x = x + self.atn_1(self.norm_atn_1(x))

        # Height token attention
        x = self.permute_2(x)
        x = x + self.atn_2(self.norm_atn_2(x))

        # MLP and back to image
        x = x + self.mlp(self.norm_mlp(x))
        x = self.permute_3(x)
        return x


class MixedResnetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: List[type],
        final: bool = False,
    ):
        super().__init__()
        # +-----------------------------+
        # | Build blocks for each stage |
        # +-----------------------------+
        blocks = []
        for i, Layer in enumerate(layers):
            layer = Layer(in_channels, in_channels)
            blocks.append(layer)
        self.blocks = nn.Sequential(*blocks)

        # Patch merging layer or final output layer
        if final:
            self.project = nn.Identity()
        else:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (2, 1), (2, 1)),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
            )

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        x = self.project(x)
        return x


class MixedResnet(nn.Module):
    def __init__(self, hidden_sizes: List[int], layers: List[List[type]], **k):
        super().__init__()
        # All stages
        stages = []

        # +------+
        # | Stem |
        # +------+
        stem = nn.Sequential(
            nn.Conv2d(3, hidden_sizes[0], (4, 2), stride=(4, 2)),
            nn.InstanceNorm2d(hidden_sizes[0]),
            nn.ReLU(True),
        )
        stages.append(stem)

        # +--------------+
        # | Mixed stages |
        # +--------------+
        for i in range(len(layers)):
            in_channels = hidden_sizes[i]
            out_channels = hidden_sizes[i + 1]
            stage = MixedResnetStage(in_channels, out_channels, layers[i])
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        # +---------------------------------+
        # | API function to get hidden size |
        # +---------------------------------+
        def get_hidden_size(hidden_size=hidden_sizes[-1]):
            return hidden_size

        self.get_hidden_size = get_hidden_size

    def forward(self, x):
        x = self.stages(x)
        b, c, h, w = 0, 1, 2, 3
        x = x.permute(b, w, h, c).flatten(1, 2)
        return x


resnet18 = partial(
    Resnet,
    channels=[32, 48, 64, 196, 256],
    num_layers=[2, 2, 2, 2],
    hconv=[True, True, False, False],
)

tr_resnet18 = partial(
    MixedResnet,
    hidden_sizes=[64, 128, 256, 192],
    layers=[
        [ResidualBlock] * 3,
        [ResidualBlock] * 3 + [TransformerBlock] * 3,
        [TransformerBlock] * 3,
    ],
)
# resnet18 = partial(Resnet, channels=[64, 128, 256, 512, 512], num_layers=[2, 2, 2, 2])

MODULES = {"resnet18": resnet18, "tr_resnet18": tr_resnet18}

if __name__ == "__main__":
    import time
    import timeit

    model = tr_resnet18()
    img = torch.rand(1, 3, 32, 128)
    model = model.eval()
    a = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            model(img)
    b = time.perf_counter()
    print("Time", (b - a) / 10, "s")
