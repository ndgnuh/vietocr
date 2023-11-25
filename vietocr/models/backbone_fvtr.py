# This version SHOULD be exportable...

import math
from functools import cached_property
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ToFiber(nn.Module):
    @cached_property
    def permutation(self):
        b, c, h, w = range(4)
        return (b, w, h, c)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(self.permutation)
        x = x.flatten(1, 2)
        return x, (H, W)


class ToImage(nn.Module):
    @cached_property
    def permutation(self):
        b, l, c = range(3)
        return (b, c, l)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        x = x.permute(self.permutation)
        x = x.reshape(B, C, W, H)
        x = x.transpose(-1, -2)
        return x


def skew(orig_x, padding_value):
    """shift every row 1 step to right converting columns into diagonals"""
    x = orig_x
    rest, H, W = x.shape[:-2], x.size(-2), x.size(-1)
    x = F.pad(x, (0, H), value=padding_value)
    x = x.reshape(*rest, -1)  # B x C x ML+MM+M
    x = x[..., :-H]  # B x C x ML+MM
    x = x.reshape(*rest, H, H + W - 1)  # B x C, M x L+M
    x = x[..., (W // 2) : -(W // 2) + (W % 2 - 1)]
    return x


class LocalAttentionMaskProvider2d(nn.Module):
    def __init__(self, locality):
        super().__init__()
        self.locality = locality
        self.cache = {}

    @torch.no_grad()
    def forward(self, x):
        key = (x.size(-2), x.size(-1))

        if key in self.cache:
            return self.cache[key]

        kh, kw = self.locality
        kernel = torch.zeros(1, 1, kh, kw, dtype=torch.bool, device=x.device)
        mask = kernel.repeat([x.size(-2), x.size(-1), 1, 1])

        # H W kh kw -> H kh W kw
        mask = mask.permute([0, 2, 1, 3])
        mask = skew(mask, True)

        # H kh W kw -> W kw H kh
        mask = mask.permute([2, 3, 0, 1])
        mask = skew(mask, True)

        # W kw H kh -> H W kh kw
        mask = mask.permute([2, 0, 3, 1])
        mask = mask.flatten(2, 3).flatten(0, 1)

        self.cache[key] = mask
        return mask


class MultiheadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads: int):
        super().__init__()
        head_dims = hidden_size // num_heads
        self.temperature = head_dims**0.5
        self.num_heads = num_heads
        self.head_dims = head_dims

        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head = nn.ModuleDict()
            head["Q"] = nn.Linear(head_dims, head_dims, bias=False)
            head["K"] = nn.Linear(head_dims, head_dims, bias=False)
            head["V"] = nn.Linear(head_dims, head_dims, bias=False)
            self.heads.append(head)

        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_head(self, i, x, attn_mask=None):
        head = self.heads[i]
        Q = head["Q"](x)
        K = head["K"](x)
        V = head["V"](x)
        energy = Q.matmul(K.transpose(2, 1))
        energy = torch.softmax(energy / self.temperature, dim=-1)
        if attn_mask is not None:
            energy = (~attn_mask) * energy
        out = energy.matmul(V)
        return out

    def forward(self, x, attn_mask=None):
        xs = torch.split(x, self.head_dims, dim=-1)
        xs = [self.forward_head(i, x, attn_mask) for i, x in enumerate(xs)]
        out = torch.cat(xs, dim=-1)
        out = self.output(out)
        return out


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


class FVTREmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        image_height: int,
        image_channel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                image_channel,
                hidden_size,
                kernel_size=(5, 3),
                stride=(4, 2),
            ),
            nn.SELU(True),
        )
        with torch.no_grad():
            img = torch.rand(1, 3, image_height, 128)
            num_hpatch = self.patch_embedding(img).shape[-2]

        self.position_encodings = PositionEncoding(hidden_size, num_hpatch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image):
        embeddings = self.patch_embedding(image)
        embeddings = self.position_encodings(embeddings) + embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class CombiningBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_attention_heads,
        dropout=0.1,
    ):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.GELU(),
        )

    def forward(self, x: Tensor):
        # b c h w -> b c w h -> b c wh -> b wh c
        x = x.transpose(-2, -1)
        x = x.flatten(2)
        x = x.transpose(-2, -1)
        x = self.output(x)
        return x


class MergingBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Conv2d(
            input_size,
            output_size,
            kernel_size=(3, 1),
            padding=(1, 0),
            stride=(2, 1),
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        out = self.conv(x)

        # == n c h w -> n h w c ==
        b, c, h, w = range(4)
        out = out.permute((b, w, h, c))
        out = self.norm(out)

        # == n h w c -> n c h w ==
        b, w, h, c = range(4)
        out = out.permute(b, c, h, w)
        return out


class MixerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_head: int,
        local: bool = False,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local = local
        self.mixer = MultiheadSelfAttention(hidden_size, num_attention_head)
        self.mixer_dropout = nn.Dropout(attn_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm_mixer = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def _get_name(self):
        if self.local:
            return "LocalMixer"
        else:
            return "GlobalMixer"

    def forward_sa(self, patches, mask):
        x = self.mixer(patches, attn_mask=mask)
        return self.mixer_dropout(x)

    def forward(self, patches, mask):
        patches = self.norm_mixer(patches)
        patches = self.forward_sa(patches, mask) + patches
        patches = self.norm_mlp(patches)
        patches = self.mlp(patches) + patches
        return patches


class FVTRStage(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_attention_head: int,
        permutation: List,
        combine: bool,
        locality: Tuple[int, int],
    ):
        super().__init__()

        mixing_blocks = nn.ModuleList()
        self.gen_mask = None
        for local in permutation:
            if local:
                self.gen_mask = LocalAttentionMaskProvider2d(locality)
            block = MixerBlock(
                hidden_size=input_size,
                num_attention_head=num_attention_head,
                local=local,
            )
            mixing_blocks.append(block)

        # merging
        if combine:
            merging = CombiningBlock(input_size, output_size, num_attention_head)
        else:
            merging = MergingBlock(
                input_size,
                output_size,
            )

        self.mixing_blocks = mixing_blocks
        self.merging = merging
        self.to_image = ToImage()
        self.to_fiber = ToFiber()

    def forward(self, image: Tensor):
        x, size = self.to_fiber(image)

        if self.gen_mask is None:
            mask = None
        else:
            mask = self.gen_mask(image)

        for block in self.mixing_blocks:
            if block.local:
                x = block(x, mask=mask)
            else:
                x = block(x, mask=None)

        # b (w h) c -> b w h c -> b c h w
        x = self.to_image(x, size)
        x = self.merging(x)
        return x


class FVTR(nn.Sequential):
    def __init__(
        self,
        hidden_sizes: List[int],
        permutations: List[List[int]],
        num_attention_heads: List[int],
        image_height: int,
        locality: Tuple[int, int] = (7, 11),
    ):
        super().__init__()
        self.locality = locality

        # FVTR Embedding
        embeddings = FVTREmbedding(
            hidden_size=hidden_sizes[0],
            image_height=image_height,
        )

        # FVTR Stages
        stages = []
        n_stages = len(hidden_sizes) - 1
        for i in range(n_stages):
            input_size = hidden_sizes[i]
            output_size = hidden_sizes[i + 1]
            permutation = permutations[i]
            num_attention_head = num_attention_heads[i]
            combine = i == n_stages - 1
            stage = FVTRStage(
                input_size=input_size,
                output_size=output_size,
                permutation=permutation,
                num_attention_head=num_attention_head,
                combine=combine,
                locality=locality,
            )
            stages.append(stage)

        # add modules
        self.embeddings = embeddings
        self.stages = nn.ModuleList(stages)
        self.output_size = hidden_sizes[-1]

    def get_hidden_size(self):
        return self.output_size

    def forward(self, images: Tensor):
        x = self.embeddings(images)
        for stage in self.stages:
            x = stage(x)
        return x


def create_permutation(localities: List, stage_length: List[int]):
    permutations: List[List] = []
    counter = 0
    for N in stage_length:
        p = []
        for i in range(N):
            p.append(localities[counter])
            counter += 1
        permutations.append(p)
    assert len(localities) == counter
    return permutations


def fvtr_t(**opts):
    L, G = True, False
    permutations = create_permutation([L] * 6 + [G] * 6, [3, 6, 3])
    heads = [2, 4, 8]
    hidden_sizes = [64, 128, 256, 192]
    return FVTR(
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


def fvtr_s(**opts):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # FVTR-S | [96, 192, 256]  | [3, 6, 6]    | [3, 6, 8]  | 192 | [L]8[G]7
    L, G = True, False
    permutations = create_permutation([L] * 8 + [G] * 7, [3, 6, 6])
    heads = [3, 6, 8]
    hidden_sizes = [96, 192, 256, 192]
    return FVTR(
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


def fvtr_b(**opts):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # FVTR-B | [128, 256, 384] | [3, 6, 9]    | [4, 8, 12] | 256 | [L]8[G]10
    L, G = True, False
    permutations = create_permutation([L] * 8 + [G] * 10, [3, 6, 9])
    heads = [4, 8, 12]
    hidden_sizes = [128, 256, 384, 256]
    return FVTR(
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


def fvtr_l(**opts):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # FVTR-L | [192, 256, 512] | [3, 9, 9]    | [6, 8, 16] | 384 | [L]10[G]11
    L, G = True, False
    permutations = create_permutation([L] * 10 + [G] * 11, [3, 9, 9])
    heads = [6, 8, 16]
    hidden_sizes = [192, 256, 512, 384]
    return FVTR(
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


MODULES = {
    "fvtr_t": fvtr_t,
    "fvtr_s": fvtr_s,
    "fvtr_b": fvtr_b,
    "fvtr_l": fvtr_l,
}
