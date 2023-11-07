# This version SHOULD be exportable...

import math
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils import LocalAttentionMaskProvider2d


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.d_head = hidden_size // num_heads
        self.n_heads = num_heads
        self.d_model = hidden_size
        self.in_weight = nn.Parameter(torch.rand(hidden_size * 3, hidden_size))
        self.in_bias = nn.Parameter(torch.rand(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.temp = math.sqrt(self.d_head)

    def attention(self, q, k, v, Ws, Bs, mask=None):
        qw, kw, vw = Ws.unbind(1)
        qb, kb, vb = Bs.unbind(1)
        Q = F.linear(q, qw, qb)
        K = F.linear(k, kw, kb)
        V = F.linear(v, vw, vb)
        atn = Q.matmul(K.transpose(1, 2))
        atn = torch.softmax(atn / self.temp, dim=2)
        if mask is not None:
            if mask.dtype == torch.bool:
                atn = torch.where(mask, torch.zeros_like(atn), atn)
            elif mask.dtype == torch.float:
                atn = atn + mask
            elif mask.dtype == torch.long:
                atn = atn * mask
        out = atn.matmul(V)
        return V, atn

    def forward(self, q, k=None, v=None, mask=None):
        k = q if k is None else k
        v = k if v is None else v

        # Attention
        qkv_weights = self.in_weight.reshape(
            self.n_heads, self.d_head, 3, self.d_model
        ).unbind(0)
        qkv_biases = self.in_bias.reshape(self.n_heads, self.d_head, 3).unbind(0)
        attentions = [
            self.attention(q, k, v, w, b, mask) for w, b in zip(qkv_weights, qkv_biases)
        ]

        # Multihead
        heads = torch.cat([a[0] for a in attentions], dim=2)
        masks = sum(a[1] for a in attentions)
        outputs = self.out_proj(heads)
        return outputs, masks


class FVTREmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        position_ids: int,
        image_height: int,
        image_channel: int = 3,
        patch_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(image_channel, hidden_size, kernel_size=3, stride=2),
            nn.SELU(True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 1), stride=(2, 1)),
            nn.SELU(True),
        )
        with torch.no_grad():
            img = torch.rand(1, 3, image_height, 30)
            num_hpatch = self.patch_embedding(img).shape[-2]
        pe = torch.randn(hidden_size, num_hpatch)
        nn.init.orthogonal_(pe)
        self.position_encodings = nn.Parameter(pe[None, :, :, None])
        self.dropout = nn.Dropout(dropout)

    def forward(self, image):
        embeddings = self.patch_embedding(image)
        embeddings = self.dropout(self.position_encodings + embeddings)
        return embeddings


class CombiningBlock(nn.Module):
    def __init__(self, input_size, output_size, num_attention_heads, dropout=0.1):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.SELU(True),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        # b c h w -> b c w
        out = images.mean(dim=2)
        # b c w -> b w c
        out = out.transpose(1, 2)
        out = self.output(out)
        return out


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
        # n c h w -> n h w c
        out = out.permute((0, 2, 3, 1))
        out = self.norm(out)
        # n h w c -> n c h w
        out = out.permute((0, 3, 1, 2))
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
        self.mixer = MultiheadAttention(hidden_size, num_attention_head)
        self.mixer_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SELU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm_mixer = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def _get_name(self):
        if self.local:
            return "LocalMixer"
        else:
            return "GlobalMixer"

    def forward_sa(self, patches, mask):
        x = self.mixer(patches, patches, patches, mask=mask)[0]
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

    def forward(self, image: Tensor):
        c, h, w = image.shape[-3:]
        # b c h w -> b w h c -> h (w h) c
        x = image.transpose(-1, -3)
        x = x.reshape(-1, (w * h), c)
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
        x = x.reshape(-1, w, h, c)
        x = x.transpose(-1, -3)

        x = self.merging(x)
        return x


class FVTR(nn.Sequential):
    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        permutations: List[List[int]],
        num_attention_heads: List[int],
        image_height: int,
        locality: Tuple[int, int] = (7, 11),
        patch_size: int = 4,
        image_channel: int = 3,
        position_ids: int = (8, 64),
        use_fc: bool = True,
    ):
        super().__init__()
        self.locality = locality
        self.patch_size = patch_size
        self.output_size = output_size

        # FVTR Embedding
        embeddings = FVTREmbedding(
            hidden_size=hidden_sizes[0],
            patch_size=patch_size,
            image_channel=image_channel,
            position_ids=position_ids,
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

        # Classification
        if use_fc:
            fc = nn.Linear(hidden_sizes[-1], self.output_size)
        else:
            fc = nn.Identity()

        # add modules
        self.embeddings = embeddings
        self.stages = nn.ModuleList(stages)
        self.fc = fc

    def forward(self, images: Tensor):
        x = self.embeddings(images)
        for stage in self.stages:
            x = stage(x)

        x = self.fc(x)
        # b t h -> t b h
        x = x.transpose(0, 1)
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


def fvtr_t(output_size, **opts):
    L, G = True, False
    permutations = create_permutation([L] * 6 + [G] * 6, [3, 6, 3])
    heads = [2, 4, 8]
    hidden_sizes = [64, 128, 256, 192]
    return FVTR(
        output_size=output_size,
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


def fvtr_s(output_size, **opts):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # FVTR-S | [96, 192, 256]  | [3, 6, 6]    | [3, 6, 8]  | 192 | [L]8[G]7
    L, G = True, False
    permutations = create_permutation([L] * 8 + [G] * 7, [3, 6, 6])
    heads = [3, 6, 8]
    hidden_sizes = [96, 192, 256, 192]
    return FVTR(
        output_size=output_size,
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


def fvtr_b(output_size, **opts):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # FVTR-B | [128, 256, 384] | [3, 6, 9]    | [4, 8, 12] | 256 | [L]8[G]10
    L, G = True, False
    permutations = create_permutation([L] * 8 + [G] * 10, [3, 6, 9])
    heads = [4, 8, 12]
    hidden_sizes = [128, 256, 384, 256]
    return FVTR(
        output_size=output_size,
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


def fvtr_l(output_size, **opts):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # FVTR-L | [192, 256, 512] | [3, 9, 9]    | [6, 8, 16] | 384 | [L]10[G]11
    L, G = True, False
    permutations = create_permutation([L] * 10 + [G] * 11, [3, 9, 9])
    heads = [6, 8, 16]
    hidden_sizes = [192, 256, 512, 384]
    return FVTR(
        output_size=output_size,
        permutations=permutations,
        num_attention_heads=heads,
        hidden_sizes=hidden_sizes,
        **opts,
    )


models = {
    "fvtr_v3_t": fvtr_t,
    "fvtr_v3_s": fvtr_s,
    "fvtr_v3_b": fvtr_b,
    "fvtr_v3_l": fvtr_l,
}
