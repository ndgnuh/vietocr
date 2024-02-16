import math
from typing import List, Optional, Tuple

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from ..utils import DConv2d, create_local_attention_mask


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first=True,
        need_weights: bool = False,
    ):
        super().__init__()
        # Make it compat with nn.MultiheadAttentionWeights
        self.in_proj_weight = nn.Parameter(torch.randn(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.randn(hidden_size * 3))
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(
            hidden_size, hidden_size
        )
        self.dropout = nn.Dropout(dropout)
        self.need_weights = need_weights
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(self, query, key, value, attn_mask=None):
        # Handle batch dims
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Unpack dimensions
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        num_heads = self.num_heads
        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
        assert (
            head_dim * self.num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Input projection
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
            3, dim=-1
        )
        q = q * scaling

        # Compute attention weights
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # Apply masks
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        # Apply energy and dropout
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        # Handle batch dimensions
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if self.need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


def get_conv_layer(deformable: bool):
    return DConv2d if deformable else nn.Conv2d


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, position_dim=-1):
        super().__init__()
        self.position_dim = position_dim
        inv_freq = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        self.register_buffer("inv_freq", inv_freq)

    def get_embed(self, inp):
        return torch.stack([inp.sin(), inp.cos()], dim=-1).flatten(-2)

    def forward(self, x):
        position = torch.arange(
            x.size(self.position_dim), dtype=x.dtype, device=x.device
        )
        position = position.unsqueeze(1)
        sin = torch.sin(position * self.inv_freq)
        cos = torch.cos(position * self.inv_freq)

        # time * hidden_size
        encoding = torch.stack([sin, cos], dim=-1).flatten(-2)
        return encoding


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        channels = int(round(num_channels / 4 + 0.5) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2) / channels))
        self.register_buffer("inv_freq", inv_freq)

    def get_embed(self, inp):
        return torch.stack([inp.sin(), inp.cos()], dim=-1).flatten(-2)

    def forward(self, x):
        # x: b c h w

        # h | w
        hsin = torch.arange(x.size(-2), dtype=x.dtype, device=x.device)
        wsin = torch.arange(x.size(-1), dtype=x.dtype, device=x.device)

        # h, (c / 4) | w, (c / 4)
        hsin = torch.einsum("i,j->ij", hsin, self.inv_freq)
        wsin = torch.einsum("i,j->ij", wsin, self.inv_freq)

        # h * 1 * (c / 2) | 1 * w * (c / 2)
        hemb = self.get_embed(hsin).unsqueeze(1)
        wemb = self.get_embed(wsin).unsqueeze(0)

        # h w c -> c h w
        posemb = torch.cat(torch.broadcast_tensors(hemb, wemb), dim=2)
        posemb = posemb.permute((2, 0, 1))

        # add batch
        posemb = posemb.unsqueeze(0).repeat([x.size(0), 1, 1, 1])

        return posemb


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class XSin(nn.Module):
    def forward(self, x):
        return x * torch.sin(x)


class LRSCPE(nn.Module):
    def __init__(self, num_channels, position_ids):
        super().__init__()
        self.lpe = nn.Parameter(torch.zeros(1, num_channels, position_ids[0], 1))
        self.spe = PositionalEncoding(num_channels, position_dim=-1)

    def forward(self, images):
        lpe = self.lpe
        spe = self.spe(images)
        spe = spe.transpose(1, 0)
        spe = spe.unsqueeze(0).unsqueeze(2)
        return spe + lpe


class FVTREmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        position_ids: int,
        image_channel: int = 3,
        patch_size: int = 4,
        deformable: bool = False,
        norm_type="batchnorm",
        pe_type="learnable",
        dropout: float = 0.1,
    ):
        super().__init__()
        Conv2d = get_conv_layer(deformable)
        self.patch_embedding = nn.Sequential(
            Conv2d(image_channel, hidden_size, kernel_size=3, stride=2),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(True),
            Conv2d(hidden_size, hidden_size, kernel_size=(3, 1), stride=(2, 1)),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(True),
        )
        # encode position along the width of the image
        self.pe_type = pe_type
        if pe_type == "learnable":
            self.pe = nn.Parameter(torch.zeros(1, 1, *position_ids))
        elif pe_type == "lrsc":
            self.pe = LRSCPE(hidden_size, position_ids)
        elif pe_type == "sin_2d":
            self.pe = PositionalEncoding2D(hidden_size)
        elif pe_type == "sin_row":
            self.pe = PositionalEncoding(hidden_size, position_dim=2)
        elif pe_type == "none":
            self.pe = None
        else:
            raise ValueError(
                f"Unsupported position embedding types {pe_type}: learnable, sin_2d, sin_1d, none"
            )
        self.dropout = nn.Dropout(dropout)

        # The good model PE type is learnable and height is 7
        # ic(self.pe_type)
        # ic(self.pe.shape)

    def forward(self, image):
        embeddings = self.patch_embedding(image)
        if self.pe_type == 'learnable':
            pe = self.pe
        elif self.pe_type == 'sin_row':
            pe = self.pe(embeddings)
            # h c -> 1 h 1 c
            pe = pe.unsqueeze(1).unsqueeze(0)
            # 1 h 1 c -> b h w c
            pe = pe.repeat([embeddings.size(0), 1, embeddings.size(3), 1])
            # b h w c -> b c h w
            pe = pe.permute([0, 3, 1, 2])
        elif self.pe is None:
            pe = 0
        elif self.pe_type == 'lrsc':
            pe = self.pe(embeddings)
        else:
            pe = self.pe(embeddings)
            # w c -> 1 1 w c
            pe = pe.unsqueeze(0).unsqueeze(1)
            # 1 1 w c -> b h w c
            pe = pe.repeat([embeddings.size(0), embeddings.size(2), 1, 1])
            # b h w c -> b c h w
            pe = pe.permute([0, 3, 1, 2])
        embeddings = self.dropout(pe + embeddings)
>>>>>>> 51ec6517ecae958c876cd7a5ce8071f7714cad0b
        return embeddings


class CombiningBlock(nn.Module):
    def __init__(self, input_size, output_size, num_attention_heads, dropout=0.1):
        super().__init__()
        self.project = nn.Linear(input_size, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def corner_pooling(self, image):
        w = image.max(dim=-1, keepdim=True).values
        h = image.max(dim=-2, keepdim=True).values
        return w + h

    def forward(self, image):
        # b c h w -> b c w
        out = (image * self.corner_pooling(image)).mean(dim=2)
        # b c w -> b w c
        out = out.transpose(1, 2)

        out = self.project(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class MergingBlock(nn.Module):
    def __init__(self, input_size, output_size, deformable: bool = False):
        super().__init__()
        Conv2d = get_conv_layer(deformable)
        self.conv = Conv2d(
            input_size, output_size, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)
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


class MixerCommon(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_head: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        # ic(dropout, attn_dropout)
        super().__init__()
        self.mixer = MultiheadAttention(
            hidden_size,
            num_attention_head,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.mixer_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm_mixer = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def forward(self, img):
        # Transpose to patches
        # b c h w -> b w h c -> h (w h) c
        c, h, w = img.shape[-3:]
        x = img.transpose(-1, -3)
        x = x.reshape(-1, (w * h), c)

        # Transformer forward
        x = self.norm_mixer(x)
        x = self.forward_sa(x, img) + x
        x = self.norm_mlp(x)
        x = self.mlp(x) + x

        # Tranpose back to image
        # b (w h) c -> b w h c -> b c h w
        x = x.reshape(-1, w, h, c)
        x = x.transpose(-1, -3)

        return x


class GlobalMixer(MixerCommon):
    def forward_sa(self, x, img):
        x = self.mixer(x, x, x)[0]
        return self.mixer_dropout(x)


class LocalMixer(MixerCommon):
    def forward_sa(self, x, img):
        mask = create_local_attention_mask(img, 7, 11)
        x = self.mixer(x, x, x, attn_mask=mask)[0]
        return self.mixer_dropout(x)


class FVTRStage(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_attention_head: int,
        permutation: List,
        combine: bool,
        locality: Tuple[int, int],
        deformable: bool,
    ):
        super().__init__()

        # ic(locality) 7 x 11
        mixing_blocks = nn.ModuleList()
        self.gen_mask = None
        for local in permutation:
            options = dict(
                hidden_size=input_size,
                num_attention_head=num_attention_head,
            )
            block = LocalMixer(**options) if local else GlobalMixer(**options)
            mixing_blocks.append(block)

        # merging
        if combine:
            merging = CombiningBlock(input_size, output_size, num_attention_head)
        else:
            merging = MergingBlock(
                input_size,
                output_size,
                deformable=deformable,
            )

        self.mixing_blocks = mixing_blocks
        self.merging = merging

    def forward(self, x: Tensor):
        for block in self.mixing_blocks:
            x = block(x)

        x = self.merging(x)
        return x


class FVTR(nn.Sequential):
    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        permutations: List[List[int]],
        num_attention_heads: List[int],
        locality: Tuple[int, int] = (7, 11),
        patch_size: int = 4,
        image_channel: int = 3,
        position_ids: int = (8, 64),
        pe_type: bool = "learnable",
        norm_type: str = "batchnorm",
        use_fc: bool = True,
        deformable: bool = False,
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
            pe_type=pe_type,
            deformable=deformable,
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
                deformable=deformable,
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


def fvtr_tg(output_size, **opts):
    G = True
    permutations = create_permutation([G] * 6 + [G] * 6, [3, 6, 3])
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
    "fvtr_v2_t": fvtr_t,
    "fvtr_v2_tg": fvtr_tg,
    "fvtr_v2_s": fvtr_s,
    "fvtr_v2_b": fvtr_b,
    "fvtr_v2_l": fvtr_l,
}
