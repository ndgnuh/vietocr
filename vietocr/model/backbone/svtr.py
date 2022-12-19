from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation
from einops.layers.torch import Rearrange, Reduce
from typing import Tuple, List, Optional
import torch


@torch.no_grad()
def get_output_size(m, shape):
    x = torch.zeros(shape)
    return m(x).shape


class IceCream(nn.Module):
    """
    This module is for debugging
    """

    def __init__(self, msg=""):
        super().__init__()
        self.msg = msg

    def forward(self, x):
        from icecream import ic
        ic(self.msg, x.shape)
        return x


class PatchEmbedding(nn.Sequential):
    def __init__(self, image_size, hidden_size, patch_size=4):
        super().__init__()
        conv1 = Conv2dNormActivation(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=3,
            stride=2,
            activation_layer=lambda *x, **k: nn.GELU()
        )
        conv2 = Conv2dNormActivation(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=2,
            activation_layer=lambda *x, **k: nn.GELU()
        )
        self.add_module("conv_bn_act_1", conv1)
        self.add_module("conv_bn_act_2", conv2)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, hidden_size):
        super().__init__()
        positional_embedding = nn.Parameter(
            torch.randn(1, num_patches, hidden_size)
        )
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self):
        return self.positional_embedding


class SVTREmbedding(nn.Module):
    def __init__(self, image_size, hidden_size, patch_size=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            hidden_size=hidden_size,
            patch_size=patch_size
        )
        b, c, h, w = get_output_size(self.patch_embedding, (1, 3, *image_size))
        self.to_seq = Rearrange("b c h w -> b (h w) c")
        self.positional_embedding = PositionalEmbedding((h * w), c)
        self.to_img = Rearrange("b (h w) c -> b c h w", h=h, w=w)

    def forward(self, image):
        patch_embedding = self.patch_embedding(image)
        positional_embedding = self.positional_embedding()
        patch_embedding = self.to_seq(patch_embedding)
        embedding = patch_embedding + positional_embedding
        embedding = self.to_img(embedding)
        return embedding


class CombiningBlock(nn.Sequential):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_module("pool", Reduce('b c h w -> b c w', 'mean'))
        self.add_module("to_seq", Rearrange('b c w -> b w c'))
        self.add_module("linear", nn.Linear(input_size, output_size))
        self.add_module("act", nn.GELU())


class MergingBlock(nn.Sequential):
    def __init__(self, input_size, output_size):
        super().__init__()
        conv = nn.Conv2d(input_size, output_size, 3, padding=1, stride=(2, 1))
        c_last = Rearrange("b c h w -> b h w c")
        ln = nn.LayerNorm(output_size)
        c_first = Rearrange("b h w c -> b c h w")

        self.add_module("conv", conv)
        self.add_module("channel_last", c_last)
        # self.add_module("preln", IceCream("preln"))
        self.add_module("ln", ln)
        self.add_module("channel_first", c_first)


class MixerLayer(nn.Module):
    """
    MixerBlock is either Global Mixer or Local Mixer

    if kernel_size is None, this block is global mixer.
    """

    def __init__(self, hidden_size, num_attention_head, image_size,
                 locality=(7, 11), attn_dropout=0., dropout=0.):
        super().__init__()
        self.locality = locality
        self.image_size = image_size
        if locality is not None:
            mask = self.init_mask()
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.attention = nn.MultiheadAttention(hidden_size,
                                               num_attention_head,
                                               batch_first=True,
                                               dropout=attn_dropout)
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout)
        )

    def _get_name(self):
        if self.locality is None:
            return "GlobalMixer"
        else:
            return "LocalMixer"

    def init_mask(self):
        ih, iw = self.image_size
        kh, kw = self.locality

        # Create a padded mask
        mask = torch.ones(
            (iw * ih, ih + kh - 1, iw + kw - 1), dtype=torch.bool)

        # Mask local patch
        for row in range(ih):
            for col in range(iw):
                mask[row * iw + col, row:(row + kh), col:(col + kw)] = 0
        mask = mask[:, (kh // 2):(ih + kh // 2), (kw // 2):(iw + kw // 2)]
        mask = mask.flatten(1)

        # attention mask is either -inf or 0
        mask_inf = torch.ones((iw * ih, iw * ih)) * -torch.inf

        # float mask so that it is added when running attention
        mask = torch.where(mask < 1, mask, mask_inf) * 1.0
        # print(mask.shape, mask.dtype, mask)
        return mask

    def forward(self, patch_embed):
        weight, score = self.attention(
            patch_embed, patch_embed, patch_embed, attn_mask=self.mask)
        # print(score.shape, weight.shape)
        output = self.project(weight)
        return output


class MixerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_head: int,
        image_size: Tuple[int, int],
        locality: Optional[Tuple[int, int]] = (7, 11),
        attn_dropout: float = 0.,
        dropout: float = 0.,
        prelayer_norm: bool = True
    ):
        super().__init__()
        self.prelayer_norm = prelayer_norm
        self.mixer = MixerLayer(hidden_size=hidden_size,
                                num_attention_head=num_attention_head,
                                image_size=image_size,
                                locality=locality,
                                attn_dropout=attn_dropout,
                                dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
            nn.Dropout(dropout),
        )
        self.norm_mixer = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def forward(self, image):
        if self.prelayer_norm:
            image = self.norm_mixer(image)
            image = self.mixer(image) + image
            image = self.norm_mlp(image)
            image = self.mlp(image) + image
        else:
            image = self.mixer(image) + image
            image = self.norm_mixer(image)
            image = self.mlp(image) + image
            image = self.norm_mlp(image)
        return image


class SVTRStage(nn.Sequential):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_attention_head: int,
        permutation: List,
        image_size: Tuple[int, int],
        combine: bool
    ):
        super().__init__()
        h, w = image_size
        # Image to sequence
        to_seq = Rearrange('b c h w -> b (h w) c')

        mixing_blocks = nn.Sequential()
        for i, locality in enumerate(permutation):
            block = MixerBlock(
                hidden_size=input_size,
                num_attention_head=num_attention_head,
                image_size=image_size,
                prelayer_norm=True,
                locality=locality
            )
            mixing_blocks.add_module(str(i), block)

        # Sequence to image
        to_img = Rearrange('b (h w) c -> b c h w', h=h, w=w)

        # merging
        if combine:
            merging = CombiningBlock(input_size, output_size)
        else:
            merging = MergingBlock(input_size, output_size)

        self.add_module("to_seq", to_seq)
        # self.add_module("ic0", IceCream(msg="pre"))
        self.add_module("mixing_blocks", mixing_blocks)
        self.add_module("to_img", to_img)
        self.add_module("merging", merging)
        # self.add_module("ic", IceCream(msg="post"))
        # self.add_module("merging", merging_block)


class SVTR(nn.Sequential):

    def __init__(self,
                 num_classes: int,
                 image_size: Tuple[int, int],
                 hidden_sizes: List[int],
                 permutations: List[List[int]],
                 num_attention_heads: List[int]):
        super().__init__()
        # SVTR Embedding
        c, h, w = 3, *image_size
        embedding = SVTREmbedding(image_size, hidden_sizes[0])

        # SVTR Stages
        stages = []
        b, c, h, w = get_output_size(embedding, (1, c, h, w))
        n_stages = len(hidden_sizes) - 1
        for i in range(n_stages):
            input_size = hidden_sizes[i]
            output_size = hidden_sizes[i + 1]
            permutation = permutations[i]
            num_attention_head = num_attention_heads[i]
            combine = (i == n_stages - 1)
            stage = SVTRStage(
                input_size=input_size,
                output_size=output_size,
                permutation=permutation,
                num_attention_head=num_attention_head,
                image_size=(h, w),
                combine=combine
            )
            stages.append(stage)
            if combine:
                b, s, d = get_output_size(stage, (1, c, h, w))
            else:
                b, c, h, w = get_output_size(stage, (1, c, h, w))

        # Classification
        fc = nn.Linear(hidden_sizes[-1], num_classes)

        # add mods
        self.add_module("svtr_embedding", embedding)
        for i, stage in enumerate(stages):
            self.add_module(f"stage_{i:02}", stage)
        self.add_module("fc", fc)

# Model configurations
# From https://arxiv.org/pdf/2205.00159.pdf
# Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
# ---    | ---             | ---          | ---        | --- | ---
# SVTR-T | [64, 128, 256]  | [3, 6, 3]    | [2, 4, 8]  | 192 | [L]6[G]6
# SVTR-S | [96, 192, 256]  | [3, 6, 6]    | [3, 6, 8]  | 192 | [L]8[G]7
# SVTR-B | [128, 256, 384] | [3, 6, 9]    | [4, 8, 12] | 256 | [L]8[G]10
# SVTR-L | [192, 256, 512] | [3, 9, 9]    | [6, 8, 16] | 384 | [L]10[G]11


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


def svtr_t(image_size, output_size):
    L, G = (7, 11), None
    permutations = create_permutation([L] * 6 + [G] * 6, [3, 6, 3])
    heads = [2, 4, 8]
    hidden_sizes = [64, 128, 256, 192]
    return SVTR(image_size=image_size,
                num_classes=output_size,
                permutations=permutations,
                num_attention_heads=heads,
                hidden_sizes=hidden_sizes)


def svtr_s(image_size, output_size):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # SVTR-S | [96, 192, 256]  | [3, 6, 6]    | [3, 6, 8]  | 192 | [L]8[G]7
    L, G = (7, 11), None
    permutations = create_permutation([L] * 8 + [G] * 7, [3, 6, 6])
    heads = [3, 6, 8]
    hidden_sizes = [96, 192, 256, 192]
    return SVTR(image_size=image_size,
                num_classes=output_size,
                permutations=permutations,
                num_attention_heads=heads,
                hidden_sizes=hidden_sizes)


def svtr_b(image_size, output_size):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # SVTR-B | [128, 256, 384] | [3, 6, 9]    | [4, 8, 12] | 256 | [L]8[G]10
    L, G = (7, 11), None
    permutations = create_permutation([L] * 8 + [G] * 10, [3, 6, 9])
    heads = [4, 8, 12]
    hidden_sizes = [128, 256, 384, 256]
    return SVTR(image_size=image_size,
                num_classes=output_size,
                permutations=permutations,
                num_attention_heads=heads,
                hidden_sizes=hidden_sizes)


def svtr_l(image_size, output_size):
    # Models | [D0, D1, D2]    | [L1, L2, L3] | Heads      | D3  | Permutation
    # ---    | ---             | ---          | ---        | --- | ---
    # SVTR-L | [192, 256, 512] | [3, 9, 9]    | [6, 8, 16] | 384 | [L]10[G]11
    L, G = (7, 11), None
    permutations = create_permutation([L] * 10 + [G] * 11, [3, 9, 9])
    heads = [6, 8, 16]
    hidden_sizes = [192, 256, 512, 384]
    return SVTR(image_size=image_size,
                num_classes=output_size,
                permutations=permutations,
                num_attention_heads=heads,
                hidden_sizes=hidden_sizes)
