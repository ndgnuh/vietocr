from torchvision.transforms import functional as TF
from einops.layers.torch import Rearrange, Reduce
from torch import nn, Tensor
from typing import Tuple, List
import torch


@torch.no_grad()
def get_output_size(m, shape):
    x = torch.zeros(shape)
    return m(x).shape


def init_mask(image: Tensor,
              locality: Tuple[int, int] = (7, 11),
              patch_size: int = 1):
    # Dynamic init mask function
    ih, iw = image.shape[-2:]
    ih = ih // patch_size
    iw = iw // patch_size
    kh, kw = locality

    # Create a padded mask
    mask = torch.ones(
        (iw * ih, ih + kh - 1, iw + kw - 1),
        dtype=torch.bool,
        device=image.device
    )

    # Mask local patch
    for row in range(ih):
        for col in range(iw):
            mask[row * iw + col, row:(row + kh), col:(col + kw)] = 0
    mask = mask[:, (kh // 2):(ih + kh // 2), (kw // 2):(iw + kw // 2)]
    mask = mask.flatten(1)

    # attention mask is either -inf or 0
    mask_inf = torch.ones((iw * ih, iw * ih), device=image.device) * -torch.inf

    # float mask so that it is added when running attention
    mask = torch.where(mask < 1, mask, mask_inf) * 1.0
    # print(mask.shape, mask.dtype, mask)
    return mask


class PatchEmbedding(nn.Sequential):
    def __init__(self,
                 hidden_size: int,
                 image_channel: int = 3,
                 patch_size: int = 4,
                 norm_type: str = 'batchnorm',
                 embedding_type: str = 'default'):
        super().__init__()
        assert norm_type in ['batchnorm', 'instancenorm', 'localresponse']
        assert embedding_type in ['default', 'simple', '2x2-overlap', '4x3']
        if norm_type == 'batchnorm':
            Norm = nn.BatchNorm2d
        elif norm_type == 'instancenorm':
            Norm = nn.InstanceNorm2d
        elif norm_type == 'localresponse':
            Norm = nn.LocalResponseNorm

        if embedding_type == 'default':
            self.conv1 = nn.Sequential(
                nn.Conv2d(image_channel,
                          hidden_size,
                          kernel_size=(3, 1),
                          stride=(2, 1),
                          bias=False),
                Norm(hidden_size),
                nn.GELU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_size,
                          hidden_size,
                          kernel_size=3,
                          stride=2,
                          bias=False),
                Norm(hidden_size),
                nn.GELU()
            )
        elif embedding_type == "4x3":
            self.patch_embed = nn.Sequential(
                nn.Conv2d(image_channel, hidden_size,
                          kernel_size=(3, 1), stride=(2, 1), bias=False),
                Norm(hidden_size),
                nn.GELU(),
                nn.Conv2d(hidden_size, hidden_size,
                          kernel_size=(3, 4), stride=(2, 3), bias=False),
                Norm(hidden_size),
                nn.GELU(),
            )
        elif embedding_type == 'simple':
            self.conv = nn.Sequential(
                nn.Conv2d(image_channel, hidden_size,
                          patch_size, stride=patch_size,
                          bias=False),
                Norm(hidden_size),
                nn.GELU()
            )
        elif embedding_type == '2x2-overlap':
            self.conv = nn.Sequential(
                nn.Conv2d(image_channel, hidden_size, 4, stride=2, bias=False),
                Norm(hidden_size),
                nn.GELU()
            )


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_position_ids: int = 128):
        super().__init__()
        embedding = nn.Parameter(
            torch.rand(hidden_size, max_position_ids, max_position_ids)
        )
        self.register_buffer("embedding", embedding)

    def forward(self, image):
        pos_embds = TF.resize(self.embedding, image.shape[-2:])
        return pos_embds


class FVTREmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        image_channel: int = 3,
        patch_size: int = 4,
        max_position_ids: int = 128,
        norm_type='batchnorm',
        patch_embedding_type='default',
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_channel=image_channel,
            norm_type=norm_type,
            embedding_type=patch_embedding_type
        )
        self.max_position_ids = max_position_ids
        if max_position_ids > 0:
            self.position_embedding = PositionEmbedding(
                hidden_size=hidden_size,
                max_position_ids=max_position_ids)

    def forward(self, image):
        patches = self.patch_embedding(image)
        if self.max_position_ids > 0:
            positions = self.position_embedding(patches)
            embeddings = positions.unsqueeze(0) + patches
        else:
            embeddings = patches
        return embeddings


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
        conv = nn.Conv2d(input_size, output_size, 3,
                         padding=1, stride=(2, 1), bias=False)
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

    if local = True, input attention mask is ignored
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_head: int,
                 local: bool = False,
                 attn_dropout: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.local = local
        self.attention = nn.MultiheadAttention(hidden_size,
                                               num_attention_head,
                                               batch_first=True,
                                               dropout=attn_dropout)
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout)
        )

    def _get_name(self):
        if self.local:
            return "LocalMixer"
        else:
            return "GlobalMixer"

    def forward(self, patch_embed, attention_mask):
        if self.local:
            attention_mask = None
        weight, score = self.attention(
            patch_embed,
            patch_embed,
            patch_embed,
            attn_mask=attention_mask)
        output = self.project(weight)
        return output


class MixerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_head: int,
        local: bool = False,
        attn_dropout: float = 0.,
        dropout: float = 0.,
        prelayer_norm: bool = False
    ):
        super().__init__()
        self.prelayer_norm = prelayer_norm
        self.mixer = MixerLayer(hidden_size=hidden_size,
                                num_attention_head=num_attention_head,
                                local=local,
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

    def forward(self, image, attention_mask):
        if self.prelayer_norm:
            image = self.norm_mixer(image)
            image = self.mixer(image, attention_mask) + image
            image = self.norm_mlp(image)
            image = self.mlp(image) + image
        else:
            image = self.mixer(image, attention_mask) + image
            image = self.norm_mixer(image)
            image = self.mlp(image) + image
            image = self.norm_mlp(image)
        return image


class FVTRStage(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_attention_head: int,
        permutation: List,
        combine: bool
    ):
        super().__init__()

        mixing_blocks = nn.ModuleList()
        for local in permutation:
            block = MixerBlock(
                hidden_size=input_size,
                num_attention_head=num_attention_head,
                prelayer_norm=True,
                local=local,
            )
            mixing_blocks.append(block)

        # merging
        if combine:
            merging = CombiningBlock(input_size, output_size)
        else:
            merging = MergingBlock(input_size, output_size)

        self.mixing_blocks = mixing_blocks
        self.merging = merging

    def forward(self, x: Tensor, attention_mask: Tensor):
        c, h, w = x.shape[-3:]
        # b c h w -> b w h c -> h (w h) c
        x = x.transpose(-1, -3)
        x = x.reshape(-1, (w * h), c)

        for block in self.mixing_blocks:
            x = block(x, attention_mask)

        # b (w h) c -> b w h c -> b c h w
        x = x.reshape(-1, w, h, c)
        x = x.transpose(-1, -3)

        x = self.merging(x)
        return x


class FVTR(nn.Sequential):
    def __init__(self,
                 hidden_sizes: List[int],
                 output_size: int,
                 permutations: List[List[int]],
                 num_attention_heads: List[int],
                 locality: Tuple[int, int] = (7, 11),
                 patch_size: int = 4,
                 image_channel: int = 3,
                 max_position_ids: int = 128,
                 norm_type: str = 'batchnorm',
                 patch_embedding_type: str = 'default'):
        super().__init__()
        self.locality = locality
        self.patch_size = patch_size
        self.output_size = output_size

        # FVTR Embedding
        embeddings = FVTREmbedding(
            hidden_size=hidden_sizes[0],
            patch_size=patch_size,
            image_channel=image_channel,
            max_position_ids=max_position_ids,
            norm_type=norm_type,
            patch_embedding_type=patch_embedding_type,
        )

        # FVTR Stages
        stages = []
        n_stages = len(hidden_sizes) - 1
        for i in range(n_stages):
            input_size = hidden_sizes[i]
            output_size = hidden_sizes[i + 1]
            permutation = permutations[i]
            num_attention_head = num_attention_heads[i]
            combine = (i == n_stages - 1)
            stage = FVTRStage(
                input_size=input_size,
                output_size=output_size,
                permutation=permutation,
                num_attention_head=num_attention_head,
                combine=combine
            )
            stages.append(stage)

        # Classification
        fc = nn.Linear(hidden_sizes[-1], self.output_size)

        # add modules
        self.embeddings = embeddings
        self.stages = nn.ModuleList(stages)
        self.fc = fc

    @staticmethod
    def init_attention_mask(images, locality, patch_size=1):
        # Convenient calls from FVTR model
        return init_mask(images, locality, patch_size=1)

    def forward(self, images: Tensor):
        x = self.embeddings(images)
        for stage in self.stages:
            attention_masks = init_mask(x, self.locality)
            x = stage(x, attention_masks)

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
        **opts)


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
        **opts)


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
        **opts)


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
        **opts)


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
        **opts)


models = {
    "fvtr_t": fvtr_t,
    "fvtr_tg": fvtr_tg,
    "fvtr_s": fvtr_s,
    "fvtr_b": fvtr_b,
    "fvtr_l": fvtr_l
}
