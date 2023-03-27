from torchvision.transforms import functional as TF
from einops.layers.torch import Rearrange, Reduce
from torch.nn import functional as F
from torch import nn, Tensor
from typing import Tuple, List
import torch
import math


def skew(orig_x, padding_value):
    x = orig_x
    '''shift every row 1 step to right converting columns into diagonals'''
    rest, H, W = x.shape[:-2], x.size(-2), x.size(-1)
    x = F.pad(x, (0, H), value=padding_value)
    x = x.reshape(*rest, -1)  # B x C x ML+MM+M
    x = x[..., :-H]  # B x C x ML+MM
    x = x.reshape(*rest, H, H + W - 1)  # B x C, M x L+M
    x = x[..., (W//2):-(W//2)+(W % 2-1)]
    return x


class MaskProvider2d(nn.Module):
    def __init__(self, locality):
        super().__init__()
        self.locality = locality

    @torch.no_grad()
    def forward(self, x):
        kh, kw = self.locality
        kernel = torch.ones(1, 1, kh, kw, dtype=torch.bool, device=x.device)
        mask = kernel.repeat([x.size(-2), x.size(-1), 1, 1])

        # H W kh kw -> H kh W kw
        mask = mask.permute([0, 2, 1, 3])
        mask = skew(mask, 0)

        # H kh W kw -> W kw H kh
        mask = mask.permute([2, 3, 0, 1])
        mask = skew(mask, 0)

        # W kw H kh -> H W kh kw
        mask = mask.permute([2, 0, 3, 1])
        mask = mask.flatten(2, 3).flatten(0, 1)

        # bool mask to inf
        # mask_inf = torch.ones_like(mask) * -torch.inf
        # mask = torch.where(mask, mask, mask_inf)
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, position_dim=-1):
        super().__init__()
        self.position_dim = position_dim
        inv_freq = torch.exp(torch.arange(0, hidden_size, 2)
                             * (-math.log(10000.0) / hidden_size))
        self.register_buffer("inv_freq", inv_freq)

    def get_embed(self, inp):
        return torch.stack([inp.sin(), inp.cos()], dim=-1).flatten(-2)

    def forward(self, x):
        position = torch.arange(x.size(self.position_dim),
                                dtype=x.dtype, device=x.device)
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
        hsin = torch.einsum('i,j->ij', hsin, self.inv_freq)
        wsin = torch.einsum('i,j->ij', wsin, self.inv_freq)

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
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class XSin(nn.Module):
    def forward(self, x):
        return x * torch.sin(x)


class FVTREmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        position_ids: int,
        image_channel: int = 3,
        patch_size: int = 4,
        norm_type='batchnorm',
        patch_embedding_type='default',
        learnable_pe=True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(image_channel, hidden_size,
                      kernel_size=3, stride=2),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size,
                      kernel_size=(3, 1), stride=(2, 1)),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(True),
        )
        # encode position along the width of the image
        self.learnable_pe = learnable_pe
        if learnable_pe:
            self.pe = nn.Parameter(torch.zeros(1, 1, *position_ids))
        else:
            self.pe = PositionalEncoding(hidden_size, position_dim=3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image):
        embeddings = self.patch_embedding(image)
        # w * c
        if self.learnable_pe:
            pe = TF.resize(
                self.pe,
                (embeddings.size(-2), embeddings.size(-1)),
                antialias=True,
            )
            pe = pe.repeat([embeddings.size(0), embeddings.size(1), 1, 1])
        else:
            pe = self.pe(embeddings)
            # w c -> 1 1 w c
            pe = pe.unsqueeze(0).unsqueeze(1)
            # 1 1 w c -> b h w c
            pe = pe.repeat([embeddings.size(0), embeddings.size(2), 1, 1])
            # b h w c -> b c h w
            pe = pe.permute([0, 3, 1, 2])
        embeddings = self.dropout(pe + embeddings)
        return embeddings


class SpatialAttention2d(nn.Module):
    def __init__(self, locality: int = 7):
        super().__init__()
        self.f = nn.Conv2d(2, 1, locality, padding=locality // 2)

    def forward(self, x):
        # b c h w -> b 1 h w
        Fmax, _ = x.max(dim=1, keepdim=True)
        # b c h w -> b 1 h w
        Favg = x.mean(dim=1, keepdim=True)
        # (b 1 h w) * 2 -> b 1 h w
        attn = self.f(torch.cat([Favg, Fmax], dim=1))
        attn = torch.sigmoid(attn)
        return attn


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
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size,
                              kernel_size=(3, 1),
                              padding=(1, 0),
                              stride=(2, 1))
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
        attn_dropout: float = 0.,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local = local
        self.mixer = nn.MultiheadAttention(
            hidden_size,
            num_attention_head,
            dropout=attn_dropout,
            batch_first=True
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

    def _get_name(self):
        if self.local:
            return "LocalMixer"
        else:
            return "GlobalMixer"

    def forward_sa(self, patches, mask):
        x = self.mixer(patches, patches, patches,
                       attn_mask=mask, need_weights=False)[0]
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
                self.gen_mask = MaskProvider2d(locality)
            block = MixerBlock(
                hidden_size=input_size,
                num_attention_head=num_attention_head,
                local=local,
            )
            mixing_blocks.append(block)

        # merging
        if combine:
            merging = CombiningBlock(
                input_size,
                output_size,
                num_attention_head
            )
        else:
            merging = MergingBlock(input_size, output_size)

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
    def __init__(self,
                 hidden_sizes: List[int],
                 output_size: int,
                 permutations: List[List[int]],
                 num_attention_heads: List[int],
                 locality: Tuple[int, int] = (9, 9),
                 patch_size: int = 4,
                 image_channel: int = 3,
                 position_ids: int = (8, 64),
                 learnable_pe: bool = True,
                 norm_type: str = 'batchnorm',
                 use_rnn: bool = False,
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
            position_ids=position_ids,
            learnable_pe=learnable_pe,
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
                combine=combine,
                locality=locality,
            )
            stages.append(stage)

        # Classification
        fc = nn.Linear(hidden_sizes[-1], self.output_size)

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
    "fvtr_v2_t": fvtr_t,
    "fvtr_v2_tg": fvtr_tg,
    "fvtr_v2_s": fvtr_s,
    "fvtr_v2_b": fvtr_b,
    "fvtr_v2_l": fvtr_l
}
