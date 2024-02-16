import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops


class DConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *a, **kw):
        super().__init__()
        self.conv = ops.DeformConv2d(in_channels, out_channels, kernel_size, *a, **kw)
        offset_channels = (
            2 * self.conv.groups * self.conv.kernel_size[1] * self.conv.kernel_size[0]
        )
        self.offset = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size,
            padding=self.conv.padding,
            stride=self.conv.stride,
            groups=self.conv.groups,
            bias=False,
        )

    def forward(self, images):
        offsets = self.offset(images)
        return self.conv(images, offsets)


def skew(orig_x, padding_value):
    x = orig_x
    """shift every row 1 step to right converting columns into diagonals"""
    B, C, H, W = x.shape
    x = F.pad(x, (0, H), value=padding_value)
    x = x.reshape(B, C, -1)  # B x C x ML+MM+M
    x = x[..., :-H]  # B x C x ML+MM
    x = x.reshape(B, C, H, H + W - 1)  # B x C, M x L+M
    x = x[..., (W // 2) : -(W // 2) + (W % 2 - 1)]
    return x


class LocalAttentionMaskProvider2d(nn.Module):
    def __init__(self, locality):
        super().__init__()
        self.locality = locality
        self.cache = {}

    def forward(self, x):
        kh, kw = self.locality
        kernel = torch.zeros(1, 1, kh, kw, dtype=torch.bool, device=x.device)
        mask = kernel.repeat([x.size(-2), x.size(-1), 1, 1])

        # H W kh kw -> H kh W kw
        mask = mask.permute([0, 2, 1, 3])
        mask = skew(mask, torch.tensor(True))

        # H kh W kw -> W kw H kh
        mask = mask.permute([2, 3, 0, 1])
        mask = skew(mask, torch.tensor(True))

        # W kw H kh -> H W kh kw
        mask = mask.permute([2, 0, 3, 1])
        mask = mask.flatten(2, 3).flatten(0, 1)

        # self.cache[key] = mask
        return mask


def create_local_attention_mask(x: torch.Tensor, kh: int, kw: int):
    kernel = torch.zeros(1, 1, kh, kw, dtype=torch.bool, device=x.device)
    mask = kernel.repeat([x.size(-2), x.size(-1), 1, 1])

    # H W kh kw -> H kh W kw
    mask = mask.permute([0, 2, 1, 3])
    mask = skew(mask, torch.tensor(True))

    # H kh W kw -> W kw H kh
    mask = mask.permute([2, 3, 0, 1])
    mask = skew(mask, torch.tensor(True))

    # W kw H kh -> H W kh kw
    mask = mask.permute([2, 0, 3, 1])
    mask = mask.flatten(2, 3).flatten(0, 1)

    # self.cache[key] = mask
    return mask


# class SpatialAttention2d(nn.Module):
#     def __init__(self, locality: int = 7):
#         super().__init__()
#         self.f = nn.Conv2d(2, 1, locality, padding=locality // 2)

#     def forward(self, x):
#         # b c h w -> b 1 h w
#         Fmax, _ = x.max(dim=1, keepdim=True)
#         # b c h w -> b 1 h w
#         Favg = x.mean(dim=1, keepdim=True)
#         # (b 1 h w) * 2 -> b 1 h w
#         attn = self.f(torch.cat([Favg, Fmax], dim=1))
#         attn = torch.sigmoid(attn)
#         return attn
