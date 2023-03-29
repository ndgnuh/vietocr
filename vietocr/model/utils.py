import torch
from torch import nn
from torch.nn import functional as F


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
