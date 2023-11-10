import copy
from typing import Optional

import torch
from torch import autograd, nn


class ReversibleFN(autograd.Function):
    @staticmethod
    def forward(ctx, Fm, Gm, x, *params):
        x = x.detach()
        with torch.no_grad():
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
            y1 = x1 + Fm(x2)
            y2 = x2 + Gm(y1)
            y = torch.cat((y1, y2), dim=1)
            del x1, x2, y1, y2
        ctx.Fm = Fm
        ctx.Gm = Gm
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        Fm = ctx.Fm
        Gm = ctx.Gm
        Fparams = tuple(Fm.parameters())
        Gparams = tuple(Gm.parameters())

        x = ctx.saved_tensors[0]
        x1, x2 = torch.chunk(x, 2, dim=1)

        # compute outputs building a sub-graph
        with torch.set_grad_enabled(True):
            x1.requires_grad = True
            x2.requires_grad = True
            y1 = x1 + Fm(x2)
            y2 = x2 + Gm(y1)
            y = torch.cat([y1, y2], dim=1)

            inputs = (x1, x2) + Fparams + Gparams
            grads = autograd.grad(y, inputs, grad_output)
        grad_input = torch.cat([grads[0], grads[1]], dim=1)
        return (None, None, grad_input) + tuple(grads[2:])


class Reversible(nn.Module):
    def __init__(self, Fm: nn.Module, Gm: Optional[nn.Module] = None):
        super().__init__()
        self.Fm = Fm
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm

    def forward(self, x):
        if self.training:
            params = list(self.Fm.parameters()) + list(self.Gm.parameters())
            y = ReversibleFN.apply(self.Fm, self.Gm, x, *params)
        else:
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
            y1 = x1 + self.Fm(x2)
            y2 = x2 + self.Gm(y1)
            y = torch.cat((y1, y2), dim=1)
        return y
