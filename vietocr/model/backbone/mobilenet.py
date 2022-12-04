from ...tool.patcher import (
    patch_net,
    change_layer_param,
    change_layer_type
)
from ..reshape import Reshape, Permute
from torch import nn
from torchvision import models
from dataclasses import dataclass


@dataclass
class PatchStride:
    skip: int = 2

    def patch(self, module, name, idx):
        # Skip the first two layers
        if idx < self.skip:
            return module

        # CHange the stride to (stride_v, 1)
        stride = module.stride
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        stride = (stride[0], 1)
        return change_layer_param(module, stride=stride)

    def patch_condition(self, module, name):
        # Only patch Conv2d layers
        if not isinstance(module, nn.Conv2d):
            return False

        # Only patch strided layers
        stride = module.stride
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        if stride[1] == 1:
            return False
        return True


def mobilenet_v3_large(*args, hidden, dropout, **kwargs):
    patch = PatchStride(skip=2)
    net = models.mobilenet_v3_large(num_classes=1).features
    net = patch_net(net, patch.patch, patch.patch_condition)
    net = nn.Sequential(
        net,
        nn.Dropout(dropout),
        nn.Conv2d(960, hidden, 1),
        Reshape('n,c,h,w->n,c,h*w'),
        Permute('n,c,s->s,n,c')
    )
    return net


def mobilenet_v3_small(*args, hidden, dropout, **kwargs):
    patch = PatchStride(skip=2)
    net = models.mobilenet_v3_large(num_classes=1).features
    net = patch_net(net, patch.patch, patch.patch_condition)
    net = nn.Sequential(
        net,
        nn.Dropout(dropout),
        nn.LazyConv2d(hidden, 1),
        Reshape('n,c,h,w->n,c,h*w'),
        Permute('n,c,s->s,n,c')
    )
    return net
