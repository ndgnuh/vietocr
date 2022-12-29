from ...tool import patcher
from torchvision import models
from dataclasses import dataclass
from torch import nn
from torchvision import models as vision_models
from functools import partial
import torch


def patched_mobilenet(arch):
    def patch(layer, name):
        return patcher.change_layer_param(layer, stride=(layer.stride[0], 1))

    def condition(layer, name):
        if not isinstance(layer, nn.Conv2d):
            return False
        if all(s == 1 for s in layer.stride):
            return False

        return True

    def patch_remove_se(layer):
        return nn.Identity()

    def condition_remove_se(layer, name):
        return type(layer).__name__ == "SqueezeExcitation"

    net = getattr(vision_models, arch)(num_classes=1).features
    net = patcher.patch_net(net, patch, condition)
    # net = patcher.patch_net(net, patch_remove_se, condition_remove_se)
    return net


class Neck(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.project = nn.Conv2d(input_size, output_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.project(x)

        # h w -> w h
        x = x.transpose(-1, -2)
        # w h -> t
        x = x.flatten(2)
        # n c t -> t n c
        x = x.permute(-1, 0, 1)

        return x


class PatchedMobileNet(nn.Sequential):
    def __init__(self, arch, output_size, dropout):
        super().__init__()
        self.model = patched_mobilenet(arch)

        with torch.no_grad():
            x = torch.rand(1, 3, 112, 112)
            x = self.model(x)
            c = x.shape[-3]  # channel dim

        self.neck = Neck(c, output_size, dropout)


mobilenet_v3_large = partial(PatchedMobileNet, arch="mobilenet_v3_large")
mobilenet_v3_small = partial(PatchedMobileNet, arch="mobilenet_v3_small")
