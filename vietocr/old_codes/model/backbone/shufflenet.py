from ...tool import patcher
from torchvision import models
from dataclasses import dataclass
from torch import nn
from torchvision import models as vision_models
from functools import partial
import torch


def patched_shufflenet(arch):
    def patch(layer):
        return patcher.change_layer_param(layer, stride=(layer.stride[0], 1))

    def condition(layer, name):
        if not isinstance(layer, nn.Conv2d):
            return False
        if "stage" not in name:
            return False
        return any(s != 1 for s in layer.stride)

    net_ = getattr(vision_models, arch)(num_classes=1)
    net = nn.Sequential()
    for name, module in net_.named_children():
        if name != "fc" and name != "conv5":
            net.add_module(name, module)
    net = patcher.patch_net(net, patch, condition)
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


class PatchedShufflenet(nn.Sequential):
    def __init__(self, arch, output_size, dropout):
        super().__init__()
        self.model = patched_shufflenet(arch)

        with torch.no_grad():
            x = torch.rand(1, 3, 112, 112)
            x = self.model(x)
            c = x.shape[-3]  # channel dim

        self.neck = Neck(c, output_size, dropout)


shufflenet_v2_x0_5 = partial(PatchedShufflenet, arch="shufflenet_v2_x0_5")
shufflenet_v2_x1_0 = partial(PatchedShufflenet, arch="shufflenet_v2_x1_0")
shufflenet_v2_x1_5 = partial(PatchedShufflenet, arch="shufflenet_v2_x1_5")
shufflenet_v2_x2_0 = partial(PatchedShufflenet, arch="shufflenet_v2_x2_0")

models = {
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
    "shufflenet": PatchedShufflenet
}
