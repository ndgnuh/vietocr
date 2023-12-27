from ...tool import patcher
from torchvision import models
from dataclasses import dataclass
from torch import nn
from torchvision import models as vision_models
from functools import partial
import torch


def patched_resnet(resnet):
    def patch(layer):
        patched = patcher.change_layer_param(
            layer, stride=(layer.stride[1], 1))
        return patched

    def condition(layer, name):
        if not isinstance(layer, nn.Conv2d):
            return False

        if 'layer' not in name:
            return False

        if all(s == 1 for s in layer.stride):
            return False

        return True

    net = getattr(vision_models, resnet)(num_classes=1)
    net = patcher.patch_net(net, patch, condition)

    rnet = nn.Sequential()
    for key, module in net.named_children():
        if key == "avgpool":
            break
        rnet.add_module(key, module)

    return rnet


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


class PatchedResnet(nn.Sequential):
    def __init__(self, arch, output_size, dropout):
        super().__init__()
        self.model = patched_resnet(arch)

        with torch.no_grad():
            x = torch.rand(1, 3, 112, 112)
            x = self.model(x)
            c = x.shape[-3]  # channel dim

        self.neck = Neck(c, output_size, dropout)


# How to search for resnet modules:
# for k, v in vars(models).items():
#     if "resnet" in k and callable(v):
#         globals()[f"patched_{k}"] = partial(patched_resnet, resnet=k)

resnet18 = partial(PatchedResnet, arch="resnet18")
resnet34 = partial(PatchedResnet, arch="resnet34")
resnet50 = partial(PatchedResnet, arch="resnet50")
resnet101 = partial(PatchedResnet, arch="resnet101")
resnet152 = partial(PatchedResnet, arch="resnet152")
wide_resnet50_2 = partial(PatchedResnet, arch="wide_resnet50_2")
wide_resnet101_2 = partial(PatchedResnet, arch="wide_resnet101_2")

models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
    "resnet": PatchedResnet
}
