import torch
from torch import nn

import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone import resnet, shufflenet, fvtr, fvtr_v2
from vietocr.model.backbone.efficientnet import PatchedEfficientNet
from vietocr.model.backbone.svtr import (
    svtr_b,
    svtr_l,
    svtr_s,
    svtr_t,
    RecSVTR
)
from vietocr.model.backbone.mobilenet import (
    mobilenet_v3_large,
    mobilenet_v3_small
)
from .inception import PatchedInception

backbone_by_names = {
    'vgg11_bn': vgg.vgg11_bn,
    'vgg19_bn': vgg.vgg19_bn,
    'mobilenet_v3_large': mobilenet_v3_large,
    'mobilenet_v3_small': mobilenet_v3_small,
    "svtr_b": svtr_b,
    "svtr_l": svtr_l,
    "svtr_s": svtr_s,
    "svtr_t": svtr_t,
    "rsvtr": RecSVTR,
    "efficientnet": PatchedEfficientNet,
    "inception": PatchedInception
}
backbone_by_names.update(resnet.models)
backbone_by_names.update(shufflenet.models)
backbone_by_names.update(fvtr.models)
backbone_by_names.update(fvtr_v2.models)


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        self.model = backbone_by_names[backbone](**kwargs)

    def forward(self, x):
        out = self.model(x)
        # from icecream import install
        # install()
        # ic(out.shape)
        return out

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
