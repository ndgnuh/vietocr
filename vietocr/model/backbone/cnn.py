import torch
from torch import nn

import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
from vietocr.model.backbone.mobilenet import (
    mobilenet_v3_large,
    mobilenet_v3_small
)

backbone_by_names = {
    'vgg11_bn': vgg.vgg11_bn,
    'vgg19_bn': vgg.vgg19_bn,
    'resnet50': Resnet50,
    'mobilenet_v3_large': mobilenet_v3_large,
    'mobilenet_v3_small': mobilenet_v3_small,
}


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        self.model = backbone_by_names[backbone](**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
