from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torch import distributions
import torch
import random

from dataclasses import dataclass
from typing import Callable
from itertools import product
from torch import nn


class MotionBlur(nn.Module):
    def __init__(self, kernel_size: int, direction: str, reflect=False):
        super().__init__()
        kernel = torch.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        value = 1 / kernel_size
        if direction == "v":
            kernel[:, kernel_size // 2] = value
        elif direction == "h":
            kernel[kernel_size // 2, :] = value
        elif direction == "d":
            kernel[kernel_size // 2, :center] = value
            kernel[:center + 1, kernel_size // 2] = value
        else:
            raise ValueError(
                "Unsupported direction in MotionBlur, supported directions are v,h,d")
        if reflect and direction == "d":
            kernel = torch.flipud(torch.fliplr(kernel))

        self.conv = nn.Conv2d(
            3, 3, kernel_size, padding=kernel_size // 2, bias=False, groups=3)
        self.conv.weight.data, _ = torch.broadcast_tensors(
            kernel,
            self.conv.weight.data
        )

        # Don't train it
        for p in self.conv.parameters():
            p.require_grad = False

    @torch.no_grad()
    def forward(self, image):
        image = self.conv(image)
        return image


def RandomMotionBlur(sizes, directions=("h", "v")):
    return T.RandomChoice([
        MotionBlur(size, direction)
        for size, direction in product(sizes, directions)
    ])


def ensure_tensor(img):
    if isinstance(img, torch.Tensor):
        return img
    else:
        return TF.to_tensor(img)


class PatchNoise:
    def __init__(self, patch_size=4, dist=None):
        self.patch_size = (patch_size, patch_size) if isinstance(
            patch_size, (int, float)) else patch_size

        if dist is None:
            dist = distributions.Normal(0, 0.05)
        self.dist = dist

    def __call__(self, image):
        mean = image.mean(dim=(-1, -2))
        std = image.std(dim=(-1, -2))
        h, w = image.shape[-2:]
        rescale = T.Resize((h, w), T.InterpolationMode.NEAREST)
        h = max(int(h // self.patch_size[0]), 1)
        w = max(int(w // self.patch_size[1]), 1)
        shape = (*image.shape[:-2], h, w)
        noise = self.dist.sample(sample_shape=shape).to(image.device)
        noise = rescale(noise)
        return torch.clamp(image + noise, 0, 1)


@ dataclass
class Sometime:
    tf: Callable
    chance: float = 0.3

    def __call__(self, image):
        if random.random() <= self.chance:
            return self.tf(image)
        else:
            return image


# List of augments
# Apply with chance, 0.3
# Select 1 to 5 out of the below list
# Blur: Gaussian or Motion
# Sigmoid Contrats?
# Invert
# Solarize
# Multiply?
# Add?
# JPEG Compression
# Random crop (0.01, 0.05)
# Perspective transform (0.01, 0.01)
# Affine transform (0.7, 1.3), translate(-0.1, 0.1)
# PiecewiseAffine?
# Dropout?


chance = 0.3
default_augment = T.RandomApply([
    T.RandomChoice([
        T.GaussianBlur(3),
        T.GaussianBlur(5),
        T.GaussianBlur(7),
        T.GaussianBlur(11),
        T.GaussianBlur(13),
        RandomMotionBlur([3, 5, 7, 9, 11, 13, 17], ["d", "v", "h"])
    ]),
    T.RandomChoice([
        T.Grayscale(3),
        T.RandomInvert(1),
        T.RandomSolarize(1),
    ]),
    T.RandomChoice([
        PatchNoise(3),
        PatchNoise(5),
        PatchNoise(7),
        PatchNoise(11),
        PatchNoise(17),
    ]),
    T.RandomPerspective(0.01),
    T.RandomAffine(
        degrees=(-5, 5),
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
        shear=(-7, 7),
    )
])
default_augment = T.Compose([
    ensure_tensor,
    default_augment,
])


# class ImgAugTransform:
#     def __init__(self):
#         def sometimes(aug): return iaa.Sometimes(0.3, aug)

#         self.aug = iaa.Sequential(iaa.SomeOf((1, 5),
#                                              [
#             # blur

#             sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
#                                  iaa.MotionBlur(k=3)])),

#             # color
#             sometimes(iaa.AddToHueAndSaturation(
#                 value=(-10, 10), per_channel=True)),
#             sometimes(iaa.SigmoidContrast(gain=(3, 10),
#                                           cutoff=(0.4, 0.6), per_channel=True)),
#             sometimes(iaa.Invert(0.25, per_channel=0.5)),
#             sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
#             sometimes(iaa.Dropout2d(p=0.5)),
#             sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
#             sometimes(iaa.Add((-40, 40), per_channel=0.5)),

#             sometimes(iaa.JpegCompression(compression=(5, 80))),

#             # distort
#             sometimes(iaa.Crop(percent=(0.01, 0.05),
#                                sample_independently=True)),
#             sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
#             sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
#                                  #                            rotate=(-5, 5), shear=(-5, 5),
#                                  order=[0, 1], cval=(0, 255),
#                                  mode=ia.ALL)),
#             sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
#             sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                  iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])),

#         ],
#             random_order=True),
#             random_order=True)

#     def __call__(self, img):
#         img = np.array(img)
#         img = self.aug.augment_image(img)
#         img = Image.fromarray(img)
#         return img


class PatchNoise:
    def __init__(self, patch_size=4, dist=None):
        self.patch_size = (patch_size, patch_size) if isinstance(
            patch_size, (int, float)) else patch_size

        if dist is None:
            dist = distributions.Normal(0, 0.05)
        self.dist = dist

    def __call__(self, image):
        mean = image.mean(dim=(-1, -2))
        std = image.std(dim=(-1, -2))
        h, w = image.shape[-2:]
        rescale = T.Resize((h, w), T.InterpolationMode.NEAREST)
        h = int(h // self.patch_size[0])
        w = int(w // self.patch_size[1])
        shape = (*image.shape[:-2], h, w)
        noise = self.dist.sample(sample_shape=shape).to(image.device)
        noise = rescale(noise)
        return torch.clamp(image + noise, 0, 1)
