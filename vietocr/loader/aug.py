from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torch import distributions
import torch
import random

from dataclasses import dataclass
from typing import Callable


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
    Sometime(T.Grayscale(3)),
    Sometime(T.GaussianBlur(5)),
    Sometime(T.RandomInvert(1)),
    Sometime(T.RandomSolarize(1)),
    T.RandomApply([
        Sometime(PatchNoise(3)),
        Sometime(PatchNoise(5)),
        Sometime(PatchNoise(7)),
        Sometime(PatchNoise(11)),
        Sometime(PatchNoise(17)),
    ]),
    Sometime(T.RandomPerspective(0.01)),
    Sometime(T.RandomAffine(
        degrees=(-3, 3),
        translate=(0, 0.1)
    ))


], p=1)
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
