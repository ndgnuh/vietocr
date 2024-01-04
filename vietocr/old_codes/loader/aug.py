import albumentations as A
from dataclasses import dataclass
from os import path, listdir
from typing import List, Tuple
import random
import numpy as np
import cv2


@dataclass
class FBMNoise:
    # Noise generated by fractional brownian motion
    p: float = 0.5
    mix: Tuple[float, float] = (0.6, 0.8)

    def __call__(self, **kw):
        if random.uniform(0, 1) >= self.p:
            return kw

        image = kw["image"]
        h, w, c = image.shape

        positions = np.indices([h, w]) * 3 / max(h, w)
        positions = positions.reshape(2, h * w).transpose(1, 0)

        noise = self.gen_fbm(positions).reshape(h, w, 1)

        mix = np.random.uniform(*self.mix)
        image = image * mix + noise * (1 - mix)
        image = image.clip(0, 1)
        image = (image * 255).astype('uint8')
        kw["image"] = image
        return kw

    def random(self, positions, seed0, seed1):
        # generate 1d noise
        noise = np.sum(seed0 * positions, axis=1)
        noise, _ = np.modf(np.sin(noise + seed1))
        return noise

    def gen_fbm(self, positions):
        seed0 = np.random.uniform(0, 100, (1, 2))
        seed1 = np.random.uniform(0, 10000)
        value = 0
        amp = np.random.uniform(0.3, 0.7)
        frequency = np.random.rand()
        octave = np.random.randint(5, 10)
        for i in range(octave):
            value = value + amp * self.gen_2d_noise(positions, seed0, seed1)
            positions = positions * 2
            amp = amp * 0.5
        return value

    def gen_2d_noise(self, positions, seed0, seed1):
        i = np.floor(positions)
        f = positions - i

        a = self.random(i, seed0, seed1)
        b = self.random(i + np.array([[1, 0]]), seed0, seed1)
        c = self.random(i + np.array([[0, 1]]), seed0, seed1)
        d = self.random(i + np.array([[1, 1]]), seed0, seed1)

        u = f * f * (3 - 2 * f)
        return (
            (a * (1 - u[..., 0]) + b * (u[..., 0])) +
            (c - a) * u[..., 1] * (1 - u[..., 0]) +
            (d - b) * u[..., 0] * u[..., 1]
        )


def color_overlay(image, color, rate=0.7):
    image = image * rate + color * (1 - rate)
    return image


def random_color_overlay(image):
    h, w, c = image.shape
    num_h_splits = random.randrange(1, 5)
    num_v_splits = random.randrange(3, 20)
    s_height = h // num_h_splits
    s_width = w // num_v_splits
    for i in range(num_h_splits):
        for j in range(num_v_splits):
            h_slice = slice(i * s_height, (i + 1) * s_height)
            v_slice = slice(j * s_width, (j + 1) * s_width)
            color = np.random.randint(0, 256, (1, 1, 3))
            image[h_slice, v_slice, :] = color_overlay(
                image[h_slice, v_slice, :],
                rate=random.uniform(0.7, 0.9),
                color=color
            )
    return image


class RandomColorOverlay:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, **kwargs):
        if random.uniform(0, 1) < self.p:
            return kwargs

        kwargs['image'] = random_color_overlay(kwargs['image'])
        return kwargs


class RandomOrderCompose:
    def __init__(self, *a, **kw):
        self.compose = A.Compose(*a, **kw)

    def __call__(self, *a, **kw):
        random.shuffle(self.compose.transforms)
        return self.compose(*a, **kw)


@dataclass
class SequentialOneOf:
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p
        self.index = 0
        self.len = len(transforms)

    def __call__(self, **kw):
        if random.uniform(0, 1) >= p:
            return kw

        transform = self.transforms[self.index]
        output = transform(**kw)

        self.index = (self.index + 1) % self.len
        return output


def scale_degrade(image, scale):
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w/scale), int(h/scale)))
    image = cv2.resize(image, (w, h))
    return image


@dataclass
class ScaleDegrade:
    min_scale: float = 1.2
    max_scale: float = 2.2
    p: float = 0.5

    def __call__(self, **kw):
        scale = random.uniform(self.min_scale, self.max_scale)

        image = kw['image']
        image = scale_degrade(image, scale)

        kw['image'] = image
        return kw


def random_crop(pattern, width, height):
    ph, pw = pattern.shape[:2]
    x0 = random.randint(0, pw - width)
    y0 = random.randint(0, ph - height)
    x1 = x0 + width
    y1 = y0 + height
    return pattern[y0:y1, x0:x1]


class PatternOverlay:
    def __init__(
        self,
        patterns: str,
        alphas=(0.7, 0.9),
        p: float = 0.5,
    ):
        if path.isdir(patterns):
            self.patterns = [cv2.imread(path.join(patterns, pattern))
                             for pattern in listdir(patterns)]
        elif path.isfile(patterns):
            self.patterns = [cv2.imread(patterns)]
        self.alphas = alphas
        self.p = p

    def __call__(self, image: np.ndarray, **kw):
        # **kw because of albumentation interface
        if random.random() > self.p:
            return dict(image=image, **kw)
        alpha = random.uniform(*self.alphas)
        pattern = random.choice(self.patterns)
        h, w = image.shape[:2]
        try:
            pattern = random_crop(pattern, width=w, height=h)
        except ValueError:
            pattern = cv2.resize(pattern, (w, h))
        image = image * alpha + pattern * (1 - alpha)
        image = np.clip(image, 0, 255).round().astype('uint8')
        kw['image'] = image
        return kw


def blur_shift(image, shift, vertical):
    if vertical:
        image[shift:, ...] = image[shift:, ...] * \
            0.5 + image[:-shift, ...] * 0.5
    else:
        image[:, shift:, ...] = image[:, shift:, ...] * \
            0.5 + image[:, :-shift, ...] * 0.5
    return image


class RandomBlurShift:
    def __call__(self, **kw):
        image = kw['image']
        verticals = random.choices([True, False], k=2)
        for shift, vertical in enumerate(verticals):
            image = blur_shift(image, shift + 1, vertical)

        kw['image'] = image
        return kw


p = 0.5
default_augment = RandomOrderCompose([
    # Overlay
    SequentialOneOf([
        RandomColorOverlay(p=1),
        PatternOverlay(patterns="vietocr/data/patterns", p=1),
    ], p=p),

    # Changing image coloring
    SequentialOneOf([
        A.CLAHE(always_apply=True),
        A.ColorJitter(always_apply=True),
        A.Emboss(always_apply=True),
        A.HueSaturationValue(always_apply=True),
        A.RandomBrightnessContrast(always_apply=True),
        A.InvertImg(always_apply=True),
        A.RGBShift(always_apply=True),
        A.ToSepia(always_apply=True),
        A.ToGray(always_apply=True),
    ], p=p),

    # Overlays
    # Fog, snow, sunflare are disabled
    # due to deadlock bug and readability
    # https://github.com/albumentations-team/albumentations/issues/361
    A.RandomShadow(p=p),

    # Noises
    SequentialOneOf([
        FBMNoise(p=1.2),
        A.ISONoise(always_apply=True),
        A.MultiplicativeNoise(always_apply=True),
    ], p=p),

    # Dropouts
    SequentialOneOf([
        A.PixelDropout(always_apply=True),
        A.ChannelDropout(always_apply=True),
    ], p=p),

    # Image degration
    SequentialOneOf([
        A.ImageCompression(always_apply=True),
        ScaleDegrade(p=10),
        A.GaussianBlur(always_apply=True),
        A.Defocus(radius=(1, 3), always_apply=True),
        A.Posterize(always_apply=True),
        A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, always_apply=True),
        A.MedianBlur(blur_limit=3, always_apply=True),
        RandomBlurShift(),
        A.MotionBlur(always_apply=True),
        A.ZoomBlur(max_factor=1.1, always_apply=True),
    ], p=p),

    # Spatial transform
    SequentialOneOf([
        A.ElasticTransform(alpha=1, sigma=1, alpha_affine=1,
                           always_apply=True),
        A.Perspective(fit_output=True, always_apply=True),

        # 4/2 is divisable by the image size
        # in hope that it will not cause any crash
        # Still crash
        # A.PiecewiseAffine(nb_rows=4, nb_cols=2, always_apply=True),

        # Removed due to making the output out of range
        # A.ShiftScaleRotate(always_apply=True),
        # A.SafeRotate((-10, 10), always_apply=True),
    ], p=p),
])