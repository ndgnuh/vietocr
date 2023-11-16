from typing import Tuple

import cv2
import numpy as np

from .custom_augmentations import make_transform


def random_fbm_noise(img: np.ndarray, noise_opacity: Tuple[float, float] = (0.3, 0.7)):
    """Apply fractional brownian noise to the image.

    Args:
        img (np.ndarray): Input image.

    Keywords:
        noise_opacity (Tuple[float, float]): Noise opacity range. Default: (0.3, 0.7).
    """
    h, w, c = img.shape

    # +--------------------+
    # | Generate fBM noise |
    # +--------------------+
    positions = np.indices([h, w]) * 3 / max(h, w)
    positions = positions.reshape(2, h * w).transpose(1, 0)
    noise = gen_fbm(positions).reshape(h, w, 1)
    noise = (np.clip(noise, 0, 1) * 255).astype(np.uint8)

    # +-------+
    # | Blend |
    # +-------+
    mix = np.random.uniform(*noise_opacity)
    noise = np.broadcast_to(noise, (h, w, c))
    w1 = (np.ones([h, w]) * mix).astype(np.float32)
    w2 = 1 - w1
    img = cv2.blendLinear(img, noise, w1, w2)
    return img


def seeded_random(positions, seed0, seed1):
    # generate 1d noise
    noise = np.sum(seed0 * positions, axis=1)
    noise, _ = np.modf(np.sin(noise + seed1))
    return noise


def gen_fbm(self, positions):
    seed0 = np.random.uniform(0, 100, (1, 2))
    seed1 = np.random.uniform(0, 10000)
    value = 0
    amp = np.random.uniform(0.3, 0.7)
    octave = np.random.randint(5, 10)
    for i in range(octave):
        value = value + amp * gen_2d_noise(positions, seed0, seed1)
        positions = positions * 2
        amp = amp * 0.5
    return value


def gen_2d_noise(positions, seed0, seed1):
    i = np.floor(positions)
    f = positions - i

    a = seeded_random(i, seed0, seed1)
    b = seeded_random(i + np.array([[1, 0]]), seed0, seed1)
    c = seeded_random(i + np.array([[0, 1]]), seed0, seed1)
    d = seeded_random(i + np.array([[1, 1]]), seed0, seed1)

    u = f * f * (3 - 2 * f)
    return (
        (a * (1 - u[..., 0]) + b * (u[..., 0]))
        + (c - a) * u[..., 1] * (1 - u[..., 0])
        + (d - b) * u[..., 0] * u[..., 1]
    )


RandomFbmNoise = make_transform("RandomFbmNoise", random_fbm_noise)
