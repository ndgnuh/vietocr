# +-------------------------------------------------------+
# | Augmentation module for fake light effect             |
# | This module could be extended for more shader effects |
# | But i'm too lazy right now                            |
# | Author: ndgnuh                                        |
# +-------------------------------------------------------+
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import albumentations as A
import cv2
import numpy as np

from .custom_augmentations import make_transform, rand


@dataclass
class ShaderBasicLight:
    """Generate color patches from location and width/height.

    This use a very basic shader formula, which is:

    ```
    p = c * (x^n) * (y^m)
    ```

    where c is the color channel value, n and m are the polynomial degree,
    x and y are the normalized patch postion (divided by width/height).

    Args:
        min_deg_x (int): Minimum x-polynomial degree.
        min_deg_y (int): Minimum y-polynomial degree.
        max_deg_x (int): Maximum x-polynomial degree.
        max_deg_y (int): Maximum y-polynomial degree.
        red (Tuple[int, int]): red channel sample range.
        green (Tuple[int, int]): green channel sample range.
        blue (Tuple[int, int]): blue channel sample range.
    """

    min_deg_x: int = 0
    min_deg_y: int = 0
    max_deg_x: int = 3
    max_deg_y: int = 3
    red: Tuple = (0, 255)
    green: Tuple = (0, 255)
    blue: Tuple = (0, 255)

    def __post_init__(self):
        self.deg_x = random.randint(self.min_deg_x, self.max_deg_x)
        self.deg_y = random.randint(self.min_deg_y, self.max_deg_y)
        self.flip_x = random.choice((True, False))
        self.flip_y = random.choice((True, False))
        self.r = random.choice(self.red)
        self.g = random.choice(self.green)
        self.b = random.choice(self.blue)

    def __call__(self, x, y, w, h):
        deg_x = self.deg_x
        deg_y = self.deg_y
        px = x**deg_x / w**deg_x
        py = y**deg_y / h**deg_y
        if self.flip_x:
            px = 1 - px
        if self.flip_y:
            py = 1 - py
        r = self.r * (1 - px * py)
        g = self.g * (1 - px * py)
        b = self.b * (1 - px * py)
        return (int(r), int(g), int(b))


def fake_light(
    image: np.ndarray,
    tile_size: int = (20, 50),
    alpha: Tuple[float, float] = (0.2, 0.6),
):
    """Create fake light effect using color patches with increasing or
    decreasing intensities.

    Args:
        image (np.ndarray): The input image.
        tize_size (int | Tuple[int, int]): The size of the color patches. Default: (20, 50).
        alpha (float | Tuple[float, float]): Overlay opacity, must be in (0, 1) range. Default: (0.2, 0.6).
    """
    # +---------+
    # | Prepare |
    # +---------+
    H, W = image.shape[:2]
    shader_fn = ShaderBasicLight()
    tile_size = rand(tile_size)
    alpha = rand(alpha)

    # +-----------------------------------+
    # | Convert image to RGB if it's gray |
    # +-----------------------------------+
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    # +-------+
    # | Tiles |
    # +-------+
    canvas = np.zeros((H, W, 3))
    for x in range(0, W, tile_size):
        for y in range(0, H, tile_size):
            br = shader_fn(x, y, W, H)
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            cv2.rectangle(canvas, (x, y), (x2, y2), br, -1)

    # +-----------------+
    # | alpha composite |
    # +-----------------+
    image = (image * (1 - alpha) + canvas * alpha).round().astype(image.dtype)
    return image


FakeLight = make_transform("FakeLight", fake_light)
