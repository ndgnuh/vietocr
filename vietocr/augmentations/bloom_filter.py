import random
from functools import partial
from typing import Tuple, Union

import cv2
import numpy as np

from .custom_augmentations import make_transform


def bloom_filter(
    img: np.ndarray,
    white_threshold: Union[Tuple[int, int], int] = (220, 240),
    blur: Union[Tuple[int, int], int] = (2, 15),
    gain: Union[Tuple[int, int], int] = (0.3, 3),
):
    """Apply bloom filter to the image

    The option for this function is either a number or a tuple `(a, b)`.
    If a tuple is used, then the number will sampled randomly inside the
    range given by `a` and `b`.

    Args:
        img (np.ndarray): Input image

    Keywords:
        white_threshold (Tuple[int, int] | int):
            Min threshold for pixels to be considered white.
            Default: (220, 240).
        blur (Tuple[float, float] | int):
            Sigma values for the gaussian blur.
            Default: (2, 15).
        gain (Tuple[float, float] | float):
            How much the bloom is weighting when blending with the original image.
            Default: (0.3, 3).
    """
    # +-------------+
    # | adapt input |
    # +-------------+
    if isinstance(white_threshold, (list, tuple)):
        white_threshold = random.uniform(*white_threshold)
    if isinstance(blur, (list, tuple)):
        blur = random.uniform(*blur)
    if isinstance(gain, (list, tuple)):
        gain = random.uniform(*gain)

    # +-------------------------------------------+
    # | convert image to hsv colorspace as floats |
    # +-------------------------------------------+
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    h, s, v = cv2.split(hsv)

    # +-----------------------------------------------------+
    # | Desire low saturation and high brightness for white |
    # | So invert saturation and multiply with brightness   |
    # +-----------------------------------------------------+
    sv = ((255 - s) * v / 255).clip(0, 255).astype(np.uint8)

    # +-----------------+
    # | threshold image |
    # +-----------------+
    thresh = cv2.threshold(sv, white_threshold, 255, cv2.THRESH_BINARY)[1]

    # +--------------------------+
    # | blur and make 3 channels |
    # +--------------------------+
    blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=blur, sigmaY=blur)
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # +-----------------------------------------+
    # | blend blur and image using gain on blur |
    # +-----------------------------------------+
    result = cv2.addWeighted(img, 1, blur, gain, 0)

    # +--------------+
    # | output image |
    # +--------------+
    return result


BloomFilter = make_transform("BloomFilter", bloom_filter)
