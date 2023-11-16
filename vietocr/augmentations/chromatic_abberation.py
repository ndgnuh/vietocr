# +----------------------------------------------------+
# | Custom augmentation module for chromatic abbration |
# | Author: ndgnuh                                     |
# +----------------------------------------------------+
import random
from functools import partial
from typing import Tuple

import albumentations as A
import cv2
import numpy as np

from .custom_augmentations import make_transform, rand


def chromatic_aberration(img, shift: Tuple[int, int] = (-10, 10)):
    """Apply chromatic aberration effect on the input image.

    This is done by resizing each channel with a random shift, then
    paste all the resized channel onto one target image. The target
    image is then resized to the original image size.

    Args:
        img (np.ndarray): The input image
        shift (Tuple[int, int] | int): The shift value or shift range. Default: (-10, 10).
    """
    # +-------------------------+
    # | Split image to channels |
    # +-------------------------+
    cs = cv2.split(img)
    h, w = img.shape[:2]
    mw, mh = w, h

    # +-----------------------------+
    # | Shift each channel randomly |
    # +-----------------------------+
    channels = []
    for i, chan in enumerate(cs):
        # +---------------------+
        # | random x y shifting |
        # +---------------------+
        dx = int(rand(shift))
        dy = int(rand(shift))

        # +---------------+
        # | New dimension |
        # +---------------+
        nw = w + dx
        nh = h + dy

        # +---------------+
        # | Max dimension |
        # +---------------+
        mw = max(mw, nw)
        mh = max(mh, nh)

        # +-------------------------+
        # | Resize to new dimension |
        # +-------------------------+
        chan = cv2.resize(chan, (nw, nh))
        channels.append(chan)

    # +----------------------------------------+
    # | Paste shifted channels to target image |
    # +----------------------------------------+
    newimg = np.ones((mh, mw, 3), dtype="uint8")
    for i, chan in enumerate(channels):
        ch, cw = chan.shape[:2]
        newimg[:ch, :cw, i] = chan
    newimg = newimg[:mh, :mw]

    # +--------------------------------------+
    # | Resize target image to original size |
    # +--------------------------------------+
    newimg = cv2.resize(newimg, (w, h))
    return newimg


ChromaticAberration = make_transform("ChromaticAberration", chromatic_aberration)
