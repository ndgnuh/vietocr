"""This module provides tools to process images."""
from functools import partial
from typing import Tuple

import cv2
import numpy as np

_no_input = object()


def create_letterbox_cv2(
    image: np.ndarray,
    width: int,
    height: int,
    fill_color: Tuple[int, int, int] = (127, 127, 127),
) -> np.ndarray:
    """Return a letterboxed version of the provided image to fit specified dimensions.

    This function resizes the input image to fit within a letterbox
    (a container with fixed width and height) while maintaining the
    image's aspect ratio. If the image already matches the specified
    dimensions, the original image is returned.

    The original image is placed in the center of the letterbox.

    Args:
        image (np.ndarray): The input image to be resized and letterboxed.
        width (int): The desired width of the letterbox.
        height (int): The desired height of the letterbox.
        fill_color (Tuple[int, int, int]): The RGB color tuple
            used to fill the letterbox. Default: (127, 127, 127).

    Returns:
        image (np.ndarray): The letterbox image.
    """
    h, w = image.shape[:2]
    if w == width and h == height:
        return image

    # +----------------------------------+
    # | Find new size and padding origin |
    # +----------------------------------+
    src_ratio = w / h
    tgt_ratio = width / height
    if tgt_ratio > src_ratio:
        new_width = int(src_ratio * height)
        new_size = (new_width, height)
        pad_x = int((width - new_width) / 2)
        pad_y = 0
    else:
        new_height = int(width / src_ratio)
        new_size = (width, new_height)
        pad_y = int((height - new_height) / 2)
        pad_x = 0

    # +-----------------------+
    # | resize and letter box |
    # +-----------------------+
    pads = [pad_y, pad_y, pad_x, pad_x]
    image = cv2.resize(image, new_size, cv2.INTER_LANCZOS4)
    image = cv2.copyMakeBorder(
        image, *pads, borderType=cv2.BORDER_CONSTANT, value=fill_color
    )
    return image


def get_width_for_height(
    orig_width: int,
    orig_height: int,
    height: int,
    min_width: int = 0,
    max_width: int = float("inf"),
    align: int = 10,
):
    """Get width for height with some constraints.

    Args:
        orig_width (int): The original width
        orig_height (int): The original height
        height (int): The desired height
        min_width (int): Minimum width to be returned, default 0.
        max_width (int): Maximum width to be returned, default inf.
        align (int): Align the width to this value before returning, default 10.

    Returns:
        (int) The desired width.
    """
    # Find the desired width
    ratio = orig_width / orig_height
    width = height * ratio

    # Limit width range
    width = max(min_width, width)
    width = min(max_width, width)

    # Align width
    width = int(width)
    width = width + (align - width % align)
    return width


def prepare_input(
    image: np.ndarray = _no_input,
    transpose: bool = True,
    normalize: bool = True,
    **sizes,
):
    """Prepare input to be Ocr'ed.

    Args:
        image (np.ndarray): The input image. If not specified,
            this function will behave like a curry function.
        height (int): Desired image height.
        min_width (int): Minimum image width.
        max_width (int): Maximum image width.
        align (int): Align image width to this value.
        transpose (bool): If true, transpose the image to [C, H, W]
            format. The default is true.
        normalize (bool): If true, normalize the image value to 0..1
            range with float32 data type. The default is true.

    Returns:
        If input image is specified, returns a numpy array for
        preprocessed input. Otherwise, the function behave like
        a curry function and returns a function that preprocess
        the image.
    """
    if image is _no_input:
        return partial(prepare_input, **sizes, normalize=normalize, transpose=transpose)

    # Find suitable width
    orig_height, orig_width = image.shape[:2]
    width = get_width_for_height(orig_width, orig_height, **sizes)
    height = sizes["height"]

    # Resize, normalize and transpose
    image = cv2.resize(image, (width, height))
    if normalize:
        image = image.astype("float32") / 255
    if transpose:
        image = image.transpose(2, 0, 1)
    return image
