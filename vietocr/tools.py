"""Helper modules"""
from collections import OrderedDict
from typing import Tuple

import numpy as np
from PIL import Image


def get_width_for_height(image_wh: Tuple[int, int], desired_height: int) -> int:
    """Compute expected width for some height according to ratio.

    Args:
        image_wh (Tuple[int, int]): Image width and height.
        desired_height (int): Desired height.

    Returns:
        width (int): expected width for `desired_height`.
    """
    W, H = image_wh
    ratio = W / H
    return int(desired_height * ratio)


def resize_image(
    image: Image.Image,
    height: int,
    min_width: int,
    max_width: int,
    rounding: int = 10,
    resampling: Image.Resampling = Image.Resampling.LANCZOS,
):
    """Resize input image using variable width and try to keep aspect ratio.

    Args:
        image (PIL.Image.Image): Image to resize
        height (int): The target height
        min_width (int): Minimum target width
        max_width (int): Maximum target width
        rounding (int): Round the target width to be dividable by this value.
        resampling (PIL.Image.Resampling): Resampling method, default `PIL.Image.Resampling.LANCZOS`.
    """
    image_size = image.size
    width = get_width_for_height(image_size, height)
    width = max(width, min_width)
    width = min(width, max_width)
    width = int(width / rounding) * rounding
    new_size = (width, height)
    return image.resize(new_size, resampling)


def load_state_dict_whatever(model, state_dict: OrderedDict):
    """Load state dict, without complaining about shapes or missing keys.

    THIS FUNCTION WILL ALTER THE MODEL WEIGHTS.

    This functions try to load every key. It will ignore:
    - missing keys: keys that appear in the model's state dictionary but not in the provided one
    - keys with mismatch shapes: keys that appear in both state dictionaries but their values have different shapes
    - residual keys: keys that do not appear in the model's state dictionary but in the input state dict

    Args:
        model (torch.nn.Module): The torch module to load state dict
        state_dict (OrderedDict): The state dict

    Returns:
        missing_keys (List[str]): Self-explanatory
        mismatch_keys (List[str]): Self-explanatory
        residual_keys (List[str]): Self-explanatory
    """
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    input_keys = set(state_dict.keys())

    missing_keys = list(model_keys - input_keys)
    residual_keys = list(input_keys - model_keys)
    candidate_keys = model_keys.intersection(input_keys)
    mismatch_keys = []

    # Replace with suitable candidate values
    for key in candidate_keys:
        if model_state_dict[key].shape != state_dict[key].shape:
            mismatch_keys.append(key)
            continue
        model_state_dict[key] = state_dict[key]

    # Load state dict
    model.load_state_dict(model_state_dict)
    return missing_keys, mismatch_keys, residual_keys


def create_letterbox_pil(
    image: Image.Image,
    width: int,
    height: int,
    fill_color: Tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Return a letterboxed version of the provided image to fit specified dimensions.

    This function resizes the input image to fit within a letterbox (a container with fixed width and height) while maintaining the image's aspect ratio. If the image already matches the specified dimensions, the original image is returned.

    The original image is placed in the center of the letterbox.

    Args:
        image (Image.Image): The input PIL image to be resized and letterboxed.
        width (int): The desired width of the letterbox.
        height (int): The desired height of the letterbox.
        fill_color (Tuple[int, int, int]):
            The RGB color tuple used to fill the letterbox.
            Default: (127, 127, 127).

    Returns:
        Image.Image: The resulting PIL image after the letterboxing process.
    """
    w, h = image.size
    if w == width and h == height:
        return image

    # Find new size and padding origin
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

    # resize and letter box
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    lb_image = Image.new("RGB", (width, height), fill_color)
    lb_image.paste(image, (pad_x, pad_y))
    return lb_image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy normalized, channel-first tensor format.

    Useful for ONNX and for using with torch.

    Args:
        image (Image.Image): The input PIL image to be converted to a NumPy array.

    Returns:
        image_np (np.ndarray):
            The resulting NumPy array representing the image.
            The shape of the array is [C, H, W], where C represents channels,
            H represents height, and W represents width.
            The value are normalized and converted to `np.float32`.
    """
    image = np.array(image, dtype=np.float32) / 255
    # [H, W, C] -> [C, H, W]
    image = image.transpose(2, 0, 1)
    return image
