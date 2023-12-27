"""Helper modules"""
from collections import OrderedDict, namedtuple
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def parse_model_string(model_string: str):
    """Parse model string to get model related information.

    Model string format:
    ```
    <backbone_name>-<head_name>-<vocab_type>-<language>-<max_width>x<height>[x(min_width)]
    ```

    Examples:
        - `fvtr_t-linear-ctc-vi-512x32`
        - `fvtr_t-rnn-ctc-vi-512x32`
        - `resnet50-attention_rnn-s2s-vi-512x32x28`

    #### Returns (Dictionary):
        `weight_name` (str): Name of weight file.
        `backbone_config` (str): backbone name,
        `head_config` (str): prediction head name,
        `vocab_type` (str): type of vocabulary (ctc or s2s),
        `language` (str): which language to load vocab,
        `max_image_width` (int): Maximum image width
        `min_image_width` (int): Minimum image width, default is 0.75 image height
        `image_height` (int): Image height
    """
    backbone, head, vocab_type, language, size = model_string.split("-")
    sizes = [int(s) for s in size.split("x")]
    if len(sizes) == 2:
        max_width, height = sizes
        min_width = int(height * 0.75)
    else:
        max_width, height, min_width = sizes
    weight_name = f"{model_string}.pt"
    return {
        "weight_name": weight_name,
        "backbone_config": backbone,
        "head_config": head,
        "vocab_type": vocab_type,
        "language": language,
        "max_image_width": max_width,
        "min_image_width": min_width,
        "image_height": height,
    }


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
    image: np.ndarray,
    height: int,
    min_width: int,
    max_width: int,
    rounding: int = 4,
    resampling: int = cv2.INTER_LINEAR,
):
    """Resize input image using variable width and try to keep aspect ratio.

    Args:
        image (np.ndarray): Image to resize
        height (int): The target height
        min_width (int): Minimum target width
        max_width (int): Maximum target width
        rounding (int):
            Round the target width to be dividable by this value.
            Default is 4, which is the model patch width.
        resampling (int): Resampling method, default `cv2.INTER_LINEAR`.
    """
    H, W = image.shape[:2]
    image_size = (W, H)
    width = get_width_for_height(image_size, height)
    width = max(width, min_width)
    width = min(width, max_width)
    width = int(round(width / rounding)) * rounding
    new_size = (width, height)
    image = cv2.resize(image, new_size)
    return image


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
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    lb_image = Image.new("RGB", (width, height), fill_color)
    lb_image.paste(image, (pad_x, pad_y))
    return lb_image


def create_letterbox_cv2(
    image: np.ndarray,
    width: int,
    height: int,
    fill_color: Tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Return a letterboxed version of the provided image to fit specified dimensions.

    This function resizes the input image to fit within a letterbox (a container with fixed width and height) while maintaining the image's aspect ratio. If the image already matches the specified dimensions, the original image is returned.

    The original image is placed in the center of the letterbox.

    Args:
        image (np.ndarray): The input image to be resized and letterboxed.
        width (int): The desired width of the letterbox.
        height (int): The desired height of the letterbox.
        fill_color (Tuple[int, int, int]):
            The RGB color tuple used to fill the letterbox.
            Default: (127, 127, 127).

    Returns:
        Image.Image: The resulting PIL image after the letterboxing process.
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
