from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

# The __init__ is allowed to use code from outside
from ..configs import OcrConfig
from ..images import create_letterbox_cv2
from .dataset_dsrecord import DsrecordOcrDataset, VietocrOcrDataset
from .samplers import SameSizeSampler


def load_data(data_path: str, **kwargs):
    """Load data from configuration.

    If the data path is a string:
    - the path endswith `.rec` -> Load OcrRecordDataset
    - the path endswith `.txt` -> Load OcrVietocrDataset
    If data path is a dict:
    - use the 'type' key to determine the dataset, and use other keywords as arguments
    """
    if isinstance(data_path, str):
        if data_path.endswith(".rec"):
            return DsrecordOcrDataset(data_path, **kwargs)
        elif data_path.endswith(".txt"):
            return VietocrOcrDataset(data_path, **kwargs)
        else:
            raise ValueError("Dataset not implemented")
    if data_path == "pretrain":
        return
    else:
        raise ValueError(f"Unsupported dataset {data_path}")


def collate_variable_width(samples: List, pad_token_id: int = 0):
    """Collate samples with variable image widths.

    It performs letterbox on the smaller images so that all of them are equal widths.
    Then, the images can be stacked together. After that, the labels are padded to some
    maximum length, the label sizes are stored for the loss calculation.

    Args:
        samples (list): List of tuple of (image, token_ids).
        pad_token_id (int): Pad value for targets.
    """
    pad_id = pad_token_id

    # +----------------------+
    # | Find out max metrics |
    # +----------------------+
    max_height = -1
    max_width = -1
    max_target_length = -1
    for image, target in samples:
        # Max image width
        max_width = max(image.shape[1], max_width)
        max_height = max(image.shape[0], max_height)

        # Max target length
        target_length = len(target)
        max_target_length = max(target_length, max_width)

    # +--------------------------------+
    # | Padding + convert to columnars |
    # +--------------------------------+
    images = []
    targets = []
    target_lengths = []
    for image, target in samples:
        # +-----------------------------------------------------------+
        # | Process image, letterbox and to move channel to first dim |
        # +-----------------------------------------------------------+
        image = create_letterbox_cv2(image, max_width, max_height)
        image = (image.astype("float32") / 255).transpose(2, 0, 1)
        images.append(image)

        # +----------------+
        # | Process target |
        # +----------------+
        target_length = len(target)
        pad_length = max_target_length - target_length
        target = target + [pad_id] * pad_length
        targets.append(target)
        target_lengths.append(target_length)

    # +--------------------+
    # | Convert to tensors |
    # +--------------------+
    images = torch.tensor(np.stack(images, axis=0))
    targets = torch.tensor(targets)
    target_lengths = torch.tensor(target_lengths)
    return images, targets, target_lengths


def build_dataloader(
    data_configs: Union[List, str],
    num_workers: int = 0,
    batch_size: int = 1,
    shuffle: bool = True,
    **data_options,
):
    # Load dataset(s)
    if isinstance(data_configs, str):
        datasets = load_data(data_configs, **data_options)
    else:
        datasets = [load_data(config, **data_options) for config in data_configs]
        datasets = ConcatDataset(datasets)

    # Sample data with similar widths
    # if shuffle:
    #     sampler = SameSizeSampler(shuffle=shuffle)
    # else:
    #     sampler = SameSizeSampler(shuffle=shuffle)

    # Build dataloader
    loader = DataLoader(
        datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_variable_width,
        shuffle=shuffle,
    )
    return loader
