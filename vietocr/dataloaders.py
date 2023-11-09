import random
from collections import defaultdict
from dataclasses import dataclass
from os import path
from typing import Callable, Hashable, List, Optional, Tuple

import numpy as np
import toolz
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from tqdm import tqdm

from .tools import create_letterbox_pil, pil_to_numpy


@dataclass
class Sample:
    """Row-based data structure for data sample"""

    image: Image.Image
    target: List[int]

    @property
    def target_length(self) -> int:
        return len(self.target)

    def __iter__(self):
        # This is for unpacking
        return iter((self.image, self.target, self.target_length))


def split_annotation(line: str, delim: str = "\t") -> Tuple[str, str]:
    idx = line.index(delim)
    image_file = line[:idx]
    text = line[idx:]
    return image_file, text


class OCRDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        transform: Optional[Callable] = None,
        delim: str = "\t",
    ):
        with open(annotation_path) as f:
            annotations = [split_annotation(line, delim) for line in f.readlines()]
        self.root_path = path.dirname(annotation_path)
        self.samples = [
            [path.join(self.root_path, image), label] for image, label in annotations
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int) -> Sample:
        image_path, label = self.samples[i]
        image = Image.open(image_path)
        sample = Sample(image=image, target=list(label))
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def get_bucket_key(self, sample: Sample) -> Hashable:
        return sample.image.width


class BucketRandomSampler(Sampler):
    def __init__(self, data_source: Dataset, batch_size: int):
        # Validate
        msg = "Please implement `get_bucket_key` function in your dataset. \
        The function should take a sample and return a hashable-key."
        assert hasattr(data_source, "get_bucket_key"), msg

        # Init
        super().__init__(data_source=data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        self.buckets = defaultdict(list)

        iterator = enumerate(tqdm(data_source, "Building buckets", leave=False))
        for i, sample in iterator:
            key = data_source.get_bucket_key(sample)
            self.buckets[key].append(i)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        batch_size = self.batch_size
        buckets = self.buckets

        # Collect indices
        indices = []
        for b_indices in buckets.values():
            indices.extend(b_indices)

        # Shuffle by partitions
        partitions = []
        for part in toolz.partition(batch_size, indices):
            part = list(part)
            random.shuffle(part)  # Shuffle inside each partition
            partitions.append(part)
        random.shuffle(partitions)  # Shuffle all the partitions
        indices_iter = toolz.concat(partitions)  # this is already an iterator
        return indices_iter


def collate_variable_width(samples: List[Sample], pad_token_id: int = 0):
    pad_id = pad_token_id

    # Find out max metrics
    max_height = -1
    max_width = -1
    max_target_length = -1
    for image, target, target_length in samples:
        # Max image width
        max_width = max(image.width, max_width)
        max_height = max(image.height, max_height)

        # Max target length
        max_target_length = max(target_length, max_width)

    # Padding + convert to columnars
    images = []
    targets = []
    target_lengths = []
    for image, target, target_length in samples:
        # Process image
        image = tools.create_letterbox_pil(image, max_width, max_height)
        image = tools.pil_to_numpy(image)
        images.append(image)

        # Process target
        pad_length = max_target_length - target_length
        target = target + [pad_id] * pad_length
        targets.append(target)
        target_lengths.append(target_length)

    # To tensor
    images = torch.tensor(np.stack(images, axis=0))
    targets = torch.tensor(targets)
    target_lengths = torch.tensor(target_lengths)
    return images, targets, target_lengths


def get_dataloader(
    annotation_path: str,
    transform: Optional[Callable] = None,
    **dataloader_kwargs,
):
    # Create dataset or datasets
    if isinstance(annotation_path, str):
        dataset = OCRDataset(annotation_path, transform)
    else:
        datasets = [OCRDataset(ann_file, transform) for ann_file in annotation_path]
        dataset = ConcatDataset(datasets)
        dataset.get_bucket_key = datasets[0].get_bucket_key

    # Create loader
    batch_size = dataloader_kwargs.get("batch_size", 1)
    dataloader_kwargs["sampler"] = BucketRandomSampler(dataset, batch_size)
    dataloader_kwargs["collate_fn"] = collate_variable_width
    loader = DataLoader(dataset, **dataloader_kwargs)
    return loader
