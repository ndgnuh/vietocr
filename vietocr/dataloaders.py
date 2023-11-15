import hashlib
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from os import makedirs, path
from typing import Callable, Hashable, List, Optional, Tuple

import numpy as np
import toolz
import torch
from dsrecords import IndexedRecordDataset, io
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from tqdm import tqdm

from .tools import create_letterbox_pil, pil_to_numpy


class disk_cache:
    cache_dir = ".cache"

    @classmethod
    def save_cache(cls, key, data):
        cls.ensure_cache_dir()
        cache_path = cls.cache_path(key)
        with open(cache_path, "wb+") as f:
            pickle.dump(data, f)

    @classmethod
    def ensure_cache_dir(cls):
        if not path.exists(cls.cache_dir):
            makedirs(cls.cache_dir, exist_ok=True)

    @classmethod
    def cache_path(cls, key: str):
        return path.join(cls.cache_dir, f"{key}")

    @classmethod
    def load_cache(cls, key):
        cache_path = cls.cache_path(key)
        if not path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data


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
        self.annotation_path = annotation_path

    def hash(self) -> str:
        with open(self.annotation_path, "rb") as f:
            hash_ = hashlib.file_digest(f, "sha256").hexdigest()
        return hash_

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int) -> Sample:
        image_path, label = self.samples[i]
        image = Image.open(image_path)
        sample = Sample(image=image, target=label)
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def get_bucket_key(self, sample: Sample) -> Hashable:
        return sample.image.width


class BucketRandomSampler(Sampler):
    def __init__(self, data_source: Dataset, batch_size: int, shuffle: bool = False):
        # Validate
        msg = "Please implement `get_bucket_key` function in your dataset. \
        The function should take a sample and return a hashable-key."
        assert hasattr(data_source, "get_bucket_key"), msg

        # Init
        super().__init__(data_source=data_source)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.cache_key = f"sampler_bucket_{data_source.hash()}"
        self.data_source = data_source
        buckets = disk_cache.load_cache(self.cache_key)
        if buckets is None:
            self.buckets = self.build_bucket_indices()
            disk_cache.save_cache(self.cache_key, self.buckets)
        else:
            self.buckets = buckets
        print("Buckets by widths", sorted(list(self.buckets.keys())))

    def build_bucket_indices(self):
        # IO bound, threads won't help
        data_source = self.data_source
        buckets = defaultdict(list)
        n = len(data_source)
        for i in tqdm(range(n), "Building bucket", leave=False):
            sample = data_source[i]
            key = data_source.get_bucket_key(sample)
            buckets[key].append(i)
        return buckets

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        batch_size = self.batch_size
        buckets = self.buckets

        # Collect indices
        indices = []
        for key in sorted(list(buckets.keys())):
            indices.extend(buckets[key])

        if self.shuffle:  # Shuffling by batch
            partitions = []
            for part in toolz.partition(batch_size, indices):
                part = list(part)
                random.shuffle(part)  # Shuffle inside each partition
                partitions.append(part)
            random.shuffle(partitions)  # Shuffle all the partitions
            indices_iter = toolz.concat(partitions)  # this is already an iterator
        else:  # No shuffling
            indices_iter = iter(indices)

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
        image = create_letterbox_pil(image, max_width, max_height)
        image = pil_to_numpy(image)
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


class OCRRecordDataset(OCRDataset):
    def __init__(self, data_path, transform):
        loaders = [io.load_pil, io.load_str]
        self.data_path = data_path
        self.data = IndexedRecordDataset(data_path, deserializers=loaders)
        self.transform = transform

    def hash(self):
        with open(self.data.index.path, "rb") as f:
            hash_ = hashlib.file_digest(f, "sha256").hexdigest()
        return hash_

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        image, label = self.data[idx]
        sample = Sample(image, label)
        if self.transform:
            return self.transform(sample)
        else:
            return sample


def get_dataset(dataset_path: str, transform: Callable):
    if dataset_path.endswith(".rec"):
        return OCRRecordDataset(dataset_path, transform=transform)
    else:
        return OCRDataset(dataset_path, transform)


def get_dataloader(
    annotation_path: str,
    transform: Optional[Callable] = None,
    **dataloader_kwargs,
):
    # Create dataset or datasets
    if isinstance(annotation_path, str):
        dataset = get_dataset(annotation_path, transform)
    else:
        datasets = [get_dataset(ann_file, transform) for ann_file in annotation_path]
        dataset = ConcatDataset(datasets)
        dataset.get_bucket_key = datasets[0].get_bucket_key

    # Create loader
    try:
        shuffle = dataloader_kwargs.pop("shuffle")
    except KeyError:
        shuffle = False
    batch_size = dataloader_kwargs.get("batch_size", 1)
    dataloader_kwargs["sampler"] = BucketRandomSampler(dataset, batch_size, shuffle=shuffle)
    dataloader_kwargs["collate_fn"] = collate_variable_width
    loader = DataLoader(dataset, **dataloader_kwargs)
    return loader
