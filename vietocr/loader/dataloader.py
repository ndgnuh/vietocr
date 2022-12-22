from vietocr.tool.image_folder import IndexedImageFolder
from vietocr.tool.translate import resize
from vietocr.tool.create_dataset import createDataset
from vietocr.tool.translate import process_image
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import six
import lmdb
import torch
import numpy as np
from collections import defaultdict
import os
import random
import re
from PIL import Image
from typing import Any, Callable
from dataclasses import dataclass
from torchvision.transforms import functional as TF
import sys


class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self

    def to(self, device):
        return AttrDict({k: v.to(device) for k, v in self.items()})


class ClusterRandomSampler(Sampler):

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size]
                       for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)


def letterbox(image, image_height, image_width):
    image_size = (image_width, image_height)
    image.thumbnail(image_size)
    lb_image = Image.new("RGB", image_size, (112, 112, 112))
    lb_image.paste(image, (0, 0))
    return lb_image


def ensure_tensor(image):
    if isinstance(image, torch.Tensor):
        return image
    if Image.isImageType(image):
        return TF.to_tensor(image)
    return torch.tensor(image)


def pad_sequence(seq, max_length, pad, truncate=True):
    length = len(seq)
    if truncate:
        seq = seq[:max_length]
        length = min(max_length, length)
    assert length <= max_length
    padding_length = max_length - length
    padded = seq + [pad] * padding_length
    mask = [1] * length + padding_length * [0]
    return padded, mask, length


class OCRDataset(IndexedImageFolder):
    def __init__(
        self,
        index: str,
        vocab: Any,
        transform: Callable,
        image_height: int,
        image_width: int,
        max_sequence_length: int
    ):
        super().__init__(index=index)
        self.index = index
        self.vocab = vocab
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.max_sequence_length = max_sequence_length

    def split_annotation(self, line):
        splits = re.split(r"\s+", line.strip())
        return splits[0], " ".join(splits[1:])

    def __getitem__(self, index: int):
        image_path, word = self.split_annotation(self.samples[index])
        image = Image.open(image_path).convert("RGB")
        image = letterbox(image, self.image_height, self.image_width)

        if self.transform is not None:
            image = self.transform(image)

        target = self.vocab.encode(word, max_length=self.max_sequence_length)
        target_mask = [
            1 if i >= 4 else 0 for i in target
        ]
        # max_length = 128
        # target, target_mask, target_length = pad_sequence(
        #     word,
        #     max_length - 1,
        #     self.vocab.pad
        # )
        # target = [self.vocab.go] + word + [self.vocab.eos]

        # Keep the keys to interface with the previous code
        # image, target input, target output, target mask
        tgt_input = ensure_tensor(target)
        tgt_output = torch.roll(tgt_input, -1)
        tgt_output[-1] = 1
        return AttrDict(
            image=ensure_tensor(image),
            target=tgt_input,
            target_mask=ensure_tensor(target_mask),
            target_output=tgt_output,
            # target_length=ensure_tensor(target_length)
        )


class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample['word']) for sample in batch)
        for sample in batch:
            img.append(sample['img'])
            filenames.append(sample['img_path'])
            label = sample['word']
            label_len = len(label)

            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len, dtype=np.float32))))

        img = np.array(img, dtype=np.float32)

        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1] = 0

        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (
                tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights) == 0

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask),
            'filenames': filenames
        }

        return rs
