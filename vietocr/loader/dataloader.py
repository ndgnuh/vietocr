from ..tool import utils
from .. import const
from os import path
from typing import Optional, Callable, Union, List
from vietocr.tool.translate import resize
from vietocr.tool.create_dataset import createDataset
from vietocr.tool.translate import process_image
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import six
import lmdb
import torch
import numpy as np
from collections import defaultdict
import sys
import os
import random
from io import BytesIO
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def letterbox_image(image, size):
    # resize image with unchanged aspect ratio using padding
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


class DatasetMuxer(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.current_idx = 0
        self.num_dataset = len(datasets)
        self.samples = []
        self.reverse_samples = dict()
        count = 0
        for (dataset_idx, num_data) in enumerate(self.lens):
            for data_idx in range(num_data):
                self.samples.append((dataset_idx, data_idx))
                self.reverse_samples[(dataset_idx, data_idx)] = count
                count += 1

        self.cluster_indices = defaultdict(list)
        pbar = tqdm(range(len(self)), "Building muxer cluster indices")
        for dataset_idx, dataset in enumerate(datasets):
            for bucket_idx, data_indices in dataset.cluster_indices.items():
                for data_idx in data_indices:
                    sample_idx = self.reverse_samples[(dataset_idx, data_idx)]
                    self.cluster_indices[bucket_idx].append(sample_idx)
                    pbar.update(1)

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        dataset_idx, data_idx = self.samples[idx]
        return self.datasets[dataset_idx][data_idx]


class OCRDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        vocab,
        image_height: int = 32,
        image_min_width: int = 32,
        image_max_width: int = 512,
        transform=None,
        letterbox: bool = False,
        align_width: int = 10,
    ):
        lmdb_path = get_lbdm_path(annotation_path)
        root_dir = path.dirname(annotation_path)
        annotation_path = path.basename(annotation_path)

        self.align_width = align_width
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.vocab = vocab
        self.transform = transform

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.lmdb_path = lmdb_path
        self.letterbox = letterbox

        if os.path.isdir(self.lmdb_path):
            print('{} exists. Remove folder if you want to create new dataset'.format(
                self.lmdb_path))
            sys.stdout.flush()
        else:
            createDataset(self.lmdb_path, root_dir, annotation_path)

        self.env = lmdb.open(
            self.lmdb_path,
            max_readers=8,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self.txn = self.env.begin(write=False)

        nSamples = int(self.txn.get('num-samples'.encode()))
        self.nSamples = nSamples

        self.build_cluster_indices()

    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)

        pbar = tqdm(range(self.__len__()),
                    desc='{} build cluster'.format(self.lmdb_path),
                    ncols=100, position=0, leave=True)

        error = 0
        for i in pbar:
            bucket = self.get_bucket(i)
            if bucket is None:
                error += 1
                continue
            self.cluster_indices[bucket].append(i)
        print(f"Skipped {error} oversize images")

    def get_bucket(self, idx):
        key = 'dim-%09d' % idx

        dim_img = self.txn.get(key.encode())
        dim_img = np.fromstring(dim_img, dtype=np.int32)
        imgH, imgW = dim_img

        try:
            new_w, image_height = resize(
                imgW, imgH, self.image_height, self.image_min_width, self.image_max_width, align_width=self.align_width)
        except AssertionError:
            return None

        return new_w

    def read_buffer(self, idx):
        img_file = 'image-%09d' % idx
        label_file = 'label-%09d' % idx
        path_file = 'path-%09d' % idx

        imgbuf = self.txn.get(img_file.encode())

        label = self.txn.get(label_file.encode()).decode()
        img_path = self.txn.get(path_file.encode()).decode()

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        return buf, label, img_path

    def read_data(self, idx):
        buf, label, img_path = self.read_buffer(idx)

        img = Image.open(buf).convert('RGB')

        if self.transform:
            img = self.transform(image=np.array(img))
            img = Image.fromarray(img['image'])

        if self.letterbox:
            img = letterbox_image(
                img,
                (self.image_max_width, self.image_height)
            )

        img_bw = process_image(img,
                               self.image_height,
                               self.image_min_width,
                               self.image_max_width,
                               align_width=self.align_width)
        word = self.vocab.encode(label)

        return img_bw, word, img_path

    def __getitem__(self, idx):
        img, word, img_path = self.read_data(idx)

        img_path = os.path.join(self.root_dir, img_path)

        sample = {
            'img': img,
            'word': word,
            'img_path': img_path,
            'target_length': len(word)
        }

        return sample

    def __len__(self):
        return self.nSamples


class ClusterRandomSampler(Sampler):

    def __init__(self,
                 data_source,
                 batch_size: int,
                 curriculum: bool = False,
                 shuffle: bool = True,
                 limit_batch_per_size: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.curriculum = curriculum
        self.limit_batch_per_size = limit_batch_per_size

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def _shuffle(self, x):
        random.shuffle(x)
        return x

    def to_list(self):
        batch_lists = []
        data = self.data_source.cluster_indices
        skipped = 0

        # Keys are image width
        keys = data.keys()
        if self.curriculum:
            keys = sorted(keys)
        else:
            keys = list(keys)
            random.shuffle(keys)

        for cluster in keys:
            cluster_indices = data[cluster]
            random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size]
                       for i in range(0, len(cluster_indices), self.batch_size)]
            for batch in batches:
                if len(batch) != self.batch_size:
                    skipped += len(batch)

            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.limit_batch_per_size is not None:
                random.shuffle(batches)
                batches = batches[:self.limit_batch_per_size]

            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)

        if skipped > 0:
            print(f"Skipped {skipped} data points for not fitting anywhere")

        # Don't shuffle if curriculum learning
        if self.shuffle and not self.curriculum:
            random.shuffle(lst)

        lst = self.flatten_list(lst)
        return lst

    def __iter__(self):
        # This was separated to do something like len(self.to_list())
        # But the bug limit feature was fixed and there's no need for this anymore
        return iter(self.to_list())

    def __len__(self):
        return len(self.data_source)


class Collator(object):
    def __init__(
        self,
        shift_target=False,
        pad_index=0
    ):
        self.shift_target = shift_target
        self.pad = [pad_index]

    def __call__(self, batch):
        images = []
        targets = []
        target_lengths = [sample['target_length'] for sample in batch]
        max_length = max(target_lengths)
        for sample in batch:
            images.append(sample['img'])
            target = sample['word']
            target_length = sample['target_length']
            target.extend(self.pad * (max_length - target_length))
            targets.append(target)

        # Shift outputs for S2S
        targets = np.array(targets)
        if self.shift_target:
            targets = np.roll(targets.T, -1, 0).T
            targets[:, -1] = 0

        images = torch.FloatTensor(np.array(images))
        targets = torch.LongTensor(targets)
        target_lengths = torch.LongTensor(np.array(target_lengths))
        return images, targets, target_lengths


def get_lbdm_path(annotation_path: str):
    uuid = utils.annotation_uuid(annotation_path)
    # Getting the basename
    annotation_path = path.normpath(annotation_path)
    basename = path.splitext(annotation_path)[0].replace("/", "-")
    lmdb_name = f"{basename}-{uuid}"
    return path.join(const.lmdb_dir, lmdb_name)


def get_dataset(annotation_paths: Union[str, List[str]], **k):
    if isinstance(annotation_paths, str):
        return OCRDataset(annotation_paths, **k)
    else:
        datasets = [
            OCRDataset(annotation_path, **k)
            for annotation_path in annotation_paths
        ]
        return DatasetMuxer(datasets)


def build_dataloader(
    annotation_path: str,
    image_height: int,
    image_min_width: int,
    image_max_width: int,
    vocab,
    transform: Optional[Callable] = None,
    shuffle: Optional[bool] = False,
    batch_size: Optional[int] = 1,
    num_workers: Optional[int] = None,
    curriculum: Optional[bool] = True,
    letterbox: Optional[bool] = False,
    shift_target: Optional[bool] = False,
    align_width: Optional[int] = 10,
    limit_batch_per_size: Optional[int] = None,
):
    dataset = get_dataset(annotation_paths=annotation_path,
                          vocab=vocab,
                          transform=transform,
                          image_height=image_height,
                          image_min_width=image_min_width,
                          image_max_width=image_max_width,
                          letterbox=letterbox,
                          align_width=align_width)

    sampler = ClusterRandomSampler(
        dataset,
        batch_size,
        shuffle=shuffle,
        curriculum=curriculum,
        limit_batch_per_size=limit_batch_per_size,
    )
    collate_fn = Collator(shift_target=shift_target)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
