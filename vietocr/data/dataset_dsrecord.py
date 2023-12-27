from abc import abstractmethod
from os import path
from typing import Callable, Optional

import cv2
from dsrecords import IndexedRecordDataset, io
from torch.utils.data import Dataset


class _OcrDataset(Dataset):
    def __iter__(self):
        """Iterate through the dataset."""
        return (self[i] for i in range(len(self)))

    def __len__(self):
        """Dataset length."""
        return len(self.samples)

    def __init__(
        self,
        samples,
        preprocess: Optional[Callable] = None,
        encode: Optional[Callable] = None,
        augment: Optional[Callable] = None,
    ):
        self.samples = samples
        self.preprocess = preprocess
        self.encode = encode
        self.augment = augment

    @abstractmethod
    def get_raw(self, i):
        ...

    def __getitem__(self, i):
        image, label = self.get_raw(i)
        image = _call_me_maybe(self.augment, image, image=image)
        image = _call_me_maybe(self.preprocess, image, image)
        label = _call_me_maybe(self.encode, label, label)
        return image, label


def _call_me_maybe(f: Optional[Callable], default, *args, **kwargs):
    if f is None:
        return default
    else:
        return f(*args, **kwargs)


class DsrecordOcrDataset(_OcrDataset):
    """Ocr dataset that loads from dsrecords format."""

    def __init__(self, data_path, **kwargs):
        """Initialize Ocr dataset.

        Args:
            data_path: Path to data.rec record file.
            transform: A function that receives (image, label) to transform data.
        """
        loaders = [io.load_cv2, io.load_str]
        samples = IndexedRecordDataset(data_path, deserializers=loaders)
        super().__init__(samples, **kwargs)

    def get_raw(self, i: int):
        """Get raw sample from dataset."""
        image, label = self.samples[i]
        image = image[..., ::-1]
        return image, label


class VietocrOcrDataset(_OcrDataset):
    """Ocr dataset that loads from dsrecords format."""

    def __init__(self, data_path: str, **kwargs):
        # Collect lines
        with open(data_path) as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [line for line in lines if len(line) > 0]

        # Collect samples
        samples = []
        for line in lines:
            image_path, label = line.split("\t")
            samples.append((image_path, label))

        # Store root path
        self.root_path = path.dirname(data_path)

        # Super initialization
        super().__init__(samples, **kwargs)

    def get_raw(self, i: int):
        """Get raw sample from dataset."""
        image_path, label = self.samples[i]
        image_path = path.join(self.root_path, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image[..., ::-1]
        return image, label
