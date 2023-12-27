"""
This module load the data from configurations. The data path can be a string, a dictionary, or a list of the aboves.

If the data path is a string:
- the path endswith `.rec` -> Load OcrRecordDataset
- the path endswith `.txt` -> Load OcrVietocrDataset
If data path is a dict:
- use the 'type' key to determine the dataset, and use other keywords as arguments
"""
from torch.utils.data import DataLoader, ConcatDataset
from typing import Callable, Optional, List
from .dataset_dsrecord import OcrRecordDataset
from .samplers import SameSizeSampler


def load_data(data_path: str, transform: Optional[Callable] = None):
    """
    If the data path is a string:
    - the path endswith `.rec` -> Load OcrRecordDataset
    - the path endswith `.txt` -> Load OcrVietocrDataset
    If data path is a dict:
    - use the 'type' key to determine the dataset, and use other keywords as arguments
    """
    if isinstance(data_path, str):
        if data_path.endswith(".rec"):
            return OcrRecordDataset(data_path, transform=transform)
        else:
            raise ValueError("Dataset not implemented")
    if data_path == "pretrain":
        return
    elif data_path.endswith(".rec"):
        return OcrRecordDataset(data_path, transform)
    else:
        raise ValueError(f"Unsupported dataset {data_path}")


def create_ocr_dataset(data_paths: List[str], transform: Optional[Callable] = None):
    datasets = [load_data(data_path, transform) for data_path in data_paths]

    return datasets


def create_ocr_dataloaders(data_paths: List[str], transform: Optional[Callable] = None):
    datasets = [load_data(data_path, transform) for data_path in data_paths]

    return datasets
