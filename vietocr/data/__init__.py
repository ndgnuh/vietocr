from torch.utils.data import DataLoader, Dataset


def parse_data_path(dpath: str):
    """Parse data path to data according to specification.

    The data path can be one of the following format
    - /path/to/dsrecord/file.rec: The dsrecord data format
    - /path/to/index/file.txt: The original vietocr data format
    - :pretrain: [repetition=32, fonts=]

    Unless the data path is :pretrain:

    Args:
        dpath (str): Data path specs
    """
    pass

class OcrDataset(Dataset):
    def __init__(self, data_config: str):
        super().__init__()
        self.data = load_data(data_path)
