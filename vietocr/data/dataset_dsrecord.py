from dsrecords import IndexedRecordDataset, io
from torch.utils.data import Dataset


class OCRRecordDataset(Dataset):
    def __init__(self, data_path, transform):
        loaders = [io.load_cv2, io.load_str]
        self.data_path = data_path
        self.data = IndexedRecordDataset(data_path, deserializers=loaders)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            return self.transform(image, label)
        else:
            return image, label
