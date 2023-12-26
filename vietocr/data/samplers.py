import random
from collections import defaultdict
from typing import List

import toolz
from torch.utils.data import Sampler
from tqdm import tqdm


class SameSizeSampler(Sampler):
    """Data sample where images have the same size are sampled together.

    Args:
        widths (list): An iterable of image widths.
        heights (list): An iterable of image heights.
        batch_size (int): batch size, will be used to shuffle
        shuffle (bool): if true the data is shuffled, within the batch, default False.
    """

    def __init__(
        self,
        widths: List[float],
        heights: List[float],
        batch_size: int,
        shuffle: bool = False,
    ):
        """See the class docstring for init function"""
        super().__init__()
        # +---------------+
        # | Store options |
        # +---------------+
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.buckets = defaultdict(list)

        # +----------------------+
        # | Build bucket indices |
        # +----------------------+
        widths = list(widths)
        heights = list(heights)
        total_length = len(widths)
        pbar = enumerate(zip(widths, heights))
        pbar = tqdm(pbar, "Building buckets", total=total_length, leave=False)
        for i, (w, h) in pbar:
            key = int(round(w / h * 32))
            key = key - key % 4
            self.buckets[key].append(i)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        batch_size = self.batch_size
        buckets = self.buckets

        # +-----------------+
        # | Collect indices |
        # +-----------------+
        indices = []
        for key in sorted(list(buckets.keys())):
            indices.extend(buckets[key])

        # +---------------------------------------------------+
        # | If shuffling is needed,                           |
        # | perform batched shuffling to not mess up the size |
        # | if not, use incrementing data size                |
        # +---------------------------------------------------+
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
