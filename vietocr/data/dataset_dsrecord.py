import random
from abc import abstractmethod
from functools import lru_cache
from os import listdir, path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from dsrecords import IndexedRecordDataset, io
from PIL import Image, ImageDraw, ImageFont
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
        vocab: Optional = None,
        preprocess: Optional[Callable] = None,
        augment: Optional[Callable] = None,
    ):
        self.samples = samples
        self.preprocess = preprocess
        self.vocab = vocab
        self.augment = augment

    @abstractmethod
    def get_raw(self, i):
        ...

    @property
    def encode(self):
        return None if self.vocab is None else self.vocab.encode

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


def render_text_box(
    text: str,
    font: ImageFont.ImageFont,
    bg: Tuple[int, int, int] = (255, 255, 255),
    fg: Tuple[int, int, int] = (0, 0, 0),
):
    """Create an image of the text box, with some pre-defined font.

    Args:
        text (str): The text to render.
        font (ImageFont): A Pillow's ImageFont, used to render the text.
        bg (Tuple): A tuple of integers for (R, G, B) background color.
        fg (Tuple): A tuple of integers for (R, G, B) text color.

    Returns:
        A Pillow Image with rendered text.
    """
    x1, y1, x2, y2 = font.getbbox(text)
    image = Image.new("RGB", (x1 + x2, y1 + y2), bg)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=text, fill=fg, font=font)
    return image


class PretrainOcrDataset(_OcrDataset):
    """Dataset for pretraining OCR models."""

    def __init__(
        self,
        repetition: int = 32,
        fonts: Optional[Union[str, List[str]]] = None,
        n: int = 33_333,
        **kwargs,
    ):
        vocab = kwargs["vocab"]
        self.characters = [c for c in vocab.chars if not c.isspace()]
        self.repetition = repetition
        self.num_characters = len(self.characters)
        self.fonts = fonts
        samples = [False] * n
        super().__init__(samples, **kwargs)
        self.get_raw = lru_cache(maxsize=n)(self.get_raw)

    def _sample_fonts(self) -> ImageFont.ImageFont:
        fonts = self.fonts
        font_size = 64
        if fonts is None:
            return ImageFont.load_default(size=font_size)
        elif isinstance(fonts, (list, tuple)):
            return ImageFont.truetype(random.choice(fonts), size=font_size)
        elif isinstance(fonts, str):
            if path.isfile(fonts):
                return ImageFont.truetype(fonts, size=font_size)
            else:
                fonts = [path.join(fonts, font) for font in listdir(fonts)]
                return ImageFont.truetype(random.choice(fonts), size=font_size)

    def get_raw(self, i: int):
        """Get dataset item."""
        i = i % self.num_characters
        char = self.characters[i]
        font = self._sample_fonts()

        # Create target
        rep = random.randint(1, self.repetition)
        text = char * rep

        # Render text box
        image = render_text_box(text, font)
        image = np.array(image)
        return image, text
