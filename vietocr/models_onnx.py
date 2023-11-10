from typing import List, Tuple

import numpy as np
from onnxruntime import InferenceSession, get_available_providers
from PIL import Image

from .tools import pil_to_numpy, resize_image
from .vocabs import Vocab


class ModelONNX:
    def __init__(
        self,
        onnx_path: int,
        vocab: Vocab,
        onnx_providers: List = get_available_providers(),
    ):
        self.session = sess = InferenceSession(onnx_path, providers=onnx_providers)
        self.input_name = sess.get_inputs()[0].name
        self.image_height = sess.get_inputs()[0].shape[2]
        self.min_width = self.image_height // 2
        self.max_width = 20000

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = resize_image(image, self.image_height, self.min_width, self.max_width)
        np_image = pil_to_numpy(image)
        return np_image

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        np_image = self.preprocess(image)[None]
        scores, indices = self.session.run(None, {self.input_name: np_image})
        result = vocab.decode(indices)
        score = sum(scores) / len(scores)
        return result, score
