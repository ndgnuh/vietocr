from .translate import build_model, process_image
import numpy as np

import torch


def score_agg(x, good_threshold):
    a = x.min(axis=-1)
    b = x.max(axis=-1)
    r = b - a
    x = (x - a) / r
    x = 1 / (1 + np.exp(30 * (good_threshold - x)))
    x = x * r + a
    # score = np.exp((np.exp(x)) / np.exp(1)) / np.exp(1)
    score = x.mean(axis=-1)
    return score


class Predictor:
    def __init__(self, config, reset_cache=False):
        config.setdefault('device', 'cpu')
        self.model, self.vocab = build_model(config, reset_cache=reset_cache)
        self.model.eval()
        self.image_height = config['image_height']
        self.image_min_width = config['image_min_width']
        self.image_max_width = config['image_max_width']

    @ torch.no_grad()
    def __call__(
        self,
        images,
        align_width: int = 10,
        prob_threshold: float = 0.75,
        returns_map: bool = False
    ):
        device = next(self.model.parameters()).device
        if not isinstance(images, (tuple, list)):
            images = [images]
        images = [process_image(image,
                                self.image_height,
                                self.image_min_width,
                                self.image_max_width,
                                align_width=align_width)
                  for image in images]

        images = [torch.tensor(image.astype('float32')) for image in images]
        images = torch.stack(images, dim=0)
        images = images.to(device)
        output = self.model(images)
        output = output.cpu().detach()
        probs, indices = torch.softmax(output, dim=-1).max(dim=-1)
        probs = probs.numpy()
        indices = indices.numpy()
        results = self.vocab.batch_decode(indices.tolist())
        scores = score_agg(probs, 0.8).tolist()
        if returns_map:
            return results, scores, output
        else:
            return results, scores
