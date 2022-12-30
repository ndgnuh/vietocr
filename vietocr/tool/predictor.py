from .translate import build_model, process_image
from .utils import download_weights

import torch


def load_weights(path):
    if path.startswith('http'):
        weights = torch.load(download_weights(path))
    else:
        weights = torch.load(path)
    return weights


class Predictor:
    def __init__(self, config):
        config.setdefault('device', 'cpu')
        self.model, self.vocab = build_model(config)

        weights = load_weights(config['weights'])
        self.model.load_state_dict(weights)

        self.model.eval()
        self.image_height = config['image_height']
        self.image_min_width = config['image_min_width']
        self.image_max_width = config['image_max_width']

    def __call__(self, image):
        image = process_image(image,
                              self.image_height,
                              self.image_min_width,
                              self.image_max_width)
        output = self.model(torch.tensor(image.astype('float32')).unsqueeze(0))
        return self.vocab.batch_decode(output.argmax(dim=-1).tolist())
