from .translate import build_model, process_image
from .utils import download_weights

import torch


class Predictor:
    def __init__(self, config):
        config.setdefault('device', 'cpu')
        self.model, self.vocab = build_model(config)
        self.model.eval()
        self.image_height = config['image_height']
        self.image_min_width = config['image_min_width']
        self.image_max_width = config['image_max_width']

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.model.parameters()).device
        if not isinstance(images, (tuple, list)):
            images = [images]
        images = [process_image(image,
                                self.image_height,
                                self.image_min_width,
                                self.image_max_width)
                  for image in images]
        images = [torch.tensor(image.astype('float32')) for image in images]
        images = torch.stack(images, dim=0)
        images = images.to(device)
        output = self.model(images)
        output = output.cpu().detach()
        return self.vocab.batch_decode(output.argmax(dim=-1).tolist())
