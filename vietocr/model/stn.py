from torch import nn
from torch.nn import functional as F
from torchvision import models


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = models.mobilenet_v3_small(num_classes=6)

    # Spatial transformer network forward function
    def forward(self, x):
        theta = self.localization(x).reshape(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
