from torch import nn
from torch.nn import functional as F
from torchvision import models


class VSTN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.grid_gen = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, 2, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        lc = self.localization(x)
        # b 2 h w -> b h w 2
        grid = self.grid_gen(lc).permute((0, 2, 3, 1))
        x = F.grid_sample(x, grid)
        return x


class AffineSpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = models.mobilenet_v3_small(num_classes=6)

    # Spatial transformer network forward function
    def forward(self, x):
        theta = self.localization(x).reshape(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


def SpatialTransformer(stn):
    if stn is None:
        return nn.Identity()
    elif stn == 'affine':
        return AffineSpatialTransformer()
    elif stn == 'vstn':
        return VSTN(3)
    else:
        raise ValueError(
            f"Unsupported stn {stn}, use one of: None, 'affine', 'vstn'")
