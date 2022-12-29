from torchvision import models
from torch import nn
from ...tool import patcher


def patched_inception():
    net = models.inception_v3()

    def patch(m):
        stride = m.stride if isinstance(
            m.stride, tuple) else (m.stride, m.stride)
        return patcher.change_layer_param(m, stride=(stride[0], 1))

    def cond(m):
        if not isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            return False

        stride = m.stride if isinstance(
            m.stride, tuple) else (m.stride, m.stride)
        return all(s > 1 for s in stride)

    net_ = nn.Sequential()

    for name, child in net.named_children():
        if "Mixed_6" in name:
            break

        net_.add_module(name, child)

    net = patcher.patch_net(net_, patch, cond)
    return net_


class PatchedInception(nn.Module):
    def __init__(self, output_size, dropout):
        super().__init__()

        self.inception = patched_inception()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(288, output_size, 1, bias=False)

    def forward(self, x):
        x = self.inception(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = x.flatten(2)
        x = x.permute([2, 0, 1])
        return x
