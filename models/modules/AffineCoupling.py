# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
import torch.nn as nn

from models.modules import thops
from models.modules.layers import Conv2d, Conv2dZeros


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.NN = nn.Sequential(
            Conv2d(in_channels, hidden_channels),
            nn.ReLU(inplace=False),
            Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(inplace=False),
            Conv2dZeros(hidden_channels, out_channels)
        )

    def forward(self, inp, logdet=None, reverse=False):
        a, b = thops.split_feature(inp, "split")
        h = self.NN(a)
        shift, scale = thops.split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.)

        if not reverse:     # Normal flow
            b = b + shift
            b = b * scale
            d_logdet = thops.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            b = b / scale
            b = b - shift
            d_logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3])

        logdet = logdet + d_logdet
        z = thops.cat_feature(a, b)

        return z, logdet
