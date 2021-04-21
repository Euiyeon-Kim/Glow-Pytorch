import numpy as np
import torch
import torch.nn as nn

import utils
from models.FlowNet import FlowNet
from models.modules import thops
from models.modules.layers import GaussianDiag, Conv2dZeros, LinearZeros


class CDGlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow = FlowNet(img_shape=config.img_shape,
                            hidden_channels=config.hidden_channels,
                            K=config.K, L=config.L,
                            actnorm_scale=config.actnorm_scale,
                            LU_decomposed=config.LU_decomposed)
        self.y_classes = config.y_classes


if __name__ == '__main__':
    pass