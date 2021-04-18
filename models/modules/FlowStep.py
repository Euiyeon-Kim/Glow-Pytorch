# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch.nn as nn

from models.modules.ActNorm import ActNorm2d
from models.modules.Permutation import Invertible1x1Conv
from models.modules.AffineCoupling import AffineCoupling


class FlowStep(nn.Module):
    """
        One step for Flow based generative model
        1. Actnorm
        2. Invertible 1x1 convolution
        3. Affine coupling
    """

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0,
                 LU_decomposed=False):

        super().__init__()
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        self.permutation = Invertible1x1Conv(in_channels, LU_decomposed)
        self.coupling = AffineCoupling(in_channels // 2, in_channels, hidden_channels)

    def normal_flow(self, inp, logdet):
        assert inp.size(1) % 2 == 0     # For affine coupling layer
        z, logdet = self.actnorm(inp, logdet=logdet, reverse=False)
        z, logdet = self.permutation(inp, logdet=logdet, reverse=False)
        z, logdet = self.coupling(inp, logdet=logdet, reverse=False)
        return z, logdet

    def reverse_flow(self, inp, logdet):
        assert inp.size(1) % 2 == 0  # For affine coupling layer
        z, logdet = self.coupling(inp, logdet=logdet, reverse=True)
        z, logdet = self.permutation(inp, logdet=logdet, reverse=True)
        z, logdet = self.actnorm(inp, logdet=logdet, reverse=True)
        return z, logdet

    def forward(self, inp, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(inp, logdet)
        else:
            return self.reverse_flow(inp, logdet)


if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = FlowStep(4, 32).to(device)
    rand_log = torch.rand(16).to(device)
    rand_inp = torch.rand(16, 4, 28, 28).to(device)
    tmp_inp, tmp_log = step(rand_inp, logdet=rand_log, reverse=False)
    print(tmp_inp.shape)
    print(tmp_log.shape)
    exit()