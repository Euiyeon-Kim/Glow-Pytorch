# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
import torch.nn as nn

from models.modules import thops


class ActNorm2d(nn.Module):
    """
        Activation Normalization Layer (Input: BCHW shape)
        1. Data dependant initialization
        2. Affine transformation using scale and bias parameters per channel
    """

    def __init__(self, num_channels, scale=1.0):
        super().__init__()
        self.initiated = False
        self.scale = float(scale)
        size = [1, num_channels, 1, 1]
        self.num_channels = num_channels

        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))

    def _check_input_dim(self, inp):
        assert len(inp.size()) == 4
        assert inp.size(1) == self.num_channels, "[ActNorm]: input should be in shape as `BCHW` " \
                                                 f"channels should be {self.num_channels} " \
                                                 f"rather than {inp.size(1)}"

    def initialize_parameters(self, inp):
        self._check_input_dim(inp)
        if not self.training:
            return
        assert inp.device == self.bias.device

        with torch.no_grad():
            bias = thops.mean(inp.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((inp.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale/(torch.sqrt(vars)+1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.initiated = True

    def _center(self, inp, reverse=False):
        if not reverse:  # Normal flow
            return inp + self.bias
        else:            # Reverse flow
            return inp - self.bias

    def _scale(self, inp, logdet, reverse=False):
        logs = self.logs
        if not reverse:  # Normal flow
            inp = inp * torch.exp(logs)
        else:            # Reverse flow
            inp = inp * torch.exp(-logs)

        if logdet is not None:
            d_logdet = thops.pixels(inp) * thops.sum(logs)
            if reverse:  # Reverse flow
                d_logdet *= -1
            logdet = logdet + d_logdet

        return inp, logdet

    def forward(self, inp, logdet=None, reverse=False):
        if not self.initiated:
            self.initialize_parameters(inp)
        self._check_input_dim(inp)

        # Channel wise normalization
        if not reverse:    # Normal flow
            inp = self._center(inp, reverse)
            inp, logdet = self._scale(inp, logdet, reverse)
        else:              # Reverse flow
            inp, logdet = self._scale(inp, logdet, reverse)
            inp = self._center(inp, reverse)

        return inp, logdet


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actnorm = ActNorm2d(64).to(device)
    rand_inp = torch.rand(16, 64, 28, 28).to(device)
    tmp_inp, tmp_log = actnorm(rand_inp, reverse=True)

