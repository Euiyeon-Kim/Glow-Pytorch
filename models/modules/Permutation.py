# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import thops


class Invertible1x1Conv(nn.Module):
    """
        Generalization of a permutation operation
        W = PL(U + diag(s))
        1. Initialize W with random rotation matrix
        2. Compute correspondin P, L, U, diag(s)
        3. Optimize L, U, diag(s)
    """
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)

        self.w_shape = w_shape
        self.LU = LU_decomposed

        if not LU_decomposed:   # Without LU decomposition
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:                   # To reduce cost of computing det(W)
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            # Parameters in register_buffer are not optimized by optimizer
            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)

    def get_weight(self, inp, reverse=False):
        w_shape = self.w_shape
        if not self.LU:                                         # Without LU decomposition
            pixels = thops.pixels(inp)
            dlogdet = torch.slogdet(self.weight)[1] * pixels    # Calculate sign and log abs of det
            if not reverse:                                     # Normal flow
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:                                               # Reverse flow
                weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:                                                   # Use LU decomposition
            self.p = self.p.to(inp.device)
            self.sign_s = self.sign_s.to(inp.device)
            self.l_mask = self.l_mask.to(inp.device)
            self.eye = self.eye.to(inp.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            d_logdet = thops.sum(self.log_s) * thops.pixels(inp)
            if not reverse:                                     # Normal flow
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:                                               # Reverse flow
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))

            return w.view(w_shape[0], w_shape[1], 1, 1), d_logdet

    def forward(self, inp, logdet=None, reverse=False):
        """
            log-det = log|abs(|W|)| * pixels
        """
        weight, d_logdet = self.get_weight(inp, reverse)
        if not reverse:                                         # Normal flow
            z = F.conv2d(inp, weight)
            if logdet is not None:
                logdet = logdet + d_logdet
            return z, logdet
        else:                                                   # Reverse flow
            z = F.conv2d(inp, weight)
            if logdet is not None:
                logdet = logdet - d_logdet
            return z, logdet
