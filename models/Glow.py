import numpy as np
import torch
import torch.nn as nn

import utils
from models.FlowNet import FlowNet
from models.modules import thops
from models.modules.layers import GaussianDiag, Conv2dZeros, LinearZeros


class Glow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow = FlowNet(img_shape=config.img_shape,
                            hidden_channels=config.hidden_channels,
                            K=config.K, L=config.L,
                            actnorm_scale=config.actnorm_scale,
                            LU_decomposed=config.LU_decomposed)
        self.y_classes = config.y_classes

        if config.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = Conv2dZeros(C * 2, C * 2)
        if config.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(config.y_classes, 2 * C)
            self.project_class = LinearZeros(C, config.y_classes)

        num_device = len(utils.get_proper_device(config.device, False))
        assert config.batch_size % num_device == 0
        self.register_parameter("prior_h", nn.Parameter(torch.zeros([config.batch_size // num_device,
                                                                     self.flow.output_shapes[-1][1] * 2,
                                                                     self.flow.output_shapes[-1][2],
                                                                     self.flow.output_shapes[-1][3]])))

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0

        if self.config.learn_top:
            h = self.learn_top(h)
        if self.config.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp

        return thops.split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, y_onehot)
        else:
            return self.reverse_flow(z, y_onehot, eps_std)

    def normal_flow(self, x, y_onehot):
        pixels = thops.pixels(x)
        z = x + torch.normal(mean=torch.zeros_like(x),
                             std=torch.ones_like(x) * (1. / 256.))
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        logdet += float(-np.log(256.) * pixels)

        # encode
        z, objective = self.flow(z, logdet=logdet, reverse=False)

        # prior
        mean, logs = self.prior(y_onehot)
        objective += GaussianDiag.logp(mean, logs, z)

        if self.config.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # return
        nll = (-objective) / float(np.log(2.) * pixels)
        return z, nll, y_logits

    def reverse_flow(self, z, y_onehot, eps_std):
        with torch.no_grad():
            mean, logs = self.prior(y_onehot)
            if z is None:   # Sample from Gaussian distribution
                z = GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited

    def generate_z(self, img):
        self.eval()
        B = self.config.batch_size
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z, _, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    @staticmethod
    def loss_generative(nll):
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())


if __name__ == '__main__':
    from config import Config
    glow = Glow(Config)
    print(glow)