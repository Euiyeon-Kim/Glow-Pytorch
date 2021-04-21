import torch.nn as nn

from models.modules.layers import SqueezeLayer, Split2d
from models.modules.FlowStep import FlowStep


class FlowNet(nn.Module):
    """
        Squeeze -> K steps -> Split
    """
    def __init__(self, img_shape, hidden_channels, K, L, actnorm_scale=1.0, LU_decomposed=False):
        super().__init__()
        self.K = K
        self.L = L
        self.layers = nn.ModuleList()
        self.output_shapes = []

        H, W, C = img_shape
        assert C == 1 or C == 3         # Image channel should be 1 or 3
        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(FlowStep(in_channels=C,
                                            hidden_channels=hidden_channels,
                                            actnorm_scale=actnorm_scale,
                                            LU_decomposed=LU_decomposed))
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def encode(self, x, logdet=0.0):
        z = x
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, eps_std=None):
        x = z
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                x, logdet = layer(x, logdet=0, reverse=True, eps_std=eps_std)
            else:
                x, logdet = layer(x, logdet=0, reverse=True)
        return x

    def forward(self, inp, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            return self.encode(inp, logdet)
        else:
            return self.decode(inp, eps_std)


if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = FlowNet(img_shape=(28, 28, 1), hidden_channels=128, K=16, L=2).to(device)
    rand_inp = torch.rand(16, 1, 28, 28).to(device)
    tmp_inp, tmp_log = flow(rand_inp, reverse=False)