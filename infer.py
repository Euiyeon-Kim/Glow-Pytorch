import os

import numpy as np

from test_conf import Config
from models.Glow import Glow
from utils import get_proper_device, load
from dataloader.mnist import get_train_loader


INFER_SAMPLE_NUM = 100


if __name__ == '__main__':
    infer_dir = f'{Config.exp_path}/infer'
    os.makedirs(infer_dir, exist_ok=True)

    model = Glow(Config)
    load(step_or_path='best', graph=model, pkg_dir=f'{Config.exp_path}/chkpt', device='cpu')
    test_loader = iter(get_train_loader(Config))

    devices = get_proper_device(Config.device, False)
    data_device = devices[0]
    model = model.cuda(device=devices[0])

    for i in range(INFER_SAMPLE_NUM):
        img, _ = next(test_loader)
        img = img.to(data_device)

        z, nll, logits = model(x=img, y_onehot=None)
        tmp = z.cpu().detach().numpy()
        print(np.min(tmp), np.max(tmp))
        recon = model(z=z, y_onehot=None, reverse=True)
        recon = recon.cpu().numpy()

        print(z.shape)
        print(recon.shape)
        print(np.min(recon), np.max(recon))
        exit()
