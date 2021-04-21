import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_train_loader(config):
    transform = transforms.Compose([transforms.Pad(2),
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.expand(3, -1, -1))])
    train_dataset = MNIST(root='datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              pin_memory=True, drop_last=True, shuffle=True)
    return train_loader


def get_test_loader(config):
    transform = transforms.Compose([transforms.Pad(2),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.expand(3, -1, -1))])
    test_dataset = MNIST(root='datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             pin_memory=True, drop_last=False, shuffle=False)
    return test_loader


if __name__ == '__main__':
    import cv2
    import numpy as np
    from config import Config
    t = get_train_loader(Config)
    images, labels = next(iter(t))
    tmp = (images[0, 0].numpy()) * 255.
    print(images.shape, labels.shape)
    print(np.min(tmp), np.max(tmp))
    cv2.imwrite('tmp.png', tmp)
