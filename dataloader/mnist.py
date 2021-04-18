import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_train_loader(config):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              pin_memory=True, drop_last=True, shuffle=True)
    return train_loader


if __name__ == '__main__':
    from config import Config
    t = get_train_loader(Config)
    images, labels = next(iter(t))
    print(images.shape, labels.shape)
