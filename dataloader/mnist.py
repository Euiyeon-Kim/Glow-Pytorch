import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_train_valid_loader(config):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='../datasets', train=True, transform=transform, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=True)
    return train_loader, valid_loader


if __name__ == '__main__':
    from config import Config
    t, v = get_train_valid_loader(Config)
    images, labels = next(iter(v))
    print(images.shape, labels.shape)
