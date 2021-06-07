from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def get_train_loader(config):
    transform = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.RandomRotation(15),
                                    transforms.ToTensor()])
    train_dataset = CIFAR10(root='../datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              pin_memory=True, drop_last=True, shuffle=True)
    return train_loader


def get_test_loader(config):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CIFAR10(root='datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             pin_memory=True, drop_last=False, shuffle=False)
    return test_loader


if __name__ == '__main__':
    import cv2
    import numpy as np
    from config import Config
    t = get_train_loader(Config)
    images, labels = next(iter(t))
    print(images.shape, labels.shape)

    tmp = (images[0].numpy()) * 255
    tmp = np.transpose(tmp, [1, 2, 0])
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
    print(tmp.shape)
    print(np.min(tmp), np.max(tmp))
    cv2.imwrite('tmp.png', tmp)
