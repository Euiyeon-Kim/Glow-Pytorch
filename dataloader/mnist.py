import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_train_loader(config):
    transform = transforms.Compose([transforms.Resize((config.img_shape[0], config.img_shape[1])),
                                    transforms.ToTensor()])
    train_dataset = MNIST(root='datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              pin_memory=True, drop_last=True, shuffle=True)
    return train_loader


def get_test_loader(config):
    transform = transforms.Compose([transforms.Resize((config.img_shape[0], config.img_shape[1])),
                                    transforms.ToTensor()])
    test_dataset = MNIST(root='datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             pin_memory=True, drop_last=False, shuffle=False)
    return test_loader



if __name__ == '__main__':
    import cv2
    from config import Config
    t = get_train_loader(Config)
    images, labels = next(iter(t))
    print(images.shape, labels.shape)
    tmp = (images[0, 0].numpy() + 1.0) * 127.5
    cv2.imwrite('tmp.png', tmp)
