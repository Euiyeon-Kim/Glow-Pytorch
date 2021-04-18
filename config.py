from enum import Enum

from dotmap import DotMap


class Datasets(Enum):
    def __str__(self):
        return '%s' % self.value

    CelebA = 'celeba',
    MNIST = 'mnist'


class Config:
    # Options
    exp_name = 'tmp'
    dataset_name = Datasets.MNIST
    device = ["cuda:0"]     # ["cuda:0", "cuda:1"]

    # Train
    batch_size = 16

    # Paths
    exp_path = f'exps/{exp_name}'
    train_root_dir = f'datasets/{dataset_name}/train'
    valid_root_dir = f'datasets/{dataset_name}/valid'

    # Architecture
    K = 16
    L = 2
    hidden_channels = 64
    actnorm_scale = 1.0
    LU_decomposed = False

    # Ablation
    learn_top = False
    y_condition = False
    y_classes = 10

    # Dataset
    img_shape = (28, 28, 1)

    # Optimizer
    beta1 = 0.9
    beta2 = 0.999
