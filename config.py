from enum import Enum



class Datasets(Enum):
    def __str__(self):
        return '%s' % self.value

    CelebA = 'celeba',
    MNIST = 'mnist'


class Config:
    # Options
    exp_name = 'why'
    dataset_name = Datasets.MNIST
    device = ["cuda:0"]     # ["cuda:0", "cuda:1"]

    # Train
    num_iters = 1000000
    batch_size = 128

    # Paths
    exp_path = f'exps/{exp_name}'
    train_root_dir = f'datasets/{dataset_name}/train'
    valid_root_dir = f'datasets/{dataset_name}/valid'

    # Architecture
    K = 10
    L = 3
    hidden_channels = 512
    actnorm_scale = 1.0
    LU_decomposed = False

    # Ablation
    learn_top = False
    classification_weight = 0.01
    y_condition = False
    y_classes = 10
    y_criterion = "multi-classes"           # ["multi-classes", "single-class"]

    # Dataset
    img_shape = (32, 32, 3)

    # Optimizer
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    lr_scheduler = "noam_learning_rate_decay"
    lr_scheduler_args = {
        'warmup_steps': 4000,
        'minimum': 1e-4
    }
    max_grad_clip = 5
    max_grad_norm = 100

    # Log
    lr_log_freq = 50
    ckpt_save_freq = 2000
    max_ckpts = 20
    sanity_freq = 1000
    infer_freq = 1000
