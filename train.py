import os
import shutil

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models.Glow import Glow
from models import lr_scheduler
from models.modules import thops
from utils import get_proper_device, save
from dataloader.mnist import get_train_loader


def train(config):
    log_dir = f'{config.exp_path}/logs'
    chkpt_dir = f'{config.exp_path}/chkpt'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)
    shutil.copy('config.py', f'{config.exp_path}/config.py')
    writer = SummaryWriter(log_dir=log_dir)

    # Define model
    model = Glow(config)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    schedule_args = config.lr_scheduler_args
    schedule_args["init_lr"] = config.lr
    lr_schedule = {
        "func": getattr(lr_scheduler, config.lr_scheduler),
        "args": schedule_args
    }

    # Set devices
    devices = get_proper_device(config.device, False)
    data_device = devices[0]
    model = model.cuda(device=devices[0])

    # Get dataloader
    train_loader = get_train_loader(Config)

    n_epochs = config.num_iters // len(train_loader)
    cur_step = 0

    model.train()
    for epoch in range(n_epochs):
        progress = tqdm(train_loader)
        for idx, batch in enumerate(progress):
            # Learning rate scheduling
            lr = lr_schedule["func"](global_step=cur_step, **lr_schedule["args"])
            for param_group in optim.param_groups:
                param_group['lr'] = lr

            optim.zero_grad()

            img, label = batch
            img = img.to(data_device)
            label = label.to(data_device)
            onehot_label = thops.onehot(label, num_classes=config.y_classes)
            # onehot_label = None

            # Initialize ActNorm weights
            if cur_step == 0:
                model(img[:config.batch_size//len(devices), ...],
                      onehot_label[:config.batch_size // len(devices), ...] if onehot_label is not None else None)

            # Parallel
            if len(devices) > 1 and not hasattr(model, "module"):
                print(f"[Parallel] move to {devices}")
                model = torch.nn.parallel.DataParallel(model, devices, devices[0])

            # Forward
            z, nll, logits = model(x=img, y_onehot=onehot_label)
            loss_generative = Glow.loss_generative(nll)
            loss_classes = 0
            if config.y_condition:
                loss_classes = (Glow.loss_multi_classes(logits, onehot_label)
                                if config.y_criterion == "multi-classes" else Glow.loss_class(logits, label))
            loss = loss_generative + loss_classes * config.classification_weight

            # Optimize
            model.zero_grad()
            optim.zero_grad()
            loss.backward()
            if config.max_grad_clip is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.max_grad_clip)
            if config.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), config.max_grad_norm)
            optim.step()

            cur_step += 1

            # Logging optimization info
            if cur_step % config.lr_log_freq == 0:
                writer.add_scalar("train_info/lr", lr, cur_step)
                if config.max_grad_norm is not None:
                    writer.add_scalar("train_info/grad_norm", grad_norm, cur_step)

            # Logging losses
            # print(f"Train Epoch {epoch} / Iter {idx} || {loss_generative}")
            writer.add_scalar("loss/nll", loss_generative, cur_step)
            if config.y_condition:
                writer.add_scalar("loss/classification", loss_classes, cur_step)

            # Save checkpoints
            if cur_step % config.ckpt_save_freq == 0:
                save(global_step=cur_step, graph=model, optim=optim,
                     pkg_dir=chkpt_dir, is_best=True, max_checkpoints=config.max_ckpts)

            # Sanity check
            if cur_step % config.sanity_freq == 0:
                recon = model(z=z, y_onehot=onehot_label, reverse=True)
                print(f'Cur Step:{cur_step} | minimum: {torch.min(recon)}, maximum: {torch.max(recon)}')
                for bi in range(min([len(recon), 4])):
                    writer.add_image("reverse/{}".format(bi), torch.cat((recon[bi], img[bi]), dim=1), cur_step)

            if cur_step % config.infer_freq == 0:
                inf = model(z=None, y_onehot=onehot_label, eps_std=0.5, reverse=True)
                for bi in range(min([len(inf), 4])):
                    writer.add_image("sample/{}".format(bi), inf[bi], cur_step)


if __name__ == '__main__':
    train(Config)
