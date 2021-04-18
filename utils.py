import re
import copy

import torch


def get_proper_cuda_device(device, verbose=True):
    if not isinstance(device, list):
        device = [device]
    count = torch.cuda.device_count()
    if verbose:
        print("[Builder]: Found {} gpu".format(count))
    for i in range(len(device)):
        d = device[i]
        did = None
        if isinstance(d, str):
            if re.search("cuda:[\d]+", d):
                did = int(d[5:])
        elif isinstance(d, int):
            did = d
        if did is None:
            raise ValueError("[Builder]: Wrong cuda id {}".format(d))
        if did < 0 or did >= count:
            if verbose:
                print("[Builder]: {} is not found, ignore.".format(d))
            device[i] = None
        else:
            device[i] = did
    device = [d for d in device if d is not None]
    return device


def get_proper_device(devices, verbose=True):
    origin = copy.copy(devices)
    devices = copy.copy(devices)
    if not isinstance(devices, list):
        devices = [devices]
    use_cpu = any([d.find("cpu")>=0 for d in devices])
    use_gpu = any([(d.find("cuda")>=0 or isinstance(d, int)) for d in devices])
    assert not (use_cpu and use_gpu), "{} contains cpu and cuda device.".format(devices)
    if use_gpu:
        devices = get_proper_cuda_device(devices, verbose)
        if len(devices) == 0:
            if verbose:
                print("[Builder]: Failed to find any valid gpu in {}, use `cpu`.".format(origin))
            devices = ["cpu"]
    return devices
