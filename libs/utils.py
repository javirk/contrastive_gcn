import torch
import torch.nn as nn
import numpy as np
import yaml
import wandb


@torch.no_grad()
def get_cam_segmentation(seg, device):
    # Coarse segmentation
    coarse = seg.float()
    coarse = nn.functional.interpolate(coarse, (14, 14))
    coarse = nn.functional.interpolate(coarse, (28, 28))

    # This dilates the coarse segmentation
    kernel = torch.ones((1, 1, 3, 3), device=device)
    coarse = torch.clamp(nn.functional.conv2d(coarse, kernel, padding=(1, 1)), min=0, max=1)

    coarse_onehot = coarse.repeat_interleave(2, dim=1)
    coarse_onehot[:, 0] = (coarse[:, 0] == 0).float()
    coarse_onehot[:, 1] = (coarse[:, 0] == 1).float()
    return coarse_onehot


def one_hot(vec, ndims):
    if torch.is_tensor(vec):
        return torch.eye(ndims)[vec.reshape(-1)].squeeze()
    else:
        return np.squeeze(np.eye(ndims)[vec.reshape(-1)])


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def to_wandb(self, batch, prefix=None):
        entries = {f'{prefix}/{i.name}': i.avg for i in self.meters}
        wandb.log(entries, step=batch)
