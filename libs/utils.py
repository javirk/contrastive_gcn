import torch
import torch.nn as nn
import numpy as np
import yaml
import wandb
import shutil
from typing import Optional

_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}


def str2bool(value, raise_exc=False):
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


@torch.no_grad()
def get_cam_segmentation(seg):
    device = seg.device
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


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    device = labels.device if device is None else device
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}".format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one. Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width, device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)  # + eps


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

    def reset(self):
        for m in self.meters:
            m.reset()

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


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

def mapping_tensor(A, B):
    A_enc = torch.zeros((int(A.max()) + 1,) * 2)
    A_enc[A, torch.arange(A.shape[0])] = 1

    v = torch.argmax(A_enc, dim=0)
    B_enc = torch.zeros(A_enc.shape[0], B.shape[0])
    B_enc[B, torch.arange(B.shape[0])] = 1
    return v @ B_enc.long()