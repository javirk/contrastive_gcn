import torch
import torch.nn as nn
import numpy as np
import yaml
import wandb
import shutil
from typing import Optional
import torch.nn.functional as F

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

def remove_module(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.rsplit('module.')[1]
        new_state[k] = v
    return new_state


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


class SemsegMeter(object):
    def __init__(self, num_classes, class_names, has_bg=True, ignore_index=255):
        self.num_classes = num_classes + int(has_bg)
        self.class_names = class_names
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
        assert (ignore_index == 255)
        self.ignore_index = ignore_index

    def update(self, pred, gt):
        valid = (gt != self.ignore_index)

        for i_part in range(0, self.num_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes

    def return_score(self, verbose=True):
        jac = [0] * self.num_classes
        for i_part in range(self.num_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)

        if verbose:
            print('Evaluation of semantic segmentation ')
            print('mIoU is %.2f' % (100 * eval_result['mIoU']))
            for i_part in range(self.num_classes):
                print('IoU class %s is %.2f' % (self.class_names[i_part], 100 * jac[i_part]))

        return eval_result


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

def dict_to_file(d, dst):
    with open(dst, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)


def unfold(img, radius):
    assert img.dim() == 4, 'Unfolding requires NCHW batch'
    N, C, H, W = img.shape
    diameter = 2 * radius + 1
    return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)


def generate_aff(f_w, f_h, aff_mat, radius):
    bs = aff_mat.shape[0]
    aff = torch.zeros([bs, f_w * f_h, f_w * f_h])
    aff_mask = torch.zeros([bs, f_w, f_h])
    aff_mask_pad = F.pad(aff_mask, (radius, radius, radius, radius), 'constant')
    # aff_mat = torch.squeeze(aff_mat)
    for i in range(f_w):
        for j in range(f_h):
            ind = i * f_h + j
            center_x = i + radius
            center_y = j + radius
            aff_mask_pad[:, center_x - radius: (center_x + radius + 1),
            center_y - radius: (center_y + radius + 1)] = aff_mat[:, :, :, i, j]
            aff_mask_nopad = aff_mask_pad[:, radius:-radius, radius:-radius]
            aff[:, ind] = aff_mask_nopad.reshape(bs, -1)
            aff_mask_pad = 0 * aff_mask_pad

    return aff
