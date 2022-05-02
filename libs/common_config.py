import torch
import torch.nn as nn
import math
import numpy as np
import os
import torchvision.transforms as T
import torchvision.models.resnet as resnet
from libs.data.transforms import NodeDropping, EdgePerturbation, ToTensor
from models.backbones.unet import UNet
from models.modules.deeplab import ContrastiveDeeplab


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    elif p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_augmentation_transforms(p):
    # Add something to use the configurations
    return T.Compose([NodeDropping(percentage_keep=0.90), EdgePerturbation()])


def get_image_transforms(p):
    import albumentations as A
    return A.Compose([
        A.Resize(p['resolution'], p['resolution']), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor(transpose_mask=True)
    ])
    # return T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def get_dataset(p, root, image_set, transform=None, aug_transformations=None):
    if p['dataset'] == 'MNIST':
        from libs.data.mnist import MNISTSuperpixel
        train = True if 'train' in image_set else False
        return MNISTSuperpixel(root, train=train, aug_transform=aug_transformations, download=True)

    elif p['dataset'].upper() == 'PASCAL':
        from libs.data.pascal_voc import Pascal
        return Pascal(root=os.path.join(root, 'VOCSegmentation'), image_set=image_set, transform=transform,
                      aug_transform=aug_transformations)


def get_segmentation_model(p):
    # Get backbone
    if p['backbone'] == 'resnet18':
        backbone = resnet.__dict__['resnet18'](pretrained=False)
        backbone_channels = 512

    elif p['backbone'] == 'resnet50':
        backbone = resnet.__dict__['resnet50'](pretrained=False)
        backbone_channels = 2048

    elif p['backbone'] == 'unet':
        backbone = UNet(p, n_channels=3, n_classes=1)

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    if p['backbone_kwargs']['dilated']:
        from models.modules.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    # Get head
    if p['head']['model'] == 'deeplab' and p['backbone'] != 'unet':
        from models.modules.deeplab import DeepLabHead
        nc = p['gcn_kwargs']['ndim']  # Because ndim in gcn_kwargs is the input dim
        head = DeepLabHead(backbone_channels, nc)

    else:
        raise ValueError('Invalid head {}'.format(p['head']))

    # Compose model from backbone and head
    if p['backbone'] != 'unet':
        model = ContrastiveDeeplab(p, backbone, head, True, True)
        if p['model_kwargs']['norm_layer'].lower() == 'groupnorm':
            model = batch_norm_to_group_norm(model)
        return model
    else:
        return backbone


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer
