import torch
import math
import numpy as np
import os
import torchvision.transforms as T
from libs.data.transforms import NodeDropping, EdgePerturbation, ToTensor


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
    return T.Compose([NodeDropping(), EdgePerturbation()])

def get_image_transforms():
    import albumentations as A
    return A.Compose([
        A.Resize(224, 224), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ToTensor()
    ])
    # return T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_dataset(p, root, image_set, transform=None, aug_transformations=None):
    if p['dataset'] == 'MNIST':
        from libs.data.mnist import MNISTSuperpixel
        train = True if image_set == 'train' else False
        return MNISTSuperpixel(root, train=train, aug_transform=aug_transformations, download=True)

    elif p['dataset'].upper() == 'PASCAL':
        from libs.data.pascal_voc import Pascal
        return Pascal(root=os.path.join(root, 'VOCSegmentation'), image_set=image_set, transform=transform,
                      aug_transform=aug_transformations)
