import torch
import torchvision.transforms as T
from libs.data.transforms import NodeDropping, EdgePerturbation


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def get_augmentation_transforms(p):
    # Add something to use the configurations
    return T.Compose([NodeDropping(), EdgePerturbation()])
