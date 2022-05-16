import torch
import math
import numpy as np
import os
import torchvision.transforms as T
from libs.data.transforms import NodeDropping, EdgePerturbation, ToTensor
from models.backbones.unet import UNet
from models.modules.deeplab import AffinityDeeplab, ContrastiveDeeplab


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        del p['optimizer_kwargs']['momentum'], p['optimizer_kwargs']['nesterov']
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
        A.Resize(p['resolution'], p['resolution']),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # return T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_sal_transforms(p):
    import albumentations as A
    return A.Compose([A.Resize(p['resolution'] // 8, p['resolution'] // 8)])

def get_joint_transforms(p):
    import albumentations as A
    return A.Compose([
        ToTensor(transpose_mask=True)
    ])


# def get_dataset(p, root, image_set, transform=None, aug_transformations=None, sal_transformations=None):
def get_dataset(p, root, image_set, **kwargs):
    if p['dataset'] == 'MNIST':
        from libs.data.mnist import MNISTSuperpixel
        train = True if 'train' in image_set else False
        return MNISTSuperpixel(root, train=train, download=True, **kwargs)

    elif p['dataset'].upper() == 'PASCAL':
        from libs.data.pascal_voc import Pascal
        return Pascal(root=os.path.join(root, 'VOCSegmentation'), image_set=image_set, **kwargs)


def get_segmentation_model(p):
    # Get backbone
    if p['backbone'] == 'resnet18':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet18'](pretrained=False)
        backbone_channels = 512

    elif p['backbone'] == 'resnet50':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet50'](pretrained=False)
        backbone_channels = 2048

    elif p['backbone'] == 'resnet38_aff':
        from models.modules.resnet38_aff import AffinityNet
        backbone = AffinityNet()
        backbone_channels = 4096

    elif p['backbone'] == 'unet':
        backbone = UNet(p, n_channels=3, n_classes=1)

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    if p['backbone_kwargs']['dilated']:
        if 'aff' in p['backbone']:
            print('Dilation not applied because we are using Affinity Network')
        else:
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
    if p['backbone'] == 'unet':
        return backbone

    elif 'aff' in p['backbone']:
        return AffinityDeeplab(p, backbone, head, upsample=True)

    else:
        return ContrastiveDeeplab(p, backbone, head, upsample=True, use_classification_head=True)
