import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
import wandb
import argparse
from datetime import datetime
from models.gcn import GCN
from models.builder import SegGCN
import libs.utils as utils
from libs.losses import BalancedCrossEntropyLoss, ModelLossSemsegGatedCRF
from libs.train_utils import forward_aff
from libs.common_config import get_optimizer, get_sal_transforms, get_joint_transforms, adjust_learning_rate, \
    get_dataset, get_train_transforms, get_val_transforms, get_segmentation_model

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config',
                    default='configs/configs-default_aff.yml',
                    type=str,
                    help='Path to the config file')
parser.add_argument('-u', '--ubelix',
                    default=1,
                    type=int,
                    help='Running on ubelix (0 is no)')

FLAGS, unparsed = parser.parse_known_args()


def main(p):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}_aff.pth'
    utils.copy_file(FLAGS.config, f'runs/{current_time}.yml')  # This should be improved in the future maybe

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time + '_aff',
                   notes=f"{p['dataset']} - {p['backbone']}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sal_tf = get_sal_transforms(p)
    image_train_tf = get_train_transforms(p)
    image_val_tf = get_val_transforms(p)
    joint_tf = get_joint_transforms(p)

    train_dataset = get_dataset(p, root='data/', image_set='trainaug', transform=image_train_tf, sal_transform=sal_tf,
                                joint_transform=joint_tf)
    train_loader = DataLoader(train_dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)

    val_dataset = get_dataset(p, root='data/', image_set='val', transform=image_val_tf, sal_transform=sal_tf,
                              joint_transform=joint_tf)
    val_loader = DataLoader(val_dataset, batch_size=p['val_kwargs']['batch_size'], shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)

    model = get_segmentation_model(p)
    model = nn.DataParallel(model)
    model.to(device)
    model.train()

    crit_aff = ModelLossSemsegGatedCRF()
    crit_bce = BalancedCrossEntropyLoss()

    optimizer = get_optimizer(p, model.parameters())

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        forward_aff(p, train_loader, model, crit_aff, crit_bce, optimizer, epoch, device, phase='train')
        forward_aff(p, val_loader, model, crit_aff, crit_bce, optimizer, epoch, device, phase='val')

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['checkpoint'])

    wandb.finish()


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 4
        config['val_kwargs']['batch_size'] = 4
        num_workers = 0

    main(config)
