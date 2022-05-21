import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import wandb
import argparse
from datetime import datetime
from models.gcn import AGNN
from models.builder import SegGCN
import libs.utils as utils
from libs.train_utils import forward_seg
from libs.losses import BalancedCrossEntropyLoss
from libs.common_config import get_optimizer, get_sal_transforms, get_joint_transforms, adjust_learning_rate, get_dataset,\
    get_train_transforms, get_segmentation_model, get_val_transforms
from libs.data.transforms import AffinityPerturbation, AffinityDropping

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config',
                    default='configs/configs-default_seg.yml',
                    type=str,
                    help='Path to the config file')
parser.add_argument('-ac', '--affinity-config',
                    type=str,
                    help='Path to the config file')
parser.add_argument('-u', '--ubelix',
                    default=1,
                    type=int,
                    help='Running on ubelix (0 is no)')

FLAGS, unparsed = parser.parse_known_args()


def main(p):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}_graph.pth'
    utils.dict_to_file(p, f'runs/{current_time}.yml')

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time + '_seg',
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
    val_loader = DataLoader(val_dataset, batch_size=p['val_kwargs']['batch_size'], shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=True)

    encoder = get_segmentation_model(p)
    gcn = AGNN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
               output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, encoder=encoder, graph_network=gcn).to(device)
    model = nn.DataParallel(model)
    model.train()
    model.module.encoder.eval()

    crit_bce = BalancedCrossEntropyLoss()

    optimizer = get_optimizer(p, model.parameters())
    graph_transformation = Compose([AffinityDropping(p=0.2), AffinityPerturbation(p=0.3)])

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        last_it = forward_seg(p, train_loader, model, crit_bce, graph_transformation, optimizer, epoch, device,
                              phase='train')
        forward_seg(p, val_loader, model, crit_bce, graph_transformation, optimizer, epoch, device, phase='val',
                    last_it=last_it)

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.module.graph.state_dict(),
                    'epoch': epoch + 1}, p['checkpoint'])

    wandb.finish()


if __name__ == '__main__':
    # This the configuration for the previous affinity.
    # We read it and then update the parameters with the FLAGS.config file
    config_seg = utils.read_config(FLAGS.affinity_config)
    config = utils.read_config(FLAGS.config)
    config_seg.update(config)

    config_seg['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config_seg['train_kwargs']['batch_size'] = 4
        num_workers = 0

    if 'runs' in FLAGS.affinity_config:
        print(f'Using run {FLAGS.affinity_config}')
        date_run = FLAGS.affinity_config.split('/')[-1].split('.')[-2]
        config_seg['pretrained_backbone'] = date_run + '_aff.pth'
    print(config_seg)
    main(config_seg)