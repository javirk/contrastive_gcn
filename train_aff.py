import os
import torch
import torch.nn as nn
import torch.distributed as dist
from libs.losses import BalancedCrossEntropyLoss, ModelLossSemsegGatedCRF
from torch_geometric.loader import DataLoader
import wandb
import argparse
from datetime import datetime
from models.gcn import GCN
from models.builder import SegGCN
import libs.utils as utils
from libs.train_utils import train_aff
from libs.common_config import get_optimizer, get_sal_transforms, get_joint_transforms, adjust_learning_rate, \
    get_dataset, get_image_transforms, get_segmentation_model

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config',
                    default='configs/configs-default.yml',
                    type=str,
                    help='Path to the config file')
parser.add_argument('-u', '--ubelix',
                    default=1,
                    type=int,
                    help='Running on ubelix (0 is no)')

FLAGS, unparsed = parser.parse_known_args()


# def main(rank, world_size, p):
def main(p):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}.pth'
    utils.copy_file(FLAGS.config, f'runs/{current_time}.yml')  # This should be improved in the future maybe

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time, notes=f"{p['dataset']} - {p['backbone']}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sal_tf = get_sal_transforms(p)
    image_tf = get_image_transforms(p)
    joint_tf = get_joint_transforms(p)

    dataset = get_dataset(p, root='data/', image_set='trainaug', transform=image_tf, sal_transform=sal_tf,
                          joint_transform=joint_tf)
    dataloader = DataLoader(dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=True)

    model = get_segmentation_model(p)
    # gcn = GCN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
    #           output_dim=p['gcn_kwargs']['output_dim'])

    # model = SegGCN(p, backbone=backbone, graph_network=gcn).to(device)
    model = nn.DataParallel(model)
    model.train()

    crit_aff = ModelLossSemsegGatedCRF()
    crit_bce = BalancedCrossEntropyLoss()

    optimizer = get_optimizer(p, model.parameters())

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        train_aff(p, dataloader, model, crit_aff, crit_bce, optimizer, epoch, device)

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['checkpoint'])

    wandb.finish()


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 4
        num_workers = 0

    main(config)

    # world_size = torch.cuda.device_count()
    # print('Let\'s use', world_size, 'GPUs!')
    # if world_size > 0:
    #     mp.spawn(main, args=(world_size, config), nprocs=world_size, join=True)
    # else:
    #     main('cpu', 0, config)
