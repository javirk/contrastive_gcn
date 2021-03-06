import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
import wandb
import argparse
from datetime import datetime
from models.gcn import GCN
from models.builder import SegGCN
import libs.utils as utils
from libs.train_utils import train
from libs.common_config import get_optimizer, get_augmentation_transforms, adjust_learning_rate, get_dataset,\
    get_image_transforms, get_segmentation_model

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
    world_size = 0
    rank = 0
    if world_size > 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}.pth'
    utils.copy_file(FLAGS.config, f'runs/{current_time}.yml')  # This should be improved in the future maybe

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time, notes=f"{p['dataset']} - {p['backbone']}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    aug_tf = get_augmentation_transforms(p)
    image_tf = get_image_transforms(p)
    # dataset = MNISTSuperpixel('data/', train=True, aug_transform=aug_transformations, download=True)
    dataset = get_dataset(p, root='data/', image_set='trainaug', transform=image_tf, aug_transformations=aug_tf)
    dataloader = DataLoader(dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=True)

    # backbone = UNet(p, n_channels=3, n_classes=1)
    backbone = get_segmentation_model(p)
    gcn = GCN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
              output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, backbone=backbone, graph_network=gcn).to(device)
    if world_size > 0:
        model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model = nn.DataParallel(model)
    model.train()

    optimizer = get_optimizer(p, model.parameters())

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        train(p, dataloader, model, optimizer, epoch, device)

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['checkpoint'])

    wandb.finish()
    # dist.destroy_process_group()


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 4
        num_workers = 1

    main(config)

    # world_size = torch.cuda.device_count()
    # print('Let\'s use', world_size, 'GPUs!')
    # if world_size > 0:
    #     mp.spawn(main, args=(world_size, config), nprocs=world_size, join=True)
    # else:
    #     main('cpu', 0, config)
