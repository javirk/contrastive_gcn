import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
import wandb
import argparse
from datetime import datetime
from models.gcn import GCN
from models.builder import SegGCN
import libs.utils as utils
from libs.train_utils import train_seg
from libs.losses import BalancedCrossEntropyLoss
from libs.common_config import get_optimizer, get_sal_transforms, get_joint_transforms, adjust_learning_rate, get_dataset,\
    get_image_transforms, get_segmentation_model
from libs.data.transforms import AffinityPerturbation, AffinityDropping

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


def main(p):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}_graph.pth'
    utils.copy_file(FLAGS.config, f'runs/{current_time}.yml')  # This should be improved in the future maybe

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time + '_seg',
                   notes=f"{p['dataset']} - {p['backbone']}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sal_tf = get_sal_transforms(p)
    image_tf = get_image_transforms(p)
    joint_tf = get_joint_transforms(p)
    dataset = get_dataset(p, root='data/', image_set='trainaug', transform=image_tf, sal_transform=sal_tf,
                          joint_transform=joint_tf)
    dataloader = DataLoader(dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=True)

    # backbone = UNet(p, n_channels=3, n_classes=1)
    encoder = get_segmentation_model(p)
    gcn = GCN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
              output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, encoder=encoder, graph_network=gcn).to(device)
    model = nn.DataParallel(model)
    model.train()
    model.module.encoder.eval()

    crit_bce = BalancedCrossEntropyLoss()

    optimizer = get_optimizer(p, model.parameters())
    graph_transformation = Compose([AffinityDropping(), AffinityPerturbation(p=0.05)])

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        train_seg(p, dataloader, model, crit_bce, graph_transformation, optimizer, epoch, device)

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.module.graph.state_dict(),
                    'epoch': epoch + 1}, p['checkpoint'])

    wandb.finish()


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 2
        num_workers = 0

    main(config)