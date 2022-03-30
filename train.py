import torch
from torch_geometric.loader import DataLoader
import wandb
import argparse
from datetime import datetime
from libs.data.mnist import MNISTSuperpixel
from models.gcn import GCN
from models.builder import SegGCN
from models.backbones.unet import UNet
import libs.utils as utils
from libs.train_utils import train
from libs.common_config import get_optimizer, get_augmentation_transforms, adjust_learning_rate


def main(p):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}.pth'

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time, notes=f"{p['dataset']} - {p['backbone']}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    aug_transformations = get_augmentation_transforms(p)
    dataset = MNISTSuperpixel('data/', train=True, aug_transform=aug_transformations, download=True)
    dataloader = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True, drop_last=True)

    backbone = UNet(n_channels=1, n_classes=2)
    gcn = GCN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
              output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, backbone=backbone, graph_network=gcn)
    model.to(device)

    optimizer = get_optimizer(p, model.parameters())

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        train(p, dataloader, model, optimizer, epoch, device)

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1},
                   p['checkpoint'])

    wandb.finish()


if __name__ == '__main__':
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
    config = utils.read_config(FLAGS.config)

    if FLAGS.ubelix == 0:
        config['batch_size'] = 4

    config['ubelix'] = FLAGS.ubelix

    main(config)
