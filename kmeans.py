import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import argparse
import libs.utils as utils
from libs.kmeans_utils import save_embeddings_to_disk, eval_kmeans
from libs.data.mnist import MNISTSuperpixel
from libs.common_config import get_image_transforms, get_dataset
from models.gcn import GCN
from models.builder import SegGCN
from models.backbones.unet import UNet

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config',
                    default='configs/configs-default.yml',
                    type=str,
                    help='Path to the config file')
parser.add_argument('-u', '--ubelix',
                    default=1,
                    type=int,
                    help='Running on ubelix (0 is no)')
parser.add_argument('-crf', '--crf-postprocessing',
                    type=utils.str2bool,
                    default=True,
                    help='Apply CRF postprocessing')

FLAGS, unparsed = parser.parse_known_args()


def main(p):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset = MNISTSuperpixel('data/', train=False, download=True)
    image_tf = get_image_transforms(p)
    dataset = get_dataset(p, root='data/', image_set='val', transform=image_tf)
    dataloader = DataLoader(dataset, batch_size=p['val_kwargs']['batch_size'], shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)
    backbone = UNet(n_channels=3, n_classes=1)
    gcn = GCN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
              output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, backbone=backbone, graph_network=gcn)
    model.to(device)
    model = nn.DataParallel(model)
    model.eval()

    # Kmeans Clustering
    n_clusters = 21
    results_miou = []
    save_embeddings_to_disk(p, dataloader, model, n_clusters=n_clusters, seed=1234, device=device)
    eval_stats = eval_kmeans(p, dataset, n_clusters=n_clusters, verbose=True)
    results_miou.append(eval_stats['mIoU'])
    print('Average mIoU is %2.1f' % results_miou[0])


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config_env = utils.read_config('configs/env_configs.yml')
    config.update(config_env)
    config.update(vars(FLAGS))

    config['ubelix'] = FLAGS.ubelix
    num_workers = 8

    if FLAGS.ubelix == 0:
        config['val_kwargs']['batch_size'] = 4
        num_workers = 1

    if 'runs' in FLAGS.config:
        date_run = FLAGS.config.split('/')[-2].split('_')[-1]
        config['pretrained_model'] = date_run + '.pth'

    main(config)