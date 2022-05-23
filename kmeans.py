import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import argparse
from torchvision.transforms import Compose
import libs.utils as utils
from models.gcn import AGNN
from models.builder import SegGCN
from libs.data.transforms import AffinityPerturbation, AffinityDropping
from libs.kmeans_utils import save_embeddings_to_disk, eval_kmeans
from libs.common_config import get_val_transforms, get_sal_transforms, get_joint_transforms, get_dataset, \
    get_segmentation_model

parser = argparse.ArgumentParser()

parser.add_argument('-sc', '--segmentation-config',
                    type=str,
                    help='Config file for the environment')

parser.add_argument('-ac', '--affinity-config',
                    type=str,
                    help='Config file for the environment')

parser.add_argument('-u', '--ubelix',
                    default=1,
                    type=int,
                    help='Running on ubelix (0 is no)')

parser.add_argument('-r', '--resolution',
                    default=512,
                    type=int,
                    help='Square resolution')

parser.add_argument('-crf', '--crf-postprocessing',
                    type=utils.str2bool,
                    default=True,
                    help='Apply CRF postprocessing')

FLAGS, unparsed = parser.parse_known_args()


def main(p):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sal_tf = get_sal_transforms(p)
    image_tf = get_val_transforms(p)
    joint_tf = get_joint_transforms(p)
    dataset = get_dataset(p, root='data/', image_set='val', transform=image_tf, sal_transform=sal_tf,
                          joint_transform=joint_tf)
    dataloader = DataLoader(dataset, batch_size=p['val_kwargs']['batch_size'], shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)

    backbone = get_segmentation_model(p)
    gcn = AGNN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
               output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, encoder=backbone, graph_network=gcn).to(device)
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
    config = utils.read_config(FLAGS.segmentation_config)
    config_aff = utils.read_config(FLAGS.affinity_config)
    config_env = utils.read_config('configs/env_configs.yml')
    config.update(config_env)
    config.update(vars(FLAGS))

    print(f'Using files {FLAGS.segmentation_config} and {FLAGS.affinity_config}')

    config['ubelix'] = FLAGS.ubelix
    num_workers = 8

    if FLAGS.ubelix == 0:
        config['val_kwargs']['batch_size'] = 2
        num_workers = 0

    if 'runs' in FLAGS.affinity_config:
        date_run = FLAGS.affinity_config.split('/')[-1].split('.')[-2]
        config['pretrained_backbone'] = date_run + '_aff.pth'

    if 'runs' in FLAGS.segmentation_config:
        date_run = FLAGS.segmentation_config.split('/')[-1].split('.')[-2]
        config['pretrained_gcn'] = date_run + '_graph.pth'

    print(config)
    main(config)