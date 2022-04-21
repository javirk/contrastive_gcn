import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import wandb
import glob
from time import time
import argparse
from torch.nn.functional import cross_entropy
from models.gcn import GCN
from models.builder import SegGCN
from models.backbones.unet import UNet
import libs.utils as utils
from libs.common_config import get_optimizer, get_augmentation_transforms, get_dataset, get_image_transforms

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
    schedule = torch.profiler.schedule(warmup=1, active=1, repeat=1, wait=0)
    profile_dir = "profiler/output/"
    profiler = torch.profiler.profile(schedule=schedule,
                                      on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                                      with_stack=True)
    # wandb.init(project='Contrastive-Graphs', config=p, notes='profiler_test')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    aug_tf = get_augmentation_transforms(p)
    image_tf = get_image_transforms(p)
    dataset = get_dataset(p, root='data/', image_set='train', transform=image_tf, aug_transformations=aug_tf)
    dataloader = DataLoader(dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                            num_workers=num_workers)

    backbone = UNet(n_channels=3, n_classes=1)
    gcn = GCN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
              output_dim=p['gcn_kwargs']['output_dim'])

    model = SegGCN(p, backbone=backbone, graph_network=gcn)
    model.to(device)
    model = nn.DataParallel(model)
    model.train()

    optimizer = get_optimizer(p, model.parameters())

    with profiler:
        for i, batch in enumerate(dataloader):
            print(i)
            start_time_it = time()
            if i > 1:
                break
            input_batch = batch['img'].to(device)
            data_batch = batch['data'].to(device)
            data_aug_batch = batch['data_aug'].to(device)
            mask = batch['sal'].to(device)

            _, _, _ = model(input_batch, mask, data_batch, data_aug_batch)
            profiler.step()

    # create a wandb Artifact
    # profile_art = wandb.Artifact("trace", type="profile")
    # add the pt.trace.json files to the Artifact
    # profile_art.add_file(glob.glob(profile_dir + "*.pt.trace.json")[0])
    # log the artifact
    # profile_art.save()

    # wandb.finish()


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 4

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 2
        num_workers = 1

    main(config)
