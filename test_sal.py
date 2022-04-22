import torch
import torch.nn as nn
import torchvision.utils
from torch_geometric.loader import DataLoader
import wandb
import argparse
from datetime import datetime
from libs.losses import BalancedCrossEntropyLoss
from models.backbones.unet import UNet
import libs.utils as utils
from libs.common_config import get_optimizer, adjust_learning_rate, get_dataset, get_image_transforms

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
    class_labels = {0: 'bg', 1: 'sal'}
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}_sal.pth'
    utils.copy_file(FLAGS.config, f'runs/{current_time}.yml')  # This should be improved in the future maybe

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time + '_sal',
                   notes=f"{p['dataset']} - {p['backbone']}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # aug_tf = get_augmentation_transforms(p)
    image_tf = get_image_transforms(p)
    dataset = get_dataset(p, root='data/', image_set='train', transform=image_tf)
    dataloader = DataLoader(dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=True)

    backbone = UNet(p, n_channels=3)

    backbone.to(device)
    backbone = nn.DataParallel(backbone)
    backbone.train()

    optimizer = get_optimizer(p, backbone.parameters())
    bce = BalancedCrossEntropyLoss(size_average=True)

    for epoch in range(p['epochs']):
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        losses = utils.AverageMeter('Cam_loss', ':.4e')
        progress = utils.ProgressMeter(len(dataloader), [losses], prefix="Epoch: [{}]".format(epoch))

        for i, batch in enumerate(dataloader):
            input_batch = batch['img'].to(device)
            mask = batch['sal'].to(device)

            optimizer.zero_grad()

            out_dict = backbone(input_batch)
            seg = out_dict['seg']  # This has to be compared to the CAM. B x C x H x W

            loss = bce(seg, mask)
            losses.update(loss.item())

            loss.backward()
            optimizer.step()

            # Display progress
            if i % p['logs']['writing_freq'] == 0 and p['ubelix']:
                step_logging = epoch * len(dataloader) + i
                progress.to_wandb(step_logging, prefix='train')
                progress.reset()

                grid_sal_pred = torchvision.utils.make_grid(seg[:9], nrow=3)[0].cpu().numpy()
                grid_sal_gt = torchvision.utils.make_grid(mask[:9], nrow=3)[0].cpu().numpy()
                grid_images = torchvision.utils.make_grid(input_batch[:9], nrow=3).cpu()
                grid_images = grid_images.permute((1, 2, 0)).numpy()
                im_wandb = wandb.Image(grid_images, masks={
                    'predictions': {
                        "mask_data": (grid_sal_pred > 0.5).astype(int),
                        'class_labels': class_labels
                    },
                    'ground_truth': {
                        "mask_data": grid_sal_gt,
                        'class_labels': class_labels
                    }})
                wandb.log({'images': im_wandb})

        torch.save({'optimizer': optimizer.state_dict(), 'model': backbone.state_dict(),
                    'epoch': epoch + 1}, p['checkpoint'])

    wandb.finish()


if __name__ == '__main__':
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 2
        num_workers = 1

    config['logs']['writing_freq'] = 15

    main(config)
