import argparse
import os
from datetime import datetime
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import libs.utils as utils
from models.gcn import AGNN
from models.simple_segmentation import SimpleSegmentation
from libs.common_config import get_optimizer, get_sal_transforms, get_joint_transforms, adjust_learning_rate, \
    get_dataset, get_train_transforms, get_val_transforms, get_segmentation_model
from libs.train_utils import train_segmentation_vanilla
from libs.finetune_utils import save_results_to_disk, eval_segmentation_supervised_online, \
    eval_segmentation_supervised_offline

# Parser
parser = argparse.ArgumentParser(description='Fully-supervised segmentation - Finetune linear layer')
parser.add_argument('-sc', '--segmentation-config',
                    type=str,
                    help='Config file for the environment')

parser.add_argument('-ac', '--affinity-config',
                    type=str,
                    help='Config file for the environment')

parser.add_argument('-c', '--config',
                    default='configs/configs-default_lc.yml',
                    type=str,
                    help='Config file for the environment')

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
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p['checkpoint'] = f'./ckpt/{current_time}_lc.pth'
    p['best_model'] = f'./ckpt/{current_time}_bestmodel.pth'
    p['save_dir'] = 'results/linear_finetune/'
    utils.dict_to_file(p, f'runs/{current_time}.yml')
    # utils.copy_file(FLAGS.config, f'runs/{current_time}.yml')  # This should be improved in the future maybe

    if p['ubelix'] == 1:
        wandb.init(project='Contrastive-Graphs', config=p, name=current_time + '_lc',
                   notes=f"{p['dataset']} - {p['backbone']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get model
    encoder = get_segmentation_model(p, device)
    gcn = AGNN(num_features=p['gcn_kwargs']['ndim'], hidden_channels=p['gcn_kwargs']['hidden_channels'],
               output_dim=p['gcn_kwargs']['output_dim'])
    model = SimpleSegmentation(p, encoder, gcn)
    model = nn.DataParallel(model)
    model.train()
    model.module.freeze_all()
    model.to(device)

    # Get criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Optimizer
    parameters = list(filter(lambda x: x.requires_grad, model.parameters()))
    assert len(parameters) == 2  # linear_classifier.weight, linear_classifier.bias
    optimizer = get_optimizer(p, parameters)

    # Dataset

    sal_tf = get_sal_transforms(p)
    image_train_tf = get_train_transforms(p)
    image_val_tf = get_val_transforms(p)
    joint_tf = get_joint_transforms(p)

    train_dataset = get_dataset(p, root='data/', image_set='trainaug', transform=image_train_tf, sal_transform=sal_tf,
                                joint_transform=joint_tf)
    val_dataset = get_dataset(p, root='data/', image_set='val', transform=image_val_tf, sal_transform=sal_tf,
                              joint_transform=joint_tf)
    # True validation dataset without reshape - For validation.
    true_val_dataset = get_dataset(p, root='data/', image_set='val', transform=None, sal_transform=sal_tf,
                                   joint_transform=joint_tf)
    train_loader = DataLoader(train_dataset, batch_size=p['train_kwargs']['batch_size'], shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=p['val_kwargs']['batch_size'], shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)

    print('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)))

    # Resume from checkpoint
    # if os.path.exists(p['checkpoint']):
    #     print(('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
    #     checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     model.load_state_dict(checkpoint['model'])
    #     # model.cuda()
    #     start_epoch = checkpoint['epoch']
    #     best_epoch = checkpoint['best_epoch']
    #     best_iou = checkpoint['best_iou']
    #
    # else:
    print('No checkpoint file at {}'.format(p['checkpoint']))
    start_epoch = 0
    best_epoch = 0
    best_iou = -1

    for epoch in range(start_epoch, p['epochs']):
        print('Epoch %d/%d' % (epoch + 1, p['epochs']))
        print('-' * 10)

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        eval_train = train_segmentation_vanilla(p, train_loader, model, criterion, optimizer, epoch, device)
        step = len(train_loader) * (epoch + 1)
        # Evaluate online -> This will use batched eval where every image is resized to the same resolution.
        print('Evaluate ...')
        eval_val = eval_segmentation_supervised_online(p, val_loader, model, step, device)

        if eval_val['mIoU'] > best_iou:
            print('Found new best model: %.2f -> %.2f (mIoU)' % (100 * best_iou, 100 * eval_val['mIoU']))
            best_iou = eval_val['mIoU']
            best_epoch = epoch
            torch.save(model.state_dict(), p['best_model'])
        else:
            print('No new best model: %.2f -> %.2f (mIoU)' % (100 * best_iou, 100 * eval_val['mIoU']))
            print('Last best model was found in epoch %d' % best_epoch)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_iou': best_iou},
                   p['checkpoint'])

    # Evaluate best model at the end -> This will evaluate the predictions on the original resolution.
    print('Evaluating best model at the end')
    model.load_state_dict(torch.load(p['best_model']))
    save_results_to_disk(p, val_loader, model, device, crf_postprocess=FLAGS.crf_postprocessing)
    eval_stats = eval_segmentation_supervised_offline(p, true_val_dataset, verbose=True)


if __name__ == "__main__":
    config_aff = utils.read_config(FLAGS.affinity_config)
    config_seg = utils.read_config(FLAGS.segmentation_config)
    config = utils.read_config(FLAGS.config)
    config['ubelix'] = FLAGS.ubelix

    num_workers = 8

    if FLAGS.ubelix == 0:
        config['train_kwargs']['batch_size'] = 4
        config['val_kwargs']['batch_size'] = 4
        num_workers = 0

    if 'runs' in FLAGS.affinity_config:
        date_run = FLAGS.affinity_config.split('/')[-1].split('.')[-2]
        config['pretrained_backbone'] = date_run + '_aff.pth'

    if 'runs' in FLAGS.segmentation_config:
        date_run = FLAGS.segmentation_config.split('/')[-1].split('.')[-2]
        config['pretrained_gcn'] = date_run + '_graph.pth'

    print(config)
    main(config)
