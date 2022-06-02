import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
import libs.utils as utils


def forward_seg(p, loader, model, crit_bce, graph_tr, optimizer, epoch, device, phase='train', last_it=None):
    losses = utils.AverageMeter('Loss', ':.4e')
    contrastive_losses = utils.AverageMeter('Contrastive', ':.4e')
    cam_losses = utils.AverageMeter('Cam_loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    progress_vars = [losses, contrastive_losses, cam_losses, top1]

    if model.module.debug:
        q_var = utils.AverageMeter('Q_Var', '.4e')
        aug_var = utils.AverageMeter('Aug_Var', '.4e')
        progress_vars.extend([q_var, aug_var])

    progress = utils.ProgressMeter(len(loader), progress_vars, prefix="Epoch: [{}]".format(epoch))
    if phase == 'train':
        last_it = -1
        model.train()
    else:
        assert last_it is not None
        model.eval()

    for i, batch in enumerate(loader):
        input_batch = batch['img'].to(device)
        saliency = batch['sal_down'].to(device)  # Just the saliency (Bx1xHxW)
        # saliency = nn.functional.interpolate(saliency, size=(saliency.shape[-2] // 8, saliency.shape[-1] // 8))

        optimizer.zero_grad()
        # cam = utils.get_cam_segmentation(input_batch)

        with torch.set_grad_enabled(phase == 'train'):
            logits, labels, pred_sal, other_res = model(input_batch, saliency, graph_tr, radius=p['radius'],
                                                        training_branch=True)

        sal_loss = crit_bce(pred_sal, saliency)

        # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).to(device)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = cross_entropy(logits, labels, weight=w_class, reduction='mean')

        # Calculate total loss and update meters
        loss = p['train_kwargs']['lambda_contrastive'] * contrastive_loss + sal_loss
        contrastive_losses.update(contrastive_loss.item())
        cam_losses.update(sal_loss.item())
        losses.update(loss.item())
        if model.module.debug:
            q_var.update(other_res['q_var'])
            aug_var.update(other_res['aug_var'])

        if phase == 'train':
            loss.backward()
            optimizer.step()

        # Here are the metrics
        acc1 = accuracy(logits, labels, topk=(1,))
        top1.update(acc1[0], input_batch.size(0))

        # Display progress
        if (i + 1) % p['logs']['writing_freq'] == 0 and p['ubelix'] and phase == 'train':
            step_logging = epoch * len(loader) + i
            progress.to_wandb(step_logging, prefix='train')
            progress.reset()

        last_it = epoch * len(loader) + i if phase == 'train' else last_it

    # Display progress
    if p['ubelix'] and phase == 'val':
        progress.to_wandb(last_it, prefix=phase)
        progress.reset()

    return last_it


def forward_aff(p, loader, model, crit_aff, crit_bce, optimizer, epoch, device, phase='train', last_it=None):
    radius = p['radius']
    losses_meter = utils.AverageMeter('Loss', ':.4e')
    ce_loss_meter = utils.AverageMeter('CE_loss', ':.4e')
    aff_loss_meter = utils.AverageMeter('Affinity regularized', ':.4e')
    cam_losses_meter = utils.AverageMeter('Cam_loss', ':.4e')
    progress_vars = [losses_meter, ce_loss_meter, aff_loss_meter, cam_losses_meter]

    progress = utils.ProgressMeter(len(loader), progress_vars, prefix="Epoch: [{}]".format(epoch))
    if phase == 'train':
        last_it = -1
        model.train()
    else:
        assert last_it is not None
        model.eval()

    for i, batch in enumerate(loader):
        input_batch = batch['img'].to(device)
        saliency = batch['sal'].to(device)  # Just the saliency (Bx1xHxW)
        label = batch['sal_down'].to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            output_dict = model(input_batch, radius=radius)

        aff = output_dict['aff'].unsqueeze(dim=1)
        label = utils.unfold(label, radius=radius)
        N, _, H, W = output_dict['features'].shape

        sal_loss = crit_bce(output_dict['seg'], saliency)

        crf_img = F.interpolate(input_batch, [H, W], mode='bilinear', align_corners=False)
        crf_input = {'rgb': crf_img}

        label_center = label[:, :, radius, radius, :, :].view(N, 1, 1, 1, H, W)
        label_center = label_center.expand_as(label)

        aff_label = torch.eq(label, label_center)
        aff_label_ignore = torch.eq(label, 255)

        aff_label[aff_label_ignore == 1] = 255
        aff_label[label_center == 255] = 255

        bg_pos_label_need = torch.zeros_like(aff_label)
        bg_pos_label_need[aff_label == 1] = 1
        bg_pos_label_need[label_center != 0] = 0
        bg_pos_label_need[aff_label_ignore == 1] = 0

        fg_pos_label_need = torch.zeros_like(aff_label)
        fg_pos_label_need[aff_label == 1] = 1
        fg_pos_label_need[label_center == 0] = 0
        fg_pos_label_need[aff_label_ignore == 1] = 0
        fg_pos_label_need[label_center == 255] = 0

        neg_label_need = torch.zeros_like(aff_label)
        neg_label_need[aff_label == 0] = 1
        neg_label_need[aff_label_ignore == 1] = 0
        neg_label_need[label_center == 255] = 0

        bg_count = torch.sum(bg_pos_label_need).float() + 1e-5
        fg_count = torch.sum(fg_pos_label_need).float() + 1e-5
        neg_count = torch.sum(neg_label_need).float() + 1e-5

        bg_loss = torch.sum(- bg_pos_label_need.float() * torch.log(aff + 1e-5)) / bg_count
        fg_loss = torch.sum(- fg_pos_label_need.float() * torch.log(aff + 1e-5)) / fg_count
        neg_loss = torch.sum(- neg_label_need.float() * torch.log(1. + 1e-5 - aff)) / neg_count

        # Criterion is Gated CRF
        out_gatedcrf = crit_aff(output_dict['aff_crf'], [{'weight': 1, 'xy': 6, 'rgb': 0.1}], radius, crf_input, H, W)
        crf_loss = 3 * out_gatedcrf['loss']
        ce_loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2

        loss = ce_loss + crf_loss + sal_loss

        losses_meter.update(loss.item())
        aff_loss_meter.update(crf_loss.item())
        ce_loss_meter.update(ce_loss.item())
        cam_losses_meter.update(sal_loss.item())

        if phase == 'train':
            loss.backward()
            optimizer.step()

        # Display progress
        if (i + 1) % p['logs']['writing_freq'] == 0 and p['ubelix'] and phase == 'train':
            step_logging = epoch * len(loader) + i
            progress.to_wandb(step_logging, prefix=phase)
            progress.reset()

        last_it = epoch * len(loader) + i if phase == 'train' else last_it

    # Display progress
    if p['ubelix'] and phase == 'val':
        progress.to_wandb(last_it, prefix=phase)
        progress.reset()

    return last_it

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_segmentation_vanilla(p, train_loader, model, criterion, optimizer, epoch, device):
    """ Train a segmentation model in a fully-supervised manner """
    losses = utils.AverageMeter('Loss', ':.4e')
    semseg_meter = utils.SemsegMeter(p['num_classes'], train_loader.dataset.get_class_names(), has_bg=False,
                                     ignore_index=255)
    progress = utils.ProgressMeter(len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['img'].to(device)
        targets = batch['semseg'].to(device)

        output = model(images)
        loss = criterion(output, targets)
        losses.update(loss.item())
        semseg_meter.update(torch.argmax(output, dim=1), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % p['logs']['writing_freq'] == 0 and p['ubelix']:
            step_logging = epoch * len(train_loader) + i
            progress.to_wandb(step_logging, prefix='train')

    eval_results = semseg_meter.return_score(verbose=True)
    return eval_results