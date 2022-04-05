import torch
from torch.nn.functional import cross_entropy
import libs.utils as utils


def train(p, train_loader, model, optimizer, epoch, device):
    losses = utils.AverageMeter('Loss', ':.4e')
    contrastive_losses = utils.AverageMeter('Contrastive', ':.4e')
    cam_losses = utils.AverageMeter('Cam_loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.ProgressMeter(len(train_loader), [losses, contrastive_losses, cam_losses, top1],
                                   prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        input_batch = batch['img'].to(device)
        data_batch = batch['data'].to(device)
        data_aug_batch = batch['data_aug'].to(device)

        optimizer.zero_grad()
        cam = utils.get_cam_segmentation(input_batch)

        logits, labels, cam_loss = model(input_batch, cam, data_batch, data_aug_batch)

        # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).to(device)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = cross_entropy(logits, labels, weight=w_class, reduction='mean')

        # Calculate total loss and update meters
        loss = contrastive_loss + cam_loss
        contrastive_losses.update(contrastive_loss.item())
        cam_losses.update(cam_loss.item())
        losses.update(loss.item())

        loss.backward()
        optimizer.step()

        # Here are the metrics
        acc1 = accuracy(logits, labels, topk=(1,))
        top1.update(acc1[0], input_batch.size(0))

        # Display progress
        if i % p['logs']['writing_freq'] == 0 and p['ubelix']:
            step_logging = epoch * len(train_loader) + i
            progress.to_wandb(step_logging, prefix='train')
            progress.reset()
            # progress.display(i)


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
