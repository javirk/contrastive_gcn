# Code referenced from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ContrastiveDeeplab(nn.Module):
    def __init__(self, p, backbone, decoder, upsample, use_classification_head=False):
        super(ContrastiveDeeplab, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head

        if p['head']['type'] == 'linear':
            # Head is linear.
            # We can just use regular decoder since final conv is 1 x 1.
            self.head = decoder[-1]
            decoder[-1] = nn.Identity()
            self.decoder = decoder
        else:
            raise NotImplementedError('Head {} is currently not supported'.format(head))

        if self.use_classification_head: # Add classification head for saliency prediction
            self.classification_head = nn.Conv2d(self.head.in_channels, 1, 1, bias=False)

    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)

        # Upsample to input resolution
        if self.upsample:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)

        # Return outputs
        if self.use_classification_head:
            return {'embeddings': x, 'seg': sal}
        else:
            return {'embeddings': x}

if __name__ == '__main__':
    import torchvision.models.resnet as resnet
    from models.modules.resnet_dilated import ResnetDilated

    config = {'head': {'type': 'linear'}}

    backbone = resnet.__dict__['resnet18'](pretrained=False)
    backbone_channels = 512
    backbone = ResnetDilated(backbone)

    nc = 64
    head = DeepLabHead(backbone_channels, nc)
    c = ContrastiveDeeplab(config, backbone, head, True, True)
    im = torch.rand((2, 3, 512, 512))
    x, sal = c(im)