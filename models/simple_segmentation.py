import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import libs.utils as utils


class SimpleSegmentation(nn.Module):
    def __init__(self, p, encoder, graph_network):
        super(SimpleSegmentation, self).__init__()
        self.encoder = encoder
        self.decoder = graph_network
        self.linear_classifier = nn.Conv2d(p['gcn_kwargs']['output_dim'], p['num_classes'], kernel_size=1)

        self._load_pretrained(p)

    def _load_pretrained(self, p):
        device = next(self.encoder.parameters()).device
        if p['pretrained_backbone'] != 'None':
            state_dict = torch.load('ckpt/' + p['pretrained_backbone'], map_location=device)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            new_state = utils.remove_module(state_dict)

            msg = self.encoder.load_state_dict(new_state, strict=False)
            print('Backbone: ', msg, f' from {p["pretrained_backbone"]}')
        else:
            print('No pretrained weights have been loaded for the backbone')

        if p['pretrained_gcn'] != 'None':
            state_dict = torch.load('ckpt/' + p['pretrained_gcn'], map_location=device)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            new_state = utils.remove_module(state_dict)

            msg = self.decoder.load_state_dict(new_state, strict=False)
            print('Graph: ', msg, f' from {p["pretrained_gcn"]}')
        else:
            print('No pretrained weights have been loaded for the graph')
        return


    def forward(self, x, radius=4):
        input_shape = x.shape[-2:]
        x_dict = self.encoder(x, radius)
        features, aff = x_dict['features'], x_dict['aff']

        f_h, f_w = features.shape[-2], features.shape[-1]
        features = rearrange(features, 'b c h w -> b (h w) c')

        aff_mat = utils.generate_aff(f_h, f_w, aff, radius=radius)  # B x f_h.f_w x f_h.f_w
        aff_mat = aff_mat.to(features.device)

        features, _ = self.decoder(features, aff_mat)  # B x H.W x dim
        x = rearrange(features, 'b (h w) c -> b c h w', h=f_h, w=f_w)

        x = self.linear_classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    def freeze_all(self):
        """
        Freeze all layers except last 1x1 convolution
        :return:
        """
        for name, param in self.named_parameters():
            if name not in ['linear_classifier.weight', 'linear_classifier.bias']:
                param.requires_grad = False