# Code referenced from https://github.com/facebookresearch/astmt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, labels, void_pixels=None):
        assert (output.size() == labels.size())

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(labels.size()))
        elif self.batch_average:
            final_loss /= labels.size()[0]

        return final_loss


class ModelLossSemsegGatedCRF(torch.nn.Module):
    """
       This module is based on the Gated CRF Loss for Weakly Supervised Semantic Image Segmentation.
       This loss function promotes consistent label assignment guided by input features, such as RGBXY.
       The citation:
       @article{obukhov2019gated,
           author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
           title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
           journal={CoRR},
           volume={abs/1906.04651},
           year={2019},
           url={http://arxiv.org/abs/1906.04651},
       }
       """

    def forward(
            self, aff_dis_radius, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        if mask_src is None:
            mask_src = torch.ones_like(aff_dis_radius).unsqueeze(dim=1)
        N, _, _, height_pred, width_pred = aff_dis_radius.shape
        device = aff_dis_radius.device

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        denom = torch.sum(mask_src)
        loss = kernels.squeeze(dim=1) * aff_dis_radius * mask_src.squeeze(dim=1)

        loss = torch.sum(loss)

        out = {
            'loss': loss / denom,
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = ModelLossSemsegGatedCRF._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                    feature = ModelLossSemsegGatedCRF._downsample(
                        feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * ModelLossSemsegGatedCRF._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = ModelLossSemsegGatedCRF._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred - vis.shape[3], 0, height_pred - vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis
