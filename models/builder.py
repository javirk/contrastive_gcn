import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from einops import rearrange
from time import time
import libs.utils as utils
from libs.losses import BalancedCrossEntropyLoss


class SegGCN(nn.Module):
    def __init__(self, p, backbone, graph_network, debug=True):
        super(SegGCN, self).__init__()
        self.debug = debug
        self.K = p['seggcn_kwargs']['K']
        self.T = p['seggcn_kwargs']['T']

        self.backbone = backbone
        self.graph = graph_network

        self.dim = p['gcn_kwargs']['output_dim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.bce = BalancedCrossEntropyLoss(size_average=True)

        self._load_pretrained(p)

    def _load_pretrained(self, p):
        device = next(self.backbone.parameters()).device
        if p['pretrained_model'] != 'None':
            state_dict = torch.load('ckpt/' + p['pretrained_model'], map_location=device)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            new_state = {}
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.rsplit('module.')[1]
                new_state[k] = v
            msg = self.load_state_dict(new_state, strict=False)
            print(msg)
        else:
            print('No pretrained weights have been loaded')
        return

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, *args):
        if self.training:
            return self.forward_train(*args)
        else:
            return self.forward_val(*args)

    def forward_train(self, img, mask, data, data_aug):
        """
        :param img:
        :param mask: Can be CAM or Saliency. BxCxHxW
        :param data:
        :param data_aug:
        :return:
        """
        dict_return = {}
        bs, c, h, w = mask.size()
        keep_indices_aug = torch.where(data_aug.keep_nodes)[0]

        out_dict = self.backbone(img)

        seg = out_dict['seg']  # This has to be compared to the CAM. B x C x H x W
        embeddings = out_dict['embeddings']  # B x dim x H x W
        batch_info = data['batch']
        # dim = embeddings.shape[1]

        cam_loss = self.bce(seg, mask)

        embeddings = rearrange(embeddings, 'b d h w -> (b h w) d')
        mask_twodim = rearrange(mask, 'b c h w -> (b h w) c')

        # The embeddings have to be averaged in the superpixel region.
        # We don't care about the numbers themselves, only that they are distinct between samples in batch.
        # Then we map them to be continuous
        sp_seg_bs = data.sp_seg.view(bs, -1).max(1).values
        sp_seg_bs = sp_seg_bs.roll(1)
        sp_seg_bs[0] = 0

        offset_mask = torch.arange(0, bs, device=data.sp_seg.device)
        # offset_mask = (data.sp_seg.max() + 1) * offset_mask
        # sp_seg = data.sp_seg + offset_mask.view(-1, 1, 1)
        # sp_seg = sp_seg.view(-1)
        # map_ = {j.item(): i for i, j in enumerate(sp_seg.unique())}
        # seg_mapped = torch.tensor([map_[x.item()] for x in sp_seg], device=embeddings.device)

        offset_mask = torch.cumsum(sp_seg_bs, dim=0) + offset_mask

        sp_seg = data.sp_seg + offset_mask.view(-1, 1, 1)
        sp_seg = sp_seg.view(-1)

        # print(f'First mapping: {time() - curr_time}')
        # curr_time = time()

        features_sp = scatter_mean(embeddings, sp_seg, dim=0)  # SP x dim (all the SP in the batch)
        data.x = features_sp
        data_aug.x = torch.index_select(features_sp, index=keep_indices_aug, dim=0)

        feat_ori = self.graph(data.x, data.edge_index)['features']
        feat_ori = nn.functional.normalize(feat_ori, dim=1)  # SP x dim_gcn

        with torch.no_grad():
            mask_sp = scatter_mean(mask_twodim, sp_seg, dim=0)  # SP x C
            mask_sp = mask_sp[:, 0]
            mask_sp = (mask_sp > 0.5).long()  # More pixels belong to the saliency than not

            offset = batch_info  # * 2
            mask_sp_offset = (mask_sp + offset) * mask_sp  # all bg's to 0
            mask_sp_offset = mask_sp_offset.view(-1)
            mask_indexes = torch.nonzero(mask_sp_offset).view(-1).squeeze()
            cam_sp_reduced = torch.index_select(mask_sp_offset, index=mask_indexes, dim=0) - 1

        with torch.no_grad():
            cam_sp_offset_aug = torch.index_select(mask_sp_offset, index=keep_indices_aug, dim=0)
            assert len(cam_sp_offset_aug.unique()) == bs + 1  # For stability
            out_aug = self.graph(data_aug.x, data_aug.edge_index, batch=cam_sp_offset_aug)
            prototypes_aug = out_aug['avg_pool'][1:]  # Because index 0 is background
            feat_aug = nn.functional.normalize(prototypes_aug, dim=1)  # B x dim_gcn

        q = torch.index_select(feat_ori, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, feat_aug.t())  # Unmasked SP x B
        negatives = self.queue.clone().detach()
        l_mem = torch.matmul(q, negatives)
        logits = torch.cat([l_batch, l_mem], dim=1)  # SP x (BS + Memory size)

        logits /= self.T

        self._dequeue_and_enqueue(feat_aug)

        if self.debug:
            q_var = torch.mean(torch.var(q, dim=0))
            aug_var = torch.mean(torch.var(feat_aug, dim=1))
            dict_return['q_var'] = q_var
            dict_return['aug_var'] = aug_var

        return logits, cam_sp_reduced, cam_loss, dict_return

    @torch.no_grad()
    def forward_val(self, img, data):
        bs = img.shape[0]

        out_dict = self.backbone(img)
        embeddings = out_dict['embeddings']  # B x dim x H x W
        pred_mask = out_dict['seg']

        embeddings = rearrange(embeddings, 'b d h w -> (b h w) d')

        # The embeddings have to be averaged in the superpixel region.
        # We don't care about the numbers themselves, only that they are distinct between samples in batch.
        # Then we map them to be continuous
        sp_seg_bs = data.sp_seg.view(bs, -1).max(1).values
        sp_seg_bs = sp_seg_bs.roll(1)
        sp_seg_bs[0] = 0

        offset_mask = torch.arange(0, bs, device=data.sp_seg.device)
        offset_mask = torch.cumsum(sp_seg_bs, dim=0) + offset_mask

        sp_seg = data.sp_seg + offset_mask.view(-1, 1, 1)
        sp_seg = sp_seg.view(-1)

        features_sp = scatter_mean(embeddings, sp_seg, dim=0)  # SP x dim (all the SP in the batch)

        data.x = features_sp

        features = self.graph(data.x, data.edge_index)['features']
        return features, pred_mask, sp_seg


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
