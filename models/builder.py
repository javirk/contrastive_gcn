import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from einops import rearrange
import libs.utils as utils
from libs.losses import BalancedCrossEntropyLoss


class SegGCN(nn.Module):
    def __init__(self, p, backbone, graph_network):
        super(SegGCN, self).__init__()
        self.K = p['seggcn_kwargs']['K']
        self.T = p['seggcn_kwargs']['T']

        self.backbone = backbone
        self.graph = graph_network
        if self.training:
            self.forward = self.forward_train
        else:
            self.forward = self.forward_inference

        self.dim = p['gcn_kwargs']['output_dim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.bce = BalancedCrossEntropyLoss(size_average=True)

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

    def forward_train(self, img, cam, data, data_aug):
        """
        :param img:
        :param cam: BxCxHxW
        :param data:
        :param data_aug:
        :return:
        """
        # batch_size = img.shape[0]
        bs, c, h, w = cam.size()
        keep_indices_aug = torch.where(data_aug.keep_nodes)[0]

        out_dict = self.backbone(img)
        seg = out_dict['seg']  # This has to be compared to the CAM. B x C x H x W
        embeddings = out_dict['embeddings']  # B x dim x H x W
        batch_info = data['batch']
        # dim = embeddings.shape[1]

        cam_loss = self.bce(seg, cam)

        embeddings = rearrange(embeddings, 'b d h w -> (b h w) d')
        cam = rearrange(cam, 'b c h w -> (b h w) c')

        # We don't care about the numbers themselves, only that they are distinct between batches. Then we map them to be continuous
        offset_mask = torch.arange(0, bs, device=data.sp_seg.device)
        offset_mask = (data.sp_seg.max() + 1) * offset_mask
        sp_seg = data.sp_seg + offset_mask.view(-1, 1, 1)
        sp_seg = sp_seg.view(-1)

        map = {j.item(): i for i, j in enumerate(sp_seg.unique())}
        seg_mapped = torch.tensor([map[x.item()] for x in sp_seg], device=embeddings.device)

        features_sp = scatter_mean(embeddings, seg_mapped, dim=0)  # SP x dim (all the SP in the batch)
        data.x = features_sp
        data_aug.x = torch.index_select(features_sp, index=keep_indices_aug, dim=0)

        feat_ori = self.graph(data.x, data.edge_index)['features']
        feat_ori = nn.functional.normalize(feat_ori, dim=1)  # SP x dim_gcn

        with torch.no_grad():
            cam_sp = scatter_mean(cam, seg_mapped, dim=0)  # SP x C
            if c == 2:
                # This only works for MNIST (two classes). For more classes the next line alone (after if) should work
                cam_sp = cam_sp[:, 1]
            cam_sp = (cam_sp > 0.5).long()

            offset = batch_info  # * 2
            cam_sp_offset = (cam_sp + offset) * cam_sp  # all bg's to 0
            cam_sp_offset = cam_sp_offset.view(-1)
            mask_indexes = torch.nonzero(cam_sp_offset).view(-1).squeeze()
            cam_sp_reduced = torch.index_select(cam_sp_offset, index=mask_indexes, dim=0) - 1

        with torch.no_grad():
            cam_sp_offset_aug = torch.index_select(cam_sp_offset, index=keep_indices_aug, dim=0)
            out_aug = self.graph(data_aug.x, data_aug.edge_index, batch=cam_sp_offset_aug)
            prototypes_aug = out_aug['avg_pool'][1:]  # Because index 0 is background
            feat_aug = nn.functional.normalize(prototypes_aug, dim=1)  # B x dim_gcn
            # prototypes = cam_sp2

        q = torch.index_select(feat_ori, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, feat_aug.t())
        negatives = self.queue.clone().detach()
        l_mem = torch.matmul(q, negatives)
        logits = torch.cat([l_batch, l_mem], dim=1)

        logits /= self.T
        self._dequeue_and_enqueue(feat_aug)

        return logits, cam_sp_reduced, cam_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
