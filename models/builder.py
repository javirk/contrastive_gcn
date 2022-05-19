import copy
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from einops import rearrange
import libs.utils as utils
from libs.losses import BalancedCrossEntropyLoss


class SegGCN(nn.Module):
    def __init__(self, p, encoder, graph_network, debug=True):
        super(SegGCN, self).__init__()
        self.debug = debug
        self.K = p['seggcn_kwargs']['K']
        self.T = p['seggcn_kwargs']['T']
        self.m = p['seggcn_kwargs']['m']

        self.encoder = encoder
        self.graph = graph_network
        self.graph_k = copy.deepcopy(graph_network)

        for param_q, param_k in zip(self.graph.parameters(), self.graph_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.dim = p['gcn_kwargs']['output_dim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.bce = BalancedCrossEntropyLoss(size_average=True)

        self._load_pretrained(p)

    def _load_pretrained(self, p):
        device = next(self.encoder.parameters()).device
        if p['pretrained_backbone'] != 'None':
            state_dict = torch.load('ckpt/' + p['pretrained_backbone'], map_location=device)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            new_state = {}
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.rsplit('module.')[1]
                new_state[k] = v
            msg = self.encoder.load_state_dict(new_state, strict=False)
            print('Backbone: ', msg)
        else:
            print('No pretrained weights have been loaded for the backbone')

        if p['pretrained_gcn'] != 'None':
            state_dict = torch.load('ckpt/' + p['pretrained_gcn'], map_location=device)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            new_state = {}
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.rsplit('module.')[1]
                new_state[k] = v
            msg = self.graph.load_state_dict(new_state, strict=False)
            print('Graph: ', msg)
        else:
            print('No pretrained weights have been loaded for the graph')
        return

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.graph.parameters(), self.graph_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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

    def forward_train(self, img, mask, graph_transforms):
        """
        :param img:
        :param mask:
        :param graph_transforms:
        :return:
        """
        radius = 4
        dict_return = {}
        bs, c, h, w = img.size()
        mask = mask.squeeze(dim=1)

        # Maybe the first part has to be taken from memory. Check if it takes a lot ot time
        with torch.no_grad():
            out_dict = self.encoder(img, radius=radius)
            features, aff_mat = out_dict['features'], out_dict['aff']
            f_h, f_w = features.shape[-2], features.shape[-1]

            aff_mat = torch.pow(aff_mat, 1)  # This probably doesn't do anything
            # mask = mask.sigmoid()
            # mask = nn.functional.interpolate(mask, size=(f_h, f_w))
            # mask = (mask > 0.5).int().squeeze(1)  # B x f_h x f_w

            aff_mat = utils.generate_aff(f_h, f_w, aff_mat, radius=radius)  # B x f_h.f_w x f_h.f_w

            features = rearrange(features, 'b c h w -> b (h w) c')

            # THE AUGMENTATIONS NOW
            aff_mat_aug = graph_transforms(aff_mat.clone())

            # Unbatched affinity matrix
            # aff_mat = torch.block_diag(*aff_mat)
            # aff_mat_aug = torch.block_diag(*aff_mat_aug)
            aff_mat = aff_mat.to(features.device)
            aff_mat_aug = aff_mat_aug.to(features.device)

        # Prepare the mask and logits (in mask_ori)
        with torch.no_grad():
            offset = torch.arange(0, 2 * bs, 2).to(mask.device)
            mask_ori = (mask + torch.reshape(offset, [-1, 1, 1])) * mask  # all bg's to 0
            mask_ori = mask_ori.view(-1)
            mask_indexes = torch.nonzero(mask_ori).view(-1).squeeze()
            mask_ori = torch.div(torch.index_select(mask_ori, index=mask_indexes, dim=0), 2, rounding_mode='floor')
            mask_ori = mask_ori.long()

        # Run the main features through the GNN
        # adj = aff_mat.to_sparse().to(features.device)
        feat_ori, sal = self.graph(features, aff_mat, batch_size=bs, f_h=f_h, f_w=f_w)  # B x H.W x dim
        feat_ori = rearrange(feat_ori, 'b hw c -> (b hw) c')  # B.H.W x dim
        feat_ori = nn.functional.normalize(feat_ori, dim=-1)

        # Run the augmented features through the GNN
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            # adj_aug = aff_mat_aug.to_sparse().to(features.device)
            feat_aug, _ = self.graph_k(features, aff_mat_aug)  # B x H.W x dim
            feat_aug = rearrange(feat_aug, 'b hw c -> b c hw')  # B x dim x H.W

            mask_k = mask.reshape(bs, -1, 1).float()  # B x H.W x 1
            prototypes_foreground = torch.bmm(feat_aug, mask_k).squeeze(-1)  # B x dim
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)

        # Compute the logits
        q = torch.index_select(feat_ori, index=mask_indexes, dim=0)  # pixel x dim
        l_batch = torch.matmul(q, prototypes.t())  # pixel x B
        negatives = self.queue.clone().detach()  # dim x negatives
        l_mem = torch.matmul(q, negatives)  # pixels x negatives
        logits = torch.cat([l_batch, l_mem], dim=1)  # pixels x (B + negatives)

        logits /= self.T

        self._dequeue_and_enqueue(prototypes)

        if self.debug:
            q_var = torch.mean(torch.var(q, dim=0))
            aug_var = torch.mean(torch.var(prototypes, dim=1))
            dict_return['q_var'] = q_var
            dict_return['aug_var'] = aug_var

        return logits, mask_ori, sal, dict_return

    @torch.no_grad()
    def forward_val(self, img, apply_transforms=None):
        radius = 4

        out_dict = self.encoder(img)
        features, aff_mat, mask = out_dict['features'], out_dict['aff'], out_dict['seg']
        f_h, f_w = features.shape[-2], features.shape[-1]
        features = rearrange(features, 'b c h w -> b (h w) c')

        aff_mat = utils.generate_aff(f_h, f_w, aff_mat, radius=radius)  # B x f_h.f_w x f_h.f_w
        if apply_transforms is not None:
            aff_mat = apply_transforms(aff_mat)
        aff_mat = aff_mat.to(features.device)

        features, sal = self.graph(features, aff_mat, batch_size=features.shape[0], f_h=f_h, f_w=f_w)  # B x H.W x dim
        features = nn.functional.normalize(features, dim=-1)

        return features, mask


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
