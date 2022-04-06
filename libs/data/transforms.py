import torch
import numpy as np
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.transforms import BaseTransform
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor


class EdgePerturbation(BaseTransform):
    def __init__(self, p_drop=0.5):
        self.p_drop = p_drop

    def __call__(self, data):
        edge_index, edge_attr = dropout_adj(data.edge_index, p=self.p_drop, training=True)
        num_removed = data.edge_index.shape[1] - edge_index.shape[1]
        edge_index = self.add_edges(edge_index, num_add=num_removed, total_nodes=len(data.x + 1))

        data.edge_index = edge_index

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dropping probability={self.p_drop})'

    def add_edges(self, edge_index, num_add, total_nodes):
        edges_index_add = torch.randint(0, total_nodes, size=[2, num_add])
        edge_index = torch.cat([edge_index, edges_index_add], dim=1)
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index


class NodeDropping(BaseTransform):
    def __init__(self, percentage_keep=0.9, return_nodes=True):
        self.percentage_keep = percentage_keep
        self.return_nodes = return_nodes

    def __call__(self, data):
        all_nodes = np.array(list(range(data.x.shape[0])))
        keep_indices = torch.tensor(np.random.choice(all_nodes, size=int(all_nodes[-1] * self.percentage_keep),
                                                   replace=False))

        edge_index, edge_attr = subgraph(keep_indices, data.edge_index, relabel_nodes=True)
        x = torch.index_select(data.x, index=keep_indices, dim=0)
        pos = torch.index_select(data.pos, index=keep_indices, dim=0)

        data.edge_index = edge_index
        data.x = x
        data.pos = pos
        if self.return_nodes:
            keep_nodes = torch.zeros(len(all_nodes))
            keep_nodes[keep_indices] = 1
            data.keep_nodes = keep_nodes#.unsqueeze(0)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class ToTensor(ToTensorV2):
    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToTensor, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply, "masks": self.apply_to_masks}

    def apply(self, img, **params):
        return to_tensor(img)

    def apply_to_masks(self, masks, **params):
        return [self.apply(mask, **params) for mask in masks]
