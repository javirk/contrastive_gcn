import torch
from torch.nn import Conv2d, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from einops import rearrange


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.embedding_layer = Linear(num_features, hidden_channels, bias=False)
        self.conv1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv3 = GCNConv(hidden_channels, output_dim, add_self_loops=False)
        # self.lin = Linear(hidden_channels, dataset.num_classes)
        self.conv_sal = Conv2d(output_dim, 1, kernel_size=1)

    def forward(self, x, adj_sparse, batch_size=None, f_h=None, f_w=None):
        x = F.relu(self.embedding_layer(x))
        edge_index, edge_weight = adj_sparse.indices(), adj_sparse.values()

        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        # x = self.conv2(x, edge_index, edge_weight)
        # x = x.relu()
        features = self.conv3(x, edge_index, edge_weight)

        if batch_size is not None:
            x = features.relu()
            x = rearrange(x, '(b h w) d -> b d h w', b=batch_size, h=f_h, w=f_w)
            sal = self.conv_sal(x)
        else:
            sal = None

        return features, sal
