import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv3 = GCNConv(hidden_channels, output_dim, add_self_loops=False)
        # self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, adj_sparse):
        edge_index, edge_weight = adj_sparse.indices(), adj_sparse.values()

        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        return x
