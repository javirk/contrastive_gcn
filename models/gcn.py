import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, output_dim)
        # self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch=None):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        features = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch is not None:
            avg = global_mean_pool(features, batch)  # [batch_size, hidden_channels]
        else:
            avg = None
        #
        # # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)

        return {'features': features, 'avg_pool': avg}
