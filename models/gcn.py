import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from einops import rearrange

class GraphAttentionLayer(nn.Module):
    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            # uniform initialization
            self.beta = nn.parameter.Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=requires_grad)
        else:
            self.beta = torch.autograd.Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj):
        norm2 = torch.norm(x, p=2, dim=-1).unsqueeze(-1)
        cos = torch.div(torch.bmm(x, x.transpose(-1, -2)), torch.bmm(norm2, norm2.transpose(-1, -2)) + 1e-7)

        mask = torch.zeros_like(adj)
        # mask[aff_cropping == 0] = -1e9
        mask[cos < 0] = -1e9
        cos = self.beta * cos
        masked = cos + mask + 10 * adj

        # propagation matrix
        P = F.softmax(masked, dim=2)

        # attention-guided propagation
        output = torch.bmm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.embedding_layer = nn.Linear(num_features, hidden_channels, bias=False)
        self.conv1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv3 = GCNConv(hidden_channels, output_dim, add_self_loops=False)

        self.conv_sal = nn.Conv2d(output_dim, 1, kernel_size=1)

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

class AGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim):
        super(AGNN, self).__init__()
        self.layers = 3
        self.embedding_layer = nn.Linear(num_features, hidden_channels, bias=False)

        self.attentionlayers = nn.ModuleList()
        self.attentionlayers.append(GraphAttentionLayer(requires_grad=True))
        for i in range(1, self.layers):
            self.attentionlayers.append(GraphAttentionLayer())

        self.outputlayer = nn.Linear(hidden_channels, output_dim, bias=False)
        nn.init.xavier_uniform_(self.outputlayer.weight)

        self.conv_sal = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x, adj, batch_size=None, f_h=None, f_w=None):
        x = F.relu(self.embedding_layer(x))

        for i in range(self.layers):
            x = self.attentionlayers[i](x, adj)
        features = x.clone()

        embeddings = self.outputlayer(x)

        if batch_size is not None:
            x = features.relu()
            x = rearrange(x, 'b (h w) d -> b d h w', b=batch_size, h=f_h, w=f_w)
            sal = self.conv_sal(x)
        else:
            sal = None

        return embeddings, sal