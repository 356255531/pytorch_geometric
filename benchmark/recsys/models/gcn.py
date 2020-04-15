import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, hidden_size, repr_dim, dropout):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.conv1 = GCNConv(emb_dim, hidden_size)
        self.conv2 = GCNConv(hidden_size, repr_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
