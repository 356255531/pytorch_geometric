import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, num_nodes, num_heads, emb_dim, hidden_size, repr_dim, if_use_features, dropout):
        super(GAT, self).__init__()
        self.num_nodes = num_nodes
        self.if_use_features = if_use_features
        self.dropout = dropout

        if not self.if_use_features:
            self.x = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1)

        self.conv1 = GATConv(
            emb_dim,
            hidden_size,
            heads=num_heads,
            dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_size * num_heads,
            repr_dim,
            heads=1,
            dropout=dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        if not self.if_use_features:
            torch.nn.init.uniform_(self.x.weight, -1.0, 1.0)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, x=None):
        if not self.if_use_features:
            x = self.x.weight
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x)
        return x
