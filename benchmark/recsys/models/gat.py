import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .kg_net import KGNet

class GATNet(KGNet):
    def __init__(self, hidden_size, heads, emb_dim, repr_dim, num_nodes, num_relations):
        super(GATNet, self).__init__(emb_dim, repr_dim, num_nodes, num_relations)
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)
        self.r_emb = torch.nn.Embedding(num_relations, repr_dim, max_norm=1, norm_type=2.0)
        self.r_proj = torch.nn.Embedding(
            num_relations, emb_dim * repr_dim, max_norm=1, norm_type=2.0
        )

        self.kg_loss_func = torch.nn.MSELoss()

        self.conv1 = GATConv(emb_dim, hidden_size // heads, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_size, repr_dim, heads=1, concat=True, dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.dropout(self.node_emb.weight, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
