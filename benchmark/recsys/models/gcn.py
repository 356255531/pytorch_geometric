import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .kg_net import ExKGNet


class GCNNetEx(ExKGNet):
    def __init__(self, num_nodes, num_relations, hidden_size, emb_dim, repr_dim):
        super(GCNNetEx, self).__init__(emb_dim, repr_dim, num_nodes, num_relations)
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)
        self.r_emb = torch.nn.Embedding(num_relations, repr_dim, max_norm=1, norm_type=2.0)
        self.r_proj = torch.nn.Embedding(
            num_relations, emb_dim * repr_dim, max_norm=1, norm_type=2.0
        )

        self.kg_loss_func = torch.nn.MSELoss()

        self.conv1 = GCNConv(emb_dim, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, repr_dim, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
