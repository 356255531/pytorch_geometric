import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, hidden_size, heads, emb_dim, repr_dim, num_nodes, num_relations):
        super(GATNet, self).__init__()
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

    def get_kg_loss(self, edge_index, edge_attr):
        r_idx = edge_attr[:, 0]
        r_emb = self.r_emb.weight[r_idx]
        r_proj = self.r_proj.weight[r_idx].reshape(-1, self.emb_dim, self.repr_dim)
        proj_head = torch.matmul(self.node_emb.weight[edge_index[:, :1]], r_proj).reshape(-1, self.repr_dim)

        proj_tail = torch.matmul(self.node_emb.weight[edge_index[:, 1:2]], r_proj).reshape(-1, self.repr_dim)

        est_tail = proj_head + r_emb

        loss_t = self.kg_loss_func(est_tail, proj_tail)

        return loss_t
