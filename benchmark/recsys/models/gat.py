import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .kg_net import KGNet


class KGGATNet(KGNet):
    def __init__(self, emb_dim, num_nodes, num_relations, pretrain, hidden_size, heads, proj_node=None):
        super(KGGATNet, self).__init__(emb_dim, num_nodes, num_relations, pretrain)
        if proj_node == 'trans_e' or proj_node is None:
            self.proj_kg_node = self.trans_e_project
        elif pretrain == 'trans_r':
            self.proj_kg_node = self.trans_r_project
        elif pretrain == 'trans_h':
            self.proj_kg_node = self.trans_h_project
        else:
            raise NotImplementedError('Pretain: {} not implemented!'.format(pretrain))

        self.conv1 = GATConv(emb_dim, int(hidden_size // heads), heads=heads, dropout=0.6)
        self.conv2 = GATConv(int(hidden_size // heads) * heads, emb_dim, heads=1, concat=True, dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.dropout(self.node_emb.weight, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
