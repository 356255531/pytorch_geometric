import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .kg_net import Net, KGNet


class GCNNet(Net):
    def __init__(self, emb_dim, num_nodes, hidden_size):
        super(GCNNet, self).__init__(emb_dim, num_nodes)
        self.proj_kg_node = lambda x: x

        self.conv1 = GCNConv(emb_dim, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, emb_dim, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class KGGCNNet(KGNet):
    def __init__(self, num_nodes, num_relations, emb_dim, pretrain, hidden_size, proj_node=None):
        super(KGGCNNet, self).__init__(emb_dim, num_nodes, num_relations, pretrain)
        if proj_node == 'trans_e' or proj_node is None:
            self.proj_kg_node = self.trans_e_project
        elif pretrain == 'trans_r':
            self.proj_kg_node = self.trans_r_project
        elif pretrain == 'trans_h':
            self.proj_kg_node = self.trans_h_project
        else:
            raise NotImplementedError('Pretain: {} not implemented!'.format(pretrain))

        self.conv1 = GCNConv(emb_dim, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, emb_dim, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x