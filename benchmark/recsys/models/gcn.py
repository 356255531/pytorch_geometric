import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .kg_net import Net, KGNet


class GCNNet(Net):
    def __init__(self, num_nodes, num_relations, emb_dim, repr_dim, pretrain, hidden_size, node_projection):
        super(GCNNet, self).__init__(emb_dim, num_nodes)
        self.conv1 = GCNConv(emb_dim, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, repr_dim, cached=True)

    def check_interact_edge(self, edge_attr):
        if torch.sum(edge_attr[:, 1] == -1) > 0:
            raise ValueError('No prediction for non-interaction edges.')

    def proj_node(self, x, edge_attr):
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class KGGCNNet(KGNet):
    def __init__(self, num_nodes, num_relations, emb_dim, repr_dim, pretrain, hidden_size, node_projection):
        super(KGGCNNet, self).__init__(emb_dim, repr_dim, num_nodes, num_relations, pretrain)
        self.pretrain = pretrain
        self.node_projection = node_projection
        self.conv1 = GCNConv(emb_dim, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, emb_dim, cached=True)

    def check_interact_edge(self, edge_attr):
        if torch.sum(edge_attr[:, 1] == -1) > 0:
            raise ValueError('No prediction for non-interaction edges.')

    def proj_node(self, x, edge_attr):
        if not self.node_projection:
            proj_node = lambda x, y: x
        else:
            proj_node = self.proj_kg_node
        return proj_node(x, edge_attr)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index):
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x