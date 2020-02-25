import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, PAGATConv

from .kg_net import KGNet


class KGPGATNet(KGNet):
    def __init__(self, emb_dim, num_nodes, num_relations, pretrain, hidden_size, heads, proj_node=None):
        super(KGPGATNet, self).__init__(emb_dim, num_nodes, num_relations, pretrain)
        if proj_node == 'trans_e' or proj_node is None:
            self.proj_kg_node = self.trans_e_project
        elif pretrain == 'trans_r':
            self.proj_kg_node = self.trans_r_project
        elif pretrain == 'trans_h':
            self.proj_kg_node = self.trans_h_project
        else:
            raise NotImplementedError('Pretain: {} not implemented!'.format(pretrain))

        self.conv1 = GATConv(
            emb_dim, int(hidden_size // heads),
            heads=heads, dropout=0.6)
        self.conv2 = PAGATConv(
            int(hidden_size // heads) * heads, emb_dim,
            heads=1, dropout=0.6)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward_(self, edge_index, sec_order_edge_index):
        self.forward(self.node_emb.weight, edge_index, sec_order_edge_index)

    def forward(self, x, edge_index, sec_order_edge_index):
        '''

        :param edge_index: np.array, [2, N]
        :param sec_order_edge_index: [3, M]
        :return:
        '''
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, sec_order_edge_index)
        return x

