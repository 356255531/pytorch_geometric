import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, PAConv

from .kg_net import KGNet

class PGATNet(KGNet):
    def __init__(self, hidden_size, emb_dim, heads, repr_dim, num_nodes, num_relations):
        super(PGATNet, self).__init__(emb_dim, repr_dim, num_nodes, num_relations)
        self.emb_dim = emb_dim
        self.repr_dim = self.repr_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)
        self.r_emb = torch.nn.Embedding(num_relations, repr_dim, max_norm=1, norm_type=2.0)
        self.r_proj = torch.nn.Embedding(
            num_relations, emb_dim * repr_dim, max_norm=1, norm_type=2.0
        )

        self.kg_loss_func = torch.nn.MSELoss()

        self.conv1 = GATConv(
            emb_dim, int(hidden_size // heads),
            heads=heads, dropout=0.6)
        self.conv2 = PAConv(
            int(hidden_size // heads) * heads, repr_dim,
            heads=1, dropout=0.6)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def get_attention(self):
        pass

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, sec_order_edge_index):
        '''

        :param edge_index: np.array, [2, N]
        :param sec_order_edge_index: [3, M]
        :return:
        '''
        x = F.relu(self.conv1(self.node_emb.weight, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, sec_order_edge_index)
        return x

    def predict_rating_(self, x, edge_index):
        heads = edge_index[:, 0]
        tails = edge_index[:, 1]
        est_rating = torch.sum(
            x[heads] * x[tails],
            dim=1
        ).reshape(-1, 1)
        return est_rating

    def predict_rating(self, edge_index):
        return self.predict_rating_(self.node_emb.weight, edge_index)
