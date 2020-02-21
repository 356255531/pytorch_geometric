import torch
from torch_geometric.nn import GATConv, PAGATConv
import torch.nn.functional as F


class PGAT(torch.nn.Module):
    def __init__(self, emb_dim, hidden, repr_dim, head=1):
        super(PGAT, self).__init__()
        self.conv1 = GATConv(emb_dim, hidden // head, heads=head, dropout=0.6)
        self.conv2 = PAGATConv(hidden // head * head, repr_dim, heads=1, dropout=0.6)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def get_attention(self):
        pass

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, sec_order_edge_index):
        '''

        :param x:
        :param edge_index: np.array, [2, N]
        :param sec_order_edge_index: [3, M]
        :return:
        '''
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, sec_order_edge_index)
        return x