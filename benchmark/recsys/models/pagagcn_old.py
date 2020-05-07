import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse
import functools


class ConvEncoder(nn.Module):
    def __init__(self, in_channel):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, in_channel, 2, stride=2)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.t_conv1(x))
        x = self.sigmoid(self.t_conv2(x))
        return x

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.t_conv1.weight)
        torch.nn.init.xavier_normal_(self.t_conv2.weight)


    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.t_conv1.weight)
        torch.nn.init.xavier_normal_(self.t_conv2.weight)


class PAGAGCN(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, hidden_size, repr_dim, dropout, path_lengths):
        super(PAGAGCN, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.att_blocks = torch.nn.ModuleList([ConvEncoder(path_length) for path_length in path_lengths])

        self.conv1 = GCNConv(emb_dim, hidden_size)
        self.conv2 = GCNConv(hidden_size, repr_dim)

    def reset_parameters(self):
        for att_block in self.att_blocks:
            att_block.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def path_transform(self, meta_path_edge_indicis, att_block):
        dense_meta_path_adjs = [
            torch.sparse.FloatTensor(
                adj,
                torch.ones(adj.shape[1], device=adj.device),
                torch.Size([self.num_nodes, self.num_nodes])
            ).to_dense().unsqueeze(0)
            for adj in meta_path_edge_indicis
        ]
        meta_path_tensor = torch.cat(dense_meta_path_adjs, dim=0)

        att = att_block(meta_path_tensor.unsqueeze(0))[0][:, :self.num_nodes, :self.num_nodes]
        dense_attended_adjs = meta_path_tensor * att

        sparse_adapted_adj = torch.eye(self.num_nodes, device=meta_path_edge_indicis[0].device).to_sparse()
        for i in range(dense_attended_adjs.shape[0]):
            sparse_adapted_adj = torch.sparse.mm(sparse_adapted_adj, dense_attended_adjs[i, :, :]).to_sparse()
            edge_index, norm = self.norm(sparse_adapted_adj.indices(), self.num_nodes, edge_weight=sparse_adapted_adj.values())
            sparse_adapted_adj = torch.sparse.FloatTensor(
                edge_index,
                norm,
                torch.Size([self.num_nodes, self.num_nodes])
            ).coalesce()

        return sparse_adapted_adj.indices(), sparse_adapted_adj.values()

    def adj_encoder(self, meta_paths):
        adapt_adjs = [
            self.path_transform(meta_path, self.att_blocks[meta_path_idx])
            for meta_path_idx, meta_path in enumerate(meta_paths)
        ]
        sparse_mean_adj_mat = functools.reduce(
            lambda a, b: (torch.cat([a[0], b[0]], dim=1), torch.cat([a[1], b[1]])),
            adapt_adjs
        )
        adapt_edge_index, adp_edge_weight = torch_sparse.coalesce(
            sparse_mean_adj_mat[0], sparse_mean_adj_mat[1],
            m=self.num_nodes, n=self.num_nodes,
            op='max'
        )
        return adapt_edge_index, adp_edge_weight

    def forward(self, x, meta_paths):
        adp_adj_edge_index, adp_adj_edge_weight = self.adj_encoder(meta_paths)
        x = F.relu(self.conv1(x, adp_adj_edge_index, edge_weight=adp_adj_edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adp_adj_edge_index, edge_weight=adp_adj_edge_weight)
        return x
