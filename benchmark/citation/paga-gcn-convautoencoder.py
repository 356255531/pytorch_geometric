import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
import torch_sparse
from torch_geometric.utils import add_remaining_self_loops
import functools

from datasets import get_planetoid_dataset
from train_eval import run, random_planetoid_splits

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=5)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class ConvEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, out_channel, 2, stride=2)

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


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.size = dataset.data.x.shape[0]

        self.att_blocks = nn.ModuleList([ConvEncoder(2, 2)])

        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

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
                torch.Size([self.size, self.size])
            ).to_dense().unsqueeze(0)
            for adj in meta_path_edge_indicis
        ]
        meta_path_tensor = torch.cat(dense_meta_path_adjs, dim=0)

        att = att_block(meta_path_tensor.unsqueeze(0))[0]
        dense_attended_adjs = meta_path_tensor * att

        sparse_adapted_adj = torch.eye(self.size).to_sparse()
        for i in range(dense_attended_adjs.shape[0]):
            sparse_adapted_adj = torch.sparse.mm(sparse_adapted_adj, dense_attended_adjs[i, :, :]).to_sparse()
            edge_index, norm = self.norm(sparse_adapted_adj.indices(), self.size, edge_weight=sparse_adapted_adj.values())
            sparse_adapted_adj = torch.sparse.FloatTensor(
                edge_index,
                norm,
                torch.Size([self.size, self.size])
            ).coalesce()

        return sparse_adapted_adj.indices(), sparse_adapted_adj.values()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        meta_path = [[data.edge_index], [data.edge_index, data.edge_index]]

        att_adjs = [self.path_transform(meta_path, self.att_blocks[meta_path_idx]) for meta_path_idx, meta_path in enumerate(meta_path)]
        att_adj_edge_index = [_[0] for _ in att_adjs]
        att_adj_edge_index = functools.reduce(lambda x, y: torch.cat([x, y], dim=1), att_adj_edge_index)
        att_adj_edge_values = [_[1] for _ in att_adjs]
        att_adj_edge_values = functools.reduce(lambda x, y: torch.cat([x, y]), att_adj_edge_values)
        adapt_edge_index, adp_edge_weight = torch_sparse.coalesce(
            att_adj_edge_index, att_adj_edge_values,
            m=x.shape[0], n=x.shape[0],
            op='max'
        )

        x = F.relu(self.conv1(x, adapt_edge_index, edge_weight=adp_edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, adapt_edge_index, edge_weight=adp_edge_weight)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
Net(dataset)(dataset.data)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)