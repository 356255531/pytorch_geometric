import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_sparse
import MinkowskiEngine as ME

from datasets import get_planetoid_dataset
from train_eval import run, random_planetoid_splits

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channel, out_channel, D):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channel, 16, 3, dimension=D)
        self.conv2 = ME.MinkowskiConvolution(16, 4, 3, dimension=D)
        self.pool = ME.MinkowskiMaxPooling(2, 2, dimension=D)

        self.t_conv1 = ME.MinkowskiConvolutionTranspose(4, 16, 2, stride=2, dimension=D)
        self.t_conv2 = ME.MinkowskiConvolutionTranspose(16, out_channel, 2, stride=2, dimension=D)

        self.relu = ME.MinkowskiReLU()
        self.sigmoid = ME.MinkowskiSigmoid()

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

        self.att_block = ConvAutoencoder(2, 2, D=1)

        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.att_block.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def path_transform(self, meta_path_edge_indicis, att_block):
        device = meta_path_edge_indicis[0].device

        coords = meta_path_edge_indicis[0].T
        feats = torch.ones((meta_path_edge_indicis[0].shape[1], 2), device=device)
        minkow_meta_path_sparse_mat = ME.SparseTensor(coords=coords, feats=feats)
        att = att_block(minkow_meta_path_sparse_mat)
        minkow_sparse_attended_adjs = minkow_meta_path_sparse_mat * att

        coords, feats = minkow_sparse_attended_adjs.coords, minkow_sparse_attended_adjs.feats
        sparse_attended_adjs = []
        for feat_idx in range(feats.shape[1]):
            idx = feats[:, feat_idx] != 0
            sparse_attended_adjs.append(torch_sparse.coalesce(coords[idx].long().T, feats[idx, feat_idx], m=3, n=2, op='mean'))

        sparse_adapted_adj = torch_sparse.eye(self.size, device=device)
        for sparse_attended_adj in sparse_attended_adjs:
            sparse_adapted_adj = torch_sparse.spspmm(*sparse_attended_adj, *sparse_attended_adj, m=3, k=3, n=2)
        index, val = sparse_adapted_adj

        return index, val / len(meta_path_edge_indicis)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        meta_path = [data.edge_index, data.edge_index]

        meta_edge_index, edge_weight = self.path_transform(meta_path, self.att_block)

        sparse_meta_adj_mat = torch.sparse.FloatTensor(
            meta_edge_index,
            edge_weight,
            torch.Size([self.size, self.size])
        )
        sparse_adj_mat = torch.sparse.FloatTensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=sparse_meta_adj_mat.device),
            torch.Size([x.shape[0], x.shape[0]])
        )
        sparse_mean_adj_mat = (sparse_meta_adj_mat + sparse_adj_mat) / 2
        sparse_mean_adj_mat = sparse_mean_adj_mat.coalesce()
        adp_adj_edge_index, adp_adj_edge_weight = sparse_mean_adj_mat.indices(), sparse_mean_adj_mat.values()
        x = F.relu(self.conv1(x, adp_adj_edge_index, edge_weight=adp_adj_edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, adp_adj_edge_index, edge_weight=adp_adj_edge_weight)

        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=args.dropout, training=self.training)
        # x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
Net(dataset)(dataset.data)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
