import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

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
    def __init__(self, in_channel, out_channel):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(in_channel, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, out_channel, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.softmax(self.t_conv2(x), dim=1)

        return x


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.size = dataset.data.x.shape[0]

        self.att_block_1 = ConvAutoencoder(2, 2)
        self.att_block_2 = ConvAutoencoder(3, 3)

        self.conv1_1 = GCNConv(dataset.num_features, args.hidden)
        self.conv1_2 = GCNConv(dataset.num_features, args.hidden)
        self.conv1_3 = GCNConv(dataset.num_features, args.hidden)

        self.conv2_1 = GCNConv(args.hidden, dataset.num_classes)
        self.conv2_2 = GCNConv(args.hidden, dataset.num_classes)
        self.conv2_3 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv1_3.reset_parameters()

        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv2_3.reset_parameters()

    def path_transform(self, meta_path_edge_indicis, att_block, meta_path_edge_values=None):
        if meta_path_edge_values is None:
            # Convert sparse adj_mat to dense
            dense_meta_path_adjs = [
                torch.sparse.FloatTensor(
                    adj,
                    torch.ones(adj.shape[1], device=adj.device),
                    torch.Size([self.size, self.size])
                ).to_dense().unsqueeze(0)
                for adj in meta_path_edge_indicis
            ]
            meta_path_tensor = torch.cat(dense_meta_path_adjs, dim=0)

            # Compute the dense attended adj_mat
            att = att_block(meta_path_tensor.unsqueeze(0))[0]
            dense_attended_adjs = meta_path_tensor * att
            # dense_attended_adjs = meta_path_tensor

            # convert the dense attended adj_mat to sparse
            sparse_adapted_adj = torch.eye(self.size).to_sparse()
            for i in range(dense_attended_adjs.shape[0]):
                sparse_adapted_adj = torch.sparse.mm(sparse_adapted_adj, dense_attended_adjs[i, :, :]).to_sparse()
        else:
            raise NotImplementedError
        return sparse_adapted_adj.indices(), sparse_adapted_adj.values()

    def forward(self, data):
        meta_path_1 = [data.edge_index, data.edge_index]
        meta_path_2 = [data.edge_index, data.edge_index, data.edge_index]

        meta_edge_index_1, edge_weight_1 = self.path_transform(meta_path_1, self.att_block_1)
        meta_edge_index_2, edge_weight_2 = self.path_transform(meta_path_2, self.att_block_2)

        x = data.x
        x_1 = F.relu(self.conv1_1(x, data.edge_index))
        x_2 = F.relu(self.conv1_2(x, meta_edge_index_1, edge_weight=edge_weight_1))
        x_3 = F.relu(self.conv1_3(x, meta_edge_index_2, edge_weight=edge_weight_2))
        x = torch.mean(torch.stack([x_1, x_2, x_3]), dim=0)

        x = F.dropout(x, p=args.dropout, training=self.training)

        x_1 = self.conv2_1(x, data.edge_index)
        x_2 = self.conv2_2(x, meta_edge_index_1, edge_weight=edge_weight_1)
        x_3 = self.conv2_3(x, meta_edge_index_2, edge_weight=edge_weight_2)
        x = torch.mean(torch.stack([x_1, x_2, x_3]), dim=0)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
Net(dataset)(dataset.data)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
