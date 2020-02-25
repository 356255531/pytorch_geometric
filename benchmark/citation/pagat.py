import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import PAGATConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import create_path, remove_self_loops

from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
args = parser.parse_args()


def pre_transform(data):
    edge_index = data.edge_index
    edge_index = remove_self_loops(edge_index)[0]
    path_index = torch.tensor(create_path(edge_index, 2)).long()

    head, mid, tail = path_index
    mask = head != tail
    path_index = path_index[:, mask]
    new_path_index = torch.arange(0, data.num_nodes, dtype=torch.long,
                              device=edge_index.device)
    new_path_index = new_path_index.unsqueeze(0).repeat(3, 1)
    data.path_index = torch.cat([path_index, new_path_index], dim=1)
    data.path_index = path_index
    return data


class AttPathEncoder(torch.nn.Module):
    def __init__(self, in_channels, heads):
        super(AttPathEncoder, self).__init__()
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)

    def forward(self, path_index_without_target, x):
        x_path = x[path_index_without_target.T]


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = PAGATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = PAGATConv(
            args.hidden,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, path_index = data.x, data.path_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(self.conv1(x, path_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, path_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features, pre_transform=pre_transform)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
