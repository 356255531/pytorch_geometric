import os.path as osp

import torch
import torch.nn.functional as F
import argparse

from torch_geometric.datasets import MovieLens
from torch_geometric.nn import GATConv

from train_eval import single_run_with_kg


torch.random.manual_seed(2019)

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int, default=128, help="")
parser.add_argument("--heads", type=int, default=8, help="")

parser.add_argument("--epochs", type=int, default=40, help="")
parser.add_argument("--batch_size", type=int, default=1024, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")


parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=64, help="")

args = parser.parse_args()

task_args = {'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim}
train_args = {'epochs': args.epochs, 'batch_size': args.batch_size, 'weight_decay': args.weight_decay}


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')
data = MovieLens(path, '1m', train_ratio=0.8, debug=0.01).data


class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(args.emb_dim, args.hidden_size // args.heads, heads=args.heads, dropout=0.6)
        self.conv2 = GATConv(args.hidden_size, args.repr_dim, heads=1, concat=True, dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    model = GATNet()
    loss_func = torch.nn.MSELoss()
    single_run_with_kg(data, model, loss_func, train_args, task_args)


if __name__ == '__main__':
    main()
