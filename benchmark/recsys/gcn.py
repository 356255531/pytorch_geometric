import os.path as osp

import torch
import torch.nn.functional as F
import argparse

from torch_geometric.datasets import MovieLens
from torch_geometric.nn import GCNConv

from train_eval import single_run_with_kg


torch.random.manual_seed(2019)

parser = argparse.ArgumentParser()
parser.add_argument("--n_core", type=int, default=10, help="")

parser.add_argument("--hidden_size", type=int, default=128, help="")

parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=0.01, help="")
parser.add_argument("--epochs", type=int, default=40, help="")
parser.add_argument("--batch_size", type=int, default=1024, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")

parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=64, help="")

args = parser.parse_args()

data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')
weights_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'weights', '1m')

dataset_args = {
    'root': data_path, 'name': '1m', 'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim,
    'n_core': args.n_core, 'sec_order': False, 'train_ratio': args.train_ratio, 'debug': args.debug
}
task_args = {'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim}
train_args = {'epochs': args.epochs, 'batch_size': args.batch_size, 'weight_decay': args.weight_decay, 'weights_path': weights_path}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(task_args))
print('train params: {}'.format(train_args))

data = MovieLens(**dataset_args).data


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(args.emb_dim, args.hidden_size, cached=True)
        self.conv2 = GCNConv(args.hidden_size, args.repr_dim, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    model = GCNNet()
    loss_func = torch.nn.MSELoss()
    single_run_with_kg(data, model, loss_func, train_args, task_args)


if __name__ == '__main__':
    main()
