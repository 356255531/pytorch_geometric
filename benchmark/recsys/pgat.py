import os.path as osp

import torch
import torch.nn.functional as F
import argparse

from torch_geometric.datasets import MovieLens
from torch_geometric.nn import GATConv, PAConv

from train_eval import sec_order_single_run_with_kg


torch.random.manual_seed(2019)

parser = argparse.ArgumentParser()
parser.add_argument("--n_core", type=int, default=10, help="")

parser.add_argument("--hidden_size", type=int, default=128, help="")
parser.add_argument("--heads", type=int, default=4, help="")

parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=0.1, help="")
parser.add_argument("--epochs", type=int, default=40, help="")
parser.add_argument("--lr", type=float, default=1e-3, help="")
parser.add_argument("--batch_size", type=int, default=256, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")

parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=64, help="")

args = parser.parse_args()

data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')
weights_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'weights', '1m')

dataset_args = {
    'root': data_path, 'name': '1m', 'n_core': args.n_core, 'sec_order': True,
    'train_ratio': args.train_ratio, 'debug': args.debug
}
task_args = {'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim}
train_args = {'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size, 'weight_decay': args.weight_decay}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(task_args))
print('train params: {}'.format(train_args))

data = MovieLens(**dataset_args).data


class PGAT(torch.nn.Module):
    def __init__(self, num_nodes, num_relations):
        super(PGAT, self).__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, args.emb_dim, max_norm=1, norm_type=2.0)
        self.r_emb = torch.nn.Embedding(num_relations, args.repr_dim, max_norm=1, norm_type=2.0)
        self.r_proj = torch.nn.Embedding(
            num_relations, args.emb_dim * args.repr_dim, max_norm=1, norm_type=2.0
        )

        self.kg_loss_func = torch.nn.MSELoss()

        self.conv1 = GATConv(
            args.emb_dim, int(args.hidden_size // args.heads),
            heads=args.heads, dropout=0.6)
        self.conv2 = PAConv(
            int(args.hidden_size // args.heads) * args.heads, args.repr_dim,
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

    def predict_(self, x, edge_index):
        heads = edge_index[:, 0]
        tails = edge_index[:, 1]
        est_rating = torch.sum(
            x[heads] * x[tails],
            dim=1
        ).reshape(-1, 1)
        return est_rating

    def predict(self, edge_index):
        return self.predict_(self.node_emb.weight, edge_index)

    def get_kg_loss(self, edge_index, edge_attr):
        r_idx = edge_attr[:, 0]
        r_emb = self.r_emb.weight[r_idx]
        r_proj = self.r_proj.weight[r_idx].reshape(-1, args.emb_dim, args.repr_dim)
        proj_head = torch.matmul(self.node_emb.weight[edge_index[:, :1]], r_proj).reshape(-1, args.repr_dim)

        self.proj_tail = torch.matmul(self.node_emb.weight[edge_index[:, 1:2]], r_proj).reshape(-1, args.repr_dim)

        est_tail = proj_head + r_emb

        loss_t = self.kg_loss_func(est_tail, self.proj_tail)

        return loss_t


def main():
    model = PGAT(data.num_nodes[0], data.num_relations[0])
    loss_func = torch.nn.MSELoss()
    sec_order_single_run_with_kg(model, data, loss_func, train_args)


if __name__ == '__main__':
    main()
