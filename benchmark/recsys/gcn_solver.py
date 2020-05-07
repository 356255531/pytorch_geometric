import argparse
import torch
import os
import numpy as np

from models import GCN
from utils import get_folder_path
from base_solver import BaseSolver


MODEL = 'GCN'

parser = argparse.ArgumentParser()
# Dataset params
parser.add_argument("--dataset", type=str, default='Movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--num_feat_core", type=int, default=10, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
# Model params
parser.add_argument("--dropout", type=float, default=0.5, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
# Train params
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=100, help="")
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--batch_size", type=int, default=4, help="")
parser.add_argument("--lr", type=float, default=1e-3, help="")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")
parser.add_argument("--save_epochs", type=list, default=[50, 80, 100], help="")
parser.add_argument("--save_every_epoch", type=int, default=120, help="")


# Recommender params
parser.add_argument("--init_eval", type=bool, default=True, help="")
parser.add_argument("--num_recs", type=int, default=10, help="")
args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'train_ratio': args.train_ratio
}
model_args = {
    'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout
}
train_args = {
    'opt': args.opt, 'loss': args.loss,
    'runs': args.runs, 'epochs': args.epochs, 'batch_size': args.batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': args.save_epochs, 'save_every_epoch': args.save_every_epoch
}
rec_args = {
    'init_eval': args.init_eval, 'num_recs': args.num_recs
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))
print('rec params: {}'.format(rec_args))


class GCNSolver(BaseSolver):
    def __init__(self, GCN, dataset_args, model_args, train_args, rec_args):
        super(GCNSolver, self).__init__(GCN, dataset_args, model_args, train_args, rec_args)

    def prepare_model_input(self, data):
        edge_index_np = np.hstack(list(data.edge_index_nps[0].values()))
        edge_index_np = np.hstack([edge_index_np, np.flip(edge_index_np, 0)])
        edge_index = torch.from_numpy(edge_index_np).long().to(self.train_args['device'])
        x = data.x

        return {'edge_index': edge_index, 'x': x}

    def train_negative_sampling(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        return neg_unid_inid_map[u_nid] + test_pos_unid_inid_map[u_nid]

    def eval_sampling(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        pos_i_nids = test_pos_unid_inid_map[u_nid]
        neg_i_nids = neg_unid_inid_map[u_nid]
        return pos_i_nids, neg_i_nids


if __name__ == '__main__':
    solver = GCNSolver(GCN, dataset_args, model_args, train_args, rec_args)
    solver.run()
