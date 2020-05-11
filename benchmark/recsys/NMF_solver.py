import argparse
import torch
import os
import numpy as np
import random as rd

from models import NMF
from utils import get_folder_path
from base_solver import BaseMFSolver


MODEL = 'NMF'

parser = argparse.ArgumentParser()
# Dataset params
parser.add_argument("--dataset", type=str, default='Movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--if_use_features", type=bool, default=False, help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--num_feat_core", type=int, default=10, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
# Model params
parser.add_argument("--hidden_mf", type=int, default=8, help="")
parser.add_argument("--hidden_mlp", type=int, default=8, help="")
parser.add_argument("--layers", type=list, default=[16, 32, 16, 8], help="")
# Train params
parser.add_argument("--num_negative_samples", type=int, default=5, help="")
parser.add_argument("--init_eval", type=bool, default=False, help="")

parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=100, help="")
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--batch_size", type=int, default=4, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="")
parser.add_argument("--early_stopping", type=int, default=60, help="")
parser.add_argument("--save_epochs", type=list, default=[10, 40, 80], help="")
parser.add_argument("--save_every_epoch", type=int, default=40, help="")

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
    'if_use_features': args.if_use_features,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'train_ratio': args.train_ratio
}
model_args = {
    'hidden_mf': args.hidden_mf,
    'hidden_mlp': args.hidden_mlp, 'layers': args.layers
}
train_args = {
    'init_eval': args.init_eval, 'num_negative_samples': args.num_negative_samples,
    'opt': args.opt, 'loss': args.loss,
    'runs': args.runs, 'epochs': args.epochs, 'batch_size': args.batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': args.save_epochs, 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


class NMFSolver(BaseMFSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(NMFSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def train_negative_sampling(self, u_nid, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, data):
        """

        :param u_nid:
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :param data:
        :return:
        """
        num_pos_samples = len(train_pos_unid_inid_map[u_nid])

        negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
        nid_occs = np.array([data.item_nid_occs[0][nid] for nid in negative_inids])
        nid_occs = nid_occs / np.sum(nid_occs)
        negative_inids = rd.choices(population=negative_inids, weights=nid_occs, k=num_pos_samples * 5)

        return negative_inids

    def generate_candidates(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        pos_i_nids = test_pos_unid_inid_map[u_nid]
        neg_i_nids = np.array(neg_unid_inid_map[u_nid])

        neg_i_nids_indices = np.array(rd.sample(range(neg_i_nids.shape[0]), 99), dtype=int)

        return pos_i_nids, list(neg_i_nids[neg_i_nids_indices])


if __name__ == '__main__':
    solver = NMFSolver(NMF, dataset_args, model_args, train_args)
    solver.run()
