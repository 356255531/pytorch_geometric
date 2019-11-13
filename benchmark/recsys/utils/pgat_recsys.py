__model__ = 'PGAT'

import argparse
import torch

from torch_geometric.datasets import MovieLens

from benchmark.recsys.utils import get_folder_path
from benchmark.recsys.models import PGATNet
from benchmark.recsys.train_eval import run_with_kg
from benchmark.recsys.models import PGATNet



########################################### Parse arguments ###########################################
parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--sec_order", type=bool, default=True, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=False, help="")

# Model params
parser.add_argument("--heads", type=int, default=4, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")

# Train params
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='2', help="")
parser.add_argument("--runs", type=int, default=10, help="")
parser.add_argument("--epochs", type=int, default=50, help="")
parser.add_argument("--kg_opt", type=str, default='adam', help="")
parser.add_argument("--cf_opt", type=str, default='adam', help="")
parser.add_argument("--kg_loss", type=str, default='mse', help="")
parser.add_argument("--cf_loss", type=str, default='mse', help="")
parser.add_argument("--kg_batch_size", type=int, default=1028, help="")
parser.add_argument("--cf_batch_size", type=int, default=256, help="")
parser.add_argument("--sec_order_batch_size", type=int, default=1028, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")

args = parser.parse_args()

########################################### Initialization ###########################################
# Setup data and weights file path
data_folder, weights_folder, logger_folder = get_folder_path(__model__, args.dataset + args.dataset_name)

# Setup the torch device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

########################################### Display all arguments ###########################################
dataset_args_dict = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'sec_order': args.sec_order, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args_dict = {
    'heads': args.heads, 'hidden_size': args.hidden_size, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim
}
train_args_dict = {
    'debug': args.debug, 'runs': args.runs,
    'model': __model__,
    'kg_opt': args.kg_opt, 'kg_loss': args.kg_loss, 'cf_loss': args.cf_loss, 'cf_opt': args.cf_opt,
    'epochs': args.epochs, 'sec_order_batch_size': args.sec_order_batch_size, 'kg_batch_size': args.kg_batch_size, 'cf_batch_size': args.cf_batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': weights_folder, 'logger_folder': logger_folder}
print('dataset params: {}'.format(dataset_args_dict))
print('task params: {}'.format(model_args_dict))
print('train params: {}'.format(train_args_dict))



if __name__ == '__main__':
    main()
