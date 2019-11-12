__model__ = 'GCN'

import argparse
import torch

from utils import get_folder_path
from train_eval import run_with_kg
from models import GCNNet

########################################### Parse arguments ###########################################
parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--sec_order", type=bool, default=False, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=False, help="")


# Model params
parser.add_argument("--hidden_size", type=int, default=128, help="")
parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")

# Train params
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=10, help="")
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--batch_size", type=int, default=256, help="")
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
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'sec_order': args.sec_order, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args = {
    'model': __model__, 'hidden_size': args.hidden_size, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim
}
train_args = {
    'debug': args.debug, 'runs': args.runs,
    'kg_opt': 'adam', 'kg_loss': 'mse', 'cf_loss': 'mse', 'cf_opt': 'adam',
    'epochs': args.epochs, 'batch_size': args.batch_size, 'weight_decay': args.weight_decay,
    'lr': args.lr, 'device': device,
    'weights_folder': weights_folder, 'logger_folder': logger_folder}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


def main():
    run_with_kg(GCNNet, model_args, dataset_args, train_args)


if __name__ == '__main__':
    main()
