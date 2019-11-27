import argparse
import torch

from utils import get_folder_path
from train_eval import run
from models import KGGCNNet, GCNNet

MODEL = 'GCN'
IMPLICIT = True
SEC_ORDER = False

########################################### Parse arguments ###########################################
parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=0.01, help="")

# Model params
parser.add_argument("--hidden_size", type=int, default=128, help="")
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--repr_dim", type=int, default=64, help="")
parser.add_argument("--pretrain", type=bool, default=False, help="")
parser.add_argument("--node_projection", type=bool, default=False, help="")

# Train params
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--beta", type=float, default=0.5, help="")
parser.add_argument("--runs", type=int, default=10, help="")
parser.add_argument("--pretrain_epochs", type=int, default=10, help="")
parser.add_argument("--epochs", type=int, default=50, help="")
parser.add_argument("--kg_opt", type=str, default='adam', help="")
parser.add_argument("--cf_opt", type=str, default='adam', help="")
parser.add_argument("--kg_loss", type=str, default='mse', help="")
parser.add_argument("--cf_loss", type=str, default='mse', help="")
parser.add_argument("--kg_batch_size", type=int, default=1028, help="")
parser.add_argument("--cf_batch_size", type=int, default=1028, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")

args = parser.parse_args()

########################################### Initialization ###########################################
# Setup data and weights file path
dataset = '{}_core{}_{}_ratio{}'.format(args.dataset + args.dataset_name, args.num_core, 'implicit' if IMPLICIT else 'explicit', args.train_ratio)
dataset = dataset + '_secorder' if SEC_ORDER else dataset
dataset = dataset + '_debug{}'.format(args.debug) if args.debug else dataset
data_folder, weights_folder, logger_folder = get_folder_path(MODEL, dataset)

# Setup the torch device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

########################################### Display all arguments ###########################################
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'sec_order': SEC_ORDER, 'train_ratio': args.train_ratio,
    'implicit': IMPLICIT, 'debug': args.debug,
}
model_class = KGGCNNet if args.pretrain else GCNNet
node_projection = args.node_projection & args.pretrain
model_args = {
    'hidden_size': args.hidden_size, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim, 'pretrain': args.pretrain,
    'node_projection': node_projection
}
train_args = {
    'model': MODEL, 'debug': args.debug,
    'runs': args.runs, 'epochs': args.epochs,
    'pretrain': args.pretrain, 'pretrain_epochs': args.pretrain_epochs,
    'kg_opt': args.kg_opt, 'kg_loss': args.kg_loss, 'cf_loss': args.cf_loss, 'cf_opt': args.cf_opt,
    'kg_batch_size': args.kg_batch_size, 'cf_batch_size': args.cf_batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'early_stopping': args.early_stopping,
    'device': device, 'weights_folder': weights_folder, 'logger_folder': logger_folder}
print('dataset params: {}'.format(dataset_args))
print('model params: {}'.format(model_args))
print('train params: {}'.format(train_args))


def main():
    run(model_class, dataset_args, model_args, train_args)


if __name__ == '__main__':
    main()
