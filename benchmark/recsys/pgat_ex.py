__model__ = 'PGAT'

import os.path as osp
import torch
import argparse

from torch_geometric.datasets import MovieLens

from dataset import get_dataset
from train_eval import sec_order_single_run_with_kg
from models import PGATNet


torch.random.manual_seed(2019)

parser = argparse.ArgumentParser()
parser.add_argument("--n_core", type=int, default=20, help="")

parser.add_argument("--hidden_size", type=int, default=128, help="")

parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=False, help="")
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--batch_size", type=int, default=256, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=50, help="")


parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=64, help="")

args = parser.parse_args()


# Setup data and weights file path
data_path = osp.join(
    osp.dirname(osp.realpath(__file__)),
    '..', 'data', args.dataset_name
)
weights_path = osp.join(
    osp.dirname(osp.realpath(__file__)),
    '..', 'weights', __model__, args.dataset + args.dataset_name
)
data_path = osp.expanduser(osp.normpath(data_path))
weights_path = osp.expanduser(osp.normpath(weights_path))

# Setup the torch device
device = torch.device(args.device if args.device == 'cpu' else args.device + ':{}'.format(args.gpu_idx))
torch.random.manual_seed(2019)

# Display the arguments used in the experiments
dataset_args = {
    'root': data_path, 'dataset': args.dataset, 'name': args.dataset_name, 'n_core': args.n_core,
    'sec_order': False, 'train_ratio': args.train_ratio, 'debug': args.debug
}
task_args = {'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim}
train_args = {
    'debug': args.debug,
    'epochs': args.epochs, 'batch_size': args.batch_size, 'weight_decay': args.weight_decay,
    'lr': args.lr, 'device': device,
    'weights_path': weights_path, 'hidden_size': args.hidden_size}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(task_args))
print('train params: {}'.format(train_args))

data = get_dataset(dataset_args).data


def main():
    model = PGATNet(data.num_nodes[0], data.num_relations[0])
    loss_func = torch.nn.MSELoss()
    sec_order_single_run_with_kg(model, data, loss_func, train_args)


if __name__ == '__main__':
    main()
