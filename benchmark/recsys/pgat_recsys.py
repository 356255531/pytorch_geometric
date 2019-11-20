__model__ = 'PGAT'

import argparse
import torch

from torch_geometric.datasets import MovieLens

from benchmark.recsys.utils import get_folder_path
from benchmark.recsys.models import PGATNetEx
from benchmark.recsys.train_eval import run
from benchmark.recsys.models import PGATNetEx

class KGAT_RS(object):
    def __int__(self, weights_path):
        dataset = MovieLens(**dataset_args)
        self.data = dataset.data

        self.model = PGAT()
        self.model.restore(weights_path).train(False)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data.to(self.device)
        self.model.to(self.device)

        self.loss_func = torch.nn.MSELoss()

    def get_top_n_popular_items(self, n=10):
        """
        Get the top n movies from self.data.ratings.
        Remove the duplicates in self.data.ratings and sort it by movie count.
        After you find the top N popular movies' item id,
        look over the details information of item in self.data.movies

        :param n: the number of items, int
        :return: (item_idx, item_url, item_attr), tuple(int, str, dict(str, str))
        """
        raise NotImplemented

    def build_user(self, interactions, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param interactions: [N, 2] np.array, [:, 0] is iids and [:, 1] is ratings
        :param demographic_info: (gender, occupation), tuple
        :return:
        """
        new_user_emb = torch.nn.Embedding(1, args.emb_dim).weight.to(self.device)
        x = torch.cat([self.data.x.detach(), new_user_emb], dim=0)

        self.opt = torch.optim.Adam(new_user_emb, lr=args.lr, weight_decay=args.weight_decay)

        new_edge_index = create_new_edge_index(
            self.data, interactions, demographic_info, self.device
        )
        new_sec_order_edge_index = create_sec_order_new_edge_index(
            new_edge_index, self.data.edge_index, self.device
        )

        for _ in args.epochs:
            user_repr = self.model(x, new_edge_index, new_sec_order_edge_index)[-1:, :]
            i_node_ids = np.array([self.data.items.loc[self.data.items['iid'] == iid] for iid in iids])
            item_repr = x[i_node_ids]
            loss_t = self.loss_func(torch.sum(user_repr * item_repr, dim=1), ratings)

            self.opt.zero_gradients()
            loss_t.backward()
            self.opt.step()

            user_repr = self.model(x, new_edge_index, new_sec_order_edge_index)[-1:, :]
        self.new_user_repr = user_repr

    def get_recommendations(self, n=10):
        """
        Give users and get recommendations
        :param user_idx: int
        :param n: number of recommendations, int
        :return: [(item_idx, attr_attention, neighbor_user_attention)], list(tuple(int, dict(int, foloat), dict(int, float)))

        example:
            [(134, {'director': 0.2, 'actor': 0.8}), {13: 0.1, 15:0.2, ...}]
        """
        unseen_movie_iids = get_unseen_movie_iids(self.data.items, self.interactions)
        unseen_movie_node_ids = self.data.items[self.data.items['iids'] == unseen_movie_iids]['node_id']

        est_ratings = torch.sum(self.new_user_repr * self.data.x[unseen_movie_node_ids], dim=0).numpy()
        top_n_item

    def update_user(self, interaction):
        """
        update user profiles in the RS
        :param user_iteractions: {item_idx: rating}, dict(int, int)

        :return: None
        """
        pass

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
