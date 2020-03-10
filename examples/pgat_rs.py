import numpy as np

import torch
from torch_geometric.datasets import MovieLens
import pandas as pd
from benchmark.kernel import PGAT
from torch_geometric.nn import GATConv, PAGATConv

import argparse
import os.path as osp


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

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')

# What you need to change: root (where you put the processed data)
dataset_args = {
    'root': path, 'name': '1m', 'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim,
    'n_core': args.n_core, 'sec_order': True, 'train_ratio': args.train_ratio,
}
task_args = {'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim}
train_args = {'epochs': args.epochs, 'batch_size': args.batch_size, 'weight_decay': args.weight_decay}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(task_args))
print('train params: {}'.format(train_args))


def create_new_edge_index(interactions, demographic_info, data, device):
    iids, ratings = interactions
    gender, occupation = demographic_info

    new_user_node_id = data.x.shape[0]

    row = []
    col = []
    edge_attrs = []

    # Add gender edges
    row.append(new_user_node_id)
    col.append(data.genders_node_id_map[gender])
    edge_attrs.append([data.relation_map['gender'], -1])

    col.append(new_user_node_id)
    row.append(data.genders_node_id_map[gender])
    edge_attrs.append([data.relation_map['-gender'], -1])

    # Add occupation edges
    row.append(new_user_node_id)
    col.append(data.occupation_node_id_map[occupation])
    edge_attrs.append([data.relation_map['occupation'], -1])

    col.append(new_user_node_id)
    row.append(data.occupation_node_id_map[occupation])
    edge_attrs.append([data.relation_map['-occupation'], -1])

    # Add ratings edges
    for iid, rating in zip(iids, ratings):
        i_node_id = int(data.items[data.items['iid'] == iid]['node_id'])

        row.append(new_user_node_id)
        col.append(i_node_id)
        edge_attrs.append([data.relation_map['interact'], rating])

        col.append(new_user_node_id)
        row.append(i_node_id)
        edge_attrs.append([data.relation_map['-interact'], rating])

    row = np.array(row).reshape(1, -1)
    col = np.array(col).reshape(1, -1)
    new_edge_index = np.concatenate([row, col], axis=0)
    new_edge_index = torch.from_numpy(new_edge_index).to(device)

    return new_edge_index


def create_sec_order_new_edge_index(edge_index, new_edge_index, device):
    heads, middles = new_edge_index
    heads = heads.cpu().detach().numpy()
    middles = middles.cpu().detach().numpy()

    adj_1 = pd.DataFrame({'heads': heads, 'middles': middles})

    middles, tails = edge_index
    middles = middles.cpu().detach().numpy()
    tails = tails.cpu().detach().numpy()

    adj_2 = pd.DataFrame({'middles': middles, 'tails': tails})

    path = adj_1.join(adj_2.set_index('middles'), on='middles')
    path = path[path.heads != path.tails].values.T

    sec_order_edge_index = np.concatenate([path, np.flip(path, axis=0)], axis=1)
    sec_order_edge_index = torch.from_numpy(sec_order_edge_index).to(device)

    return sec_order_edge_index


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