__model__ = 'PGAT'

import argparse
import torch
import os.path as osp

from torch_geometric.datasets import MovieLens

from .pgat import PGATNetEx


class PGATRecSys(object):
    def __init__(self, dataset_args, model_args, train_args):
        self.data = MovieLens(**dataset_args).data.to(train_args['device'])
        model = PGATNetEx(
                self.data.num_nodes[0],
                self.data.num_relations[0],
                **model_args).reset_parameters().to(train_args['device'])
        model.load_state_dict(torch.load(train_args))

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
        new_user_node_id = self.data.node_emb.shape[0]

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

    @staticmethod
    def get_weights_path(dataset_args, model_args, train_args):
        params = 'hs{}_lr{}_core{}_reprdim{}'.format(
            model_args['hidden_size'],
            train_args['lr'], dataset_args['num_core'], model_args['repr_dim'])
        if train_args['model'] == 'GAT' or train_args['model'] == 'PGAT':
            params = 'heads{}_'.format(model_args['heads']) + params
        debug = '' if not train_args['debug'] else '_debug{}'.format(train_args['debug'])
        weights_path = osp.join(train_args['weights_folder'], params + debug)
        return weights_path