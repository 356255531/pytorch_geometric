__model__ = 'PGAT'

import numpy as np
import pandas as pd
import torch
import os.path as osp

from torch_geometric.datasets import MovieLens

from .pgat import PGATNetEx


class PGATRecSys(object):
    def __init__(self, num_recs, dataset_args, model_args, train_args):
        self.num_recs = num_recs
        self.train_args = train_args

        self.data = MovieLens(**dataset_args).data.to(train_args['device'])
        model = PGATNetEx(
                self.data.num_nodes[0],
                self.data.num_relations[0],
                **model_args)
        if not dataset_args['debug']:
            model.load_state_dict(torch.load(train_args))
        self.model = model.to(train_args['device'])

    def get_top_n_popular_items(self, n=10):
        """
        TODO: Zhe
        Get the top n movies from self.data.ratings.
        Remove the duplicates in self.data.ratings and sort it by movie count.
        After you find the top N popular movies' item id,
        look over the details information of item in self.data.movies

        :param n: the number of items, int
        :return: df: popular item dataframe, df
        """

        ratings_df = self.data.ratings
        ratings_df = ratings_df.drop_duplicates('iid','first',inplace=True)
        ratings_df = ratings_df.sort_index(axis=0,by='movie_count',ascending=False)

        return ratings_df[:n]

    def build_user(self, iids, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (gender, occupation), tuple
        :return:
        """
        # Build edges for new user
        new_user_nid = self.model.node_emb.weight.shape[0]
        new_user_gender_nid = self.data.gender2nid_map[demographic_info['gender']]
        new_user_occ_nid = self.data.gender2nid_map[demographic_info['occupation']]
        row = [new_user_nid for i in range(len(iids) + 2)]
        col = iids + [new_user_gender_nid, new_user_occ_nid]
        new_edge_index_np = torch.from_numpy(np.array([row, col]))
        new_edge_index = new_edge_index_np.long().to(self.train_args['device'])

        # Build second order edges
        new_edge_index_df = pd.DataFrame({'head': new_edge_index_np[0, :], 'middle': new_edge_index_np[1, :]})
        edge_index_np = self.data.edge_index.numpy()
        edge_index_df = pd.DataFrame({'middle': edge_index_np[0, :], 'tail': edge_index_np[1, :]})
        new_sec_order_edge_df = pd.merge(new_edge_index_df, edge_index_df, on='middle')
        new_sec_order_edge_np = new_sec_order_edge_df.to_numpy().t()
        new_sec_order_edge = torch.from_numpy(new_sec_order_edge_np)

        # Get new user embedding by applying message passing
        node_emb = self.model.node_emb.weight
        new_user_emb = torch.tensor((1, self.model.node_emb.weight.shape[1]))
        node_emb = torch.cat((node_emb, new_user_emb), dim=0)
        self.new_user_emb = self.model(node_emb, new_edge_index, new_sec_order_edge)[-1, :]
        print('user building done...')

    def get_recommendations(self):
        # Estimate the feedback values and get the recommendation
        iids = self.get_top_n_popular_items(self.num_recs + 10)
        rec_iids = self.get_top_n_popular_items(100)
        rec_iids = [iid for iid in rec_iids if rec_iids not in iids]
        rec_nids = [self.data.iid2node_map[iid] for iid in rec_iids]
        rec_item_emb = self.model.node_emb.weight[rec_nids]
        est_feedback = torch.sum(self.new_user_emb * rec_item_emb, dim=1).reshape(-1).numpy()
        rec_iid_idx = np.argsort(est_feedback)[self.num_recs]
        rec_iids = rec_iids[rec_iid_idx]

        df = self.data.items[self.data.items.iid.isin(rec_iids)]

        return df

    # @staticmethod
    # def get_weights_path(dataset_args, model_args, train_args):
    #     params = 'hs{}_lr{}_core{}_reprdim{}'.format(
    #         model_args['hidden_size'],
    #         train_args['lr'], dataset_args['num_core'], model_args['repr_dim'])
    #     if train_args['model'] == 'GAT' or train_args['model'] == 'PGAT':
    #         params = 'heads{}_'.format(model_args['heads']) + params
    #     debug = '' if not train_args['debug'] else '_debug{}'.format(train_args['debug'])
    #     weights_path = osp.join(train_args['weights_folder'], params + debug)
    #     return weights_path