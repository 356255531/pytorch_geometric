import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from itertools import product

from torch_geometric.datasets import MovieLens
from torch_geometric.datasets import LastFM


def get_dataset(dataset_args):
    ds = dataset_args['dataset']
    if ds == 'movielens':
        return MovieLens(**dataset_args)
    if ds == 'lastfm':
        return LastFM(**dataset_args)
    else:
        raise NotImplemented('{} has not been implemented!'.format(dataset_args['dataset']))


def get_explicit_iters(data, train_args):
    edge_iter = DataLoader(
        TensorDataset(
            data.edge_index[:, data.train_edge_mask].t(),
            data.edge_attr[data.train_edge_mask],
        ),
        batch_size=train_args['kg_batch_size'],
        shuffle=True
    )

    train_rating_edge_iter = DataLoader(
        TensorDataset(
            data.edge_index[:, data.train_edge_mask * data.rating_edge_mask].t(),
            data.edge_attr[data.train_edge_mask * data.rating_edge_mask],
        ),
        batch_size=train_args['cf_batch_size'],
        shuffle=True
    )

    test_rating_edge_iter = DataLoader(
        TensorDataset(
            data.edge_index[:, data.test_edge_mask * data.rating_edge_mask].t(),
            data.edge_attr[data.test_edge_mask * data.rating_edge_mask],
        ),
        batch_size=train_args['cf_batch_size'],
        shuffle=True
    )

    return edge_iter, train_rating_edge_iter, test_rating_edge_iter


def get_implicit_iters(data, train_args):
    # Build edge iterator
    edge_iter = DataLoader(
        TensorDataset(
            data.edge_index[:, data.train_edge_mask].t(),
            data.edge_attr[data.train_edge_mask],
        ),
        batch_size=train_args['kg_batch_size'],
        shuffle=True
    )

    # Build train edge iterator
    import pdb
    pdb.set_trace()
    num_users = data.users[0].shape[0]
    train_ratings = data.ratings[0].iloc[data.train_rating_idx[0]]
    test_ratings = data.ratings[0].iloc[data.test_rating_idx[0]]

    train_pos_pairs_df = pd.DataFrame({'u_nid': train_ratings.uid, 'i_nid': train_ratings.iid + num_users})
    test_pos_pairs_df = pd.DataFrame({'u_nid': test_ratings.uid, 'i_nid': test_ratings.iid + num_users})

    pair_np = np.array([[u_nid, i_nid] for u_nid, i_nid in product(data.users[0].nid, data.items[0].nid)])
    u_nids, i_nids = pair_np[:, 0], pair_np[:, 1]
    all_pairs_df = pd.DataFrame({'u_nid': u_nids, 'i_nid': i_nids})

    train_neg_pairs_df = pd.merge(all_pairs_df, train_pos_pairs_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    test_neg_pairs_df = pd.merge(all_pairs_df, test_pos_pairs_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    train_pos_pairs_df = train_pos_pairs_df.rename(columns={'i_nid': 'pos_i_nid'})
    train_neg_pairs_df = train_neg_pairs_df.rename(columns={'i_nid': 'neg_i_nid'})
    test_pos_pairs_df = test_pos_pairs_df.rename(columns={'i_nid': 'pos_i_nid'})
    test_neg_pairs_df = test_neg_pairs_df.rename(columns={'i_nid': 'neg_i_nid'})

    train_triple_df = pd.merge(train_pos_pairs_df, train_neg_pairs_df, on='u_nid', how='inner')
    test_triple_df = pd.merge(test_pos_pairs_df, test_neg_pairs_df, on='u_nid', how='inner')

    train_rating_edge_iter = DataLoader(
        TensorDataset(
            torch.from_numpy(train_triple_df.to_numpy()).to(train_args['device'])
        ),
        batch_size=train_args['cf_batch_size'],
        shuffle=True
    )

    test_rating_edge_iter = DataLoader(
        TensorDataset(
            torch.from_numpy(test_triple_df.to_numpy()).to(train_args['device'])
        ),
        batch_size=train_args['cf_batch_size'],
        shuffle=True
    )

    return edge_iter, train_rating_edge_iter, test_rating_edge_iter


def get_opt(opt_type, model, lr=10e-3, weight_decay=0):
    if opt_type == 'adam':
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('{} not implemented!'.format(opt_type))

    return opt


def get_loss_func(loss_func_type):
    if loss_func_type == 'mse':
        loss_func = torch.nn.MSELoss()
    else:
        raise ValueError('{} not implemented!'.format(loss_func_type))

    return loss_func
