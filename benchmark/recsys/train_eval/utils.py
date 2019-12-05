import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from itertools import product
from os.path import join, isfile
import pickle

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


def gen_implicit_triples(data):
    num_users = data.users[0].shape[0]

    ratings = data.ratings[0]
    train_rating_idx = data.train_rating_idx[0]
    test_rating_idx = data.test_rating_idx[0]

    train_ratings = ratings.iloc[train_rating_idx]
    test_ratings = ratings.iloc[test_rating_idx]

    all_pairs = np.array([[u_nid, i_nid] for u_nid, i_nid in product(data.users[0].nid, data.items[0].nid)])
    all_pairs = pd.DataFrame({'u_nid': all_pairs[:, 0], 'i_nid': all_pairs[:, 1]})

    train_pos_pairs_df = pd.DataFrame({'u_nid': train_ratings.uid, 'i_nid': train_ratings.iid + num_users})
    test_pos_pairs_df = pd.DataFrame({'u_nid': test_ratings.uid, 'i_nid': test_ratings.iid + num_users})

    train_neg_pairs_df = pd.merge(all_pairs, train_pos_pairs_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    test_neg_pairs_df = pd.merge(train_neg_pairs_df, test_pos_pairs_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    train_pos_pairs_df = train_pos_pairs_df.rename(columns={'i_nid': 'pos_i_nid'})
    train_neg_pairs_df = train_neg_pairs_df.rename(columns={'i_nid': 'neg_i_nid'})
    test_pos_pairs_df = test_pos_pairs_df.rename(columns={'i_nid': 'pos_i_nid'})
    test_neg_pairs_df = test_neg_pairs_df.rename(columns={'i_nid': 'neg_i_nid'})

    train_triple_df = pd.merge(train_pos_pairs_df, train_neg_pairs_df, on='u_nid', how='inner')
    test_triple_df = pd.merge(test_pos_pairs_df, test_neg_pairs_df, on='u_nid', how='inner')

    return train_triple_df.to_numpy(), test_triple_df.to_numpy(), test_pos_pairs_df, test_neg_pairs_df


def get_implicit_iters(data, data_args, train_args):
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
    root = join(data_args['root'], 'processed')
    if isfile(join(root, 'train_triples.plk')) and isfile(join(root, 'test_triples.plk')) and isfile(join(root, 'test_pos_pairs_df.plk')) and isfile(join(root, 'test_neg_pairs_df.plk')):
        print('Loading pos and neg pair!')
        train_triples = torch.load(join(root, 'train_triples.plk'))
        test_triples = torch.load(join(root, 'test_triples.plk'))
        test_pos_pairs_df = torch.load(join(root, 'test_pos_pairs_df.plk'))
        test_neg_pairs_df = torch.load(join(root, 'test_neg_pairs_df.plk'))
    else:
        print('Creating pos and neg pair!')
        train_triple_np, test_triple_np, test_pos_pairs_df, test_neg_pairs_df = gen_implicit_triples(data)
        train_triples = torch.from_numpy(train_triple_np)
        test_triples = torch.from_numpy(test_triple_np)
        torch.save(train_triples, join(root, 'train_triples.plk'))
        torch.save(test_triples, join(root, 'test_triples.plk'))
        torch.save(test_pos_pairs_df, join(root, 'test_pos_pairs_df.plk'))
        torch.save(test_neg_pairs_df, join(root, 'test_neg_pairs_df.plk'))

    train_rating_edge_iter = DataLoader(
        TensorDataset(train_triples),
        batch_size=train_args['cf_batch_size'],
        shuffle=True
    )

    test_rating_edge_iter = DataLoader(
        TensorDataset(test_triples),
        batch_size=train_args['cf_batch_size'],
        shuffle=True
    )
    test_rating_edge_iter.u_nids = data.users[0].nid.values
    test_rating_edge_iter.test_pos_pairs_df = test_pos_pairs_df
    test_rating_edge_iter.test_neg_pairs_df = test_neg_pairs_df

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


def compute_hr(dist, pos_samples, num_recs):
    indices = np.argsort(dist)[:num_recs]
    hr = np.sum(indices < pos_samples) / num_recs
    return hr


def compute_ndcg(dist, pos_samples, num_recs):
    indices = np.argsort(dist)[:num_recs]
    indices = indices < pos_samples
    if np.sum(indices) == 0:
        return 0
    nonzero_idx = np.nonzero(indices)[0]
    dcg = np.sum(1 / np.log2(nonzero_idx + 2))
    idcg = np.sum(1 / np.log2(np.arange(np.sum(indices)) + 2))
    ndcg = dcg / idcg

    return ndcg
