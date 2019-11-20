import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

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


def get_iters(data, train_args):
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
