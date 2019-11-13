__model__ = 'PGAT'

import argparse
import torch
import os.path as osp

from torch_geometric.datasets import MovieLens

from benchmark.recsys.utils import get_folder_path
from benchmark.recsys.models import PGATNet
from benchmark.recsys.train_eval import run_with_kg
from benchmark.recsys.models import PGATNet


class PGATRecSys(object):
    def __init__(self, dataset_args, model_args, train_args):
        data = MovieLens(**dataset_args).data.to(train_args['device'])
        model = PGATNet(
                data.num_nodes[0],
                data.num_relations[0],
                **model_args).reset_parameters().to(train_args['device'])
        model.load_state_dict(torch.load(train_args))

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
