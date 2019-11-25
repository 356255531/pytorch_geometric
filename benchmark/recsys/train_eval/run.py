import numpy as np
import os.path as osp
import pickle

from torch_geometric.data.makedirs import cleardir
from .train_eval import single_run, sec_order_single_run


def run(model_class, dataset_args, model_args, train_args):
    # Create new checkpoin folders and clear it
    model = 'hs{}_embdim{}_repr_dim{}_pretrain{}'.format(
        model_args['hidden_size'],
        model_args['emb_dim'], model_args['repr_dim'], model_args['pretrain'])
    if train_args['model'] == 'GAT' or train_args['model'] == 'PGAT':
        model += '_heads{}'.format(model_args['heads'])
    logger_path = osp.join(train_args['logger_folder'], model)
    weights_path = osp.join(train_args['weights_folder'], model)
    train_args['logger_path'] = logger_path
    train_args['weights_path'] = weights_path
    cleardir(train_args['logger_path'])
    cleardir(train_args['weights_path'])

    if train_args['pretrain']:
        train_kg_losses = []
        val_kg_losses = []
    train_cf_loss = []
    val_cf_losses = []
    for run in range(1, train_args['runs'] + 1):
        if dataset_args['sec_order']:
            train_statistics = sec_order_single_run(run, model_class, model_args, dataset_args, train_args)
        else:
            train_statistics = single_run(run, model_class, model_args, dataset_args, train_args)
        if train_args['pretrain']:
            single_run_kg_train_loss, single_run_cf_train_loss, single_run_best_kg_val_loss, single_run_best_cf_val_loss = \
                train_statistics
            train_kg_losses.append(single_run_kg_train_loss)
            val_kg_losses.append(single_run_best_kg_val_loss)
        else:
            single_run_cf_train_loss, single_run_best_cf_val_loss = train_statistics
        train_cf_loss.append(single_run_cf_train_loss)
        val_cf_losses.append(single_run_best_cf_val_loss)

    mean_train_cf_loss, best_train_cf_loss = np.mean(train_cf_loss), np.min(train_cf_loss)
    mean_val_cf_loss, best_val_cf_loss = np.mean(val_cf_losses), np.min(val_cf_losses)
    res_dict = {
                'mean_train_cf_loss': mean_train_cf_loss, 'best_train_cf_loss': best_train_cf_loss,
                'mean_val_cf_loss': mean_val_cf_loss, 'best_val_cf_loss': best_val_cf_loss
    }
    if train_args['pretrain']:
        mean_train_kg_loss, best_train_kg_loss = np.mean(train_kg_losses), np.min(train_kg_losses)
        mean_val_kg_loss, best_val_kg_loss = np.mean(val_kg_losses), np.min(val_kg_losses)
        add_res_dict = {
            'mean_train_kg_loss': mean_train_kg_loss, 'best_train_kg_loss': best_train_kg_loss,
            'mean_val_kg_loss': mean_val_kg_loss, 'best_val_kg_loss': best_val_kg_loss
        }
        res_dict.update(add_res_dict)

    with open(osp.join(train_args['logger_path'], 'res.pkl'), 'wb') as f:
        pickle.dump(res_dict, f)



