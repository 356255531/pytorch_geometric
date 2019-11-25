import numpy as np
import os.path as osp
import pickle

from torch_geometric.data.makedirs import cleardir
from .train_eval import single_run, sec_order_single_run


def run(model_class, model_args, dataset_args, train_args):
    # Create new checkpoin folders and clear it
    params = 'hs{}_lr{}_core{}_embdim{}'.format(
        model_args['hidden_size'],
        train_args['lr'], dataset_args['num_core'], model_args['emb_dim'])
    if train_args['model'] == 'GAT' or train_args['model'] == 'PGAT':
        params = 'heads{}_'.format(model_args['heads']) + params
    debug = '' if not train_args['debug'] else '_debug{}'.format(train_args['debug'])
    logger_path = osp.join(train_args['logger_folder'], params + debug)
    weights_path = osp.join(train_args['weights_folder'], params + debug)
    train_args['logger_path'] = logger_path
    train_args['weights_path'] = weights_path
    cleardir(train_args['logger_path'])
    cleardir(train_args['weights_path'])

    kg_train_losses = []
    cf_train_losses = []
    kg_val_losses = []
    cf_val_losses = []
    for run in range(1, train_args['runs'] + 1):
        if train_args['model'] == 'PGAT':
            train_statistics = sec_order_single_run(run, model_class, model_args, dataset_args, train_args)
        else:
            train_statistics = single_run(run, model_class, model_args, dataset_args, train_args)
        if train_args['pretrain']:
            single_run_kg_train_loss, single_run_cf_train_loss, single_run_best_kg_val_loss, single_run_best_cf_val_loss = \
                train_statistics
            kg_train_losses.append(single_run_kg_train_loss)
            kg_val_losses.append(single_run_best_kg_val_loss)
        else:
            single_run_cf_train_loss, single_run_best_cf_val_loss = train_statistics
        cf_train_losses.append(single_run_cf_train_loss)
        cf_val_losses.append(single_run_best_cf_val_loss)

    mean_train_cf_loss, best_train_cf_loss = np.mean(cf_train_losses), np.min(cf_train_losses)
    mean_val_cf_loss, best_val_cf_loss = np.mean(cf_val_losses), np.min(cf_val_losses)
    res_dict = {
                'mean_train_cf_loss': mean_train_cf_loss, 'best_train_cf_loss': best_train_cf_loss,
                'mean_val_cf_loss': mean_val_cf_loss, 'best_val_cf_loss': best_val_cf_loss
    }
    if train_args['kg_pretrain']:
        mean_train_kg_loss, best_train_kg_loss = np.mean(kg_train_losses), np.min(kg_train_losses)
        mean_val_kg_loss, best_val_kg_loss = np.mean(kg_val_losses), np.min(kg_val_losses)
        add_res_dict = {
            'mean_train_kg_loss': mean_train_kg_loss, 'best_train_kg_loss': best_train_kg_loss,
            'mean_val_kg_loss': mean_val_kg_loss, 'best_val_kg_loss': best_val_kg_loss
        }
        res_dict.update(add_res_dict)

    with open(osp.join(train_args['logger_path'], 'res.pkl'), 'wb') as f:
        pickle.dump(res_dict, f)



