from __future__ import division

from os import path as osp
import numpy as np
import os.path as osp
import pickle

import torch
from torch import tensor
from torch.utils.tensorboard import SummaryWriter

from .train_eval_cf import train_im_cf_single_epoch, \
    train_cf_single_epoch, val_cf_single_epoch, val_im_cf_single_epoch,\
    train_sec_order_cf_single_epoch, val_sec_order_cf_single_epoch
from .train_eval_kg import train_kg_single_epoch, val_kg_single_epoch
from .utils import get_dataset, get_explicit_iters, get_implicit_iters

from torch_geometric.data.makedirs import cleardir, makedirs


def run(model_class, dataset_args, model_args, train_args):
    # Create new checkpoin folders and clear it
    model = 'hs{}_emb{}_repr{}'.format(model_args['hidden_size'], model_args['emb_dim'], model_args['repr_dim'])
    model += '_pretrain{}'.format(model_args['pretrain']) if model_args['pretrain'] else ''
    model += '_nodeproj{}'.format(model_args['node_projection']) if model_args['pretrain'] else ''
    if train_args['model'] == 'GAT' or train_args['model'] == 'PGAT':
        model += '_heads{}'.format(model_args['heads'])
    model += '_pretrain{}'.format(model_args['pretrain'])

    train_args['logger_path'] = osp.join(train_args['logger_folder'], model)
    train_args['weights_path'] = osp.join(train_args['weights_folder'], model)
    cleardir(train_args['logger_path'])
    cleardir(train_args['weights_path'])

    if train_args['pretrain']:
        train_kg_losses = []
        val_kg_losses = []
    train_cf_loss = []
    val_cf_losses = []
    for run in range(1, train_args['runs'] + 1):
        if dataset_args['sec_order']:
            if train_args['pretrain']:
                train_kg_losses = []
                val_kg_losses = []
                single_run_kg_train_loss, single_run_cf_train_loss, single_run_kg_val_loss, single_run_cf_val_loss = \
                    sec_order_single_run(run, model_class, model_args, dataset_args, train_args)
                train_kg_losses.append(single_run_kg_train_loss)
                val_kg_losses.append(single_run_kg_val_loss)
            else:
                single_run_cf_train_loss, single_run_cf_val_loss = single_run(run, model_class, model_args, dataset_args, train_args)

            train_statistics = sec_order_single_run(run, model_class, model_args, dataset_args, train_args)
        else:
            if train_args['pretrain']:
                single_run_kg_train_loss, single_run_cf_train_loss, single_run_kg_val_loss, single_run_cf_val_loss = \
                    single_run(run, model_class, model_args, dataset_args, train_args)
                train_kg_losses.append(single_run_kg_train_loss)
                val_kg_losses.append(single_run_kg_val_loss)
            else:
                single_run_cf_train_loss, single_run_cf_val_loss = single_run(run, model_class, model_args, dataset_args, train_args)
        train_cf_loss.append(single_run_cf_train_loss)
        val_cf_losses.append(single_run_cf_val_loss)

    mean_train_cf_loss, best_train_cf_loss = np.mean(train_cf_loss), np.min(train_cf_loss)
    mean_val_cf_loss, best_val_cf_loss = np.mean(val_cf_losses), np.min(val_cf_losses)
    res_dict = {
                'mean_train_cf_loss': best_train_cf_loss, 'best_train_cf_loss': best_train_cf_loss,
                'mean_val_cf_loss': best_val_cf_loss, 'best_val_cf_loss': best_val_cf_loss
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
        print(res_dict)
        pickle.dump(res_dict, f)


def single_run(run, model_class, model_args, dataset_args, train_args):
    seed = run + 2019
    torch.random.manual_seed(seed)

    # Get dataset
    dataset_args['seed'] = seed
    data = get_dataset(dataset_args).data.to(train_args['device'])
    if dataset_args['implicit']:
        trs_edge_iter, trs_train_rating_edge_iter, trs_test_rating_edge_iter = \
            get_implicit_iters(data, dataset_args, train_args)
    else:
        raise NotImplementedError
        # trs_edge_iter, trs_train_rating_edge_iter, trs_test_rating_edge_iter = \
        #     get_explicit_iters(data, train_args)

    # Init model
    model = model_class(data.num_nodes[0], data.num_relations[0], **model_args).to(train_args['device'])
    model.reset_parameters()

    # Init logger
    logger_path = osp.join(train_args['logger_path'], 'run{}seed{}'.format(run, seed))
    makedirs(logger_path)
    logger = SummaryWriter(logger_path)

    # Start training
    cf_train_losses = []
    cf_val_losses = []
    hrs = []
    ndcgs = []
    if train_args['pretrain']:
        kg_train_losses = []
        kg_val_losses = []
        for epoch in range(1, train_args['pretrain_epochs'] + 1):
            kg_train_loss = train_kg_single_epoch(epoch, model, trs_edge_iter, train_args)
            kg_val_loss = val_kg_single_epoch(epoch, model, trs_test_rating_edge_iter)
            kg_train_losses.append(kg_train_loss)
            kg_val_losses.append(kg_val_loss)
    for epoch in range(1, train_args['epochs'] + 1):
        if dataset_args['implicit']:
            cf_train_loss = train_im_cf_single_epoch(
                epoch, model,
                trs_train_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
                train_args)
            cf_val_loss, hr, ndcg = val_im_cf_single_epoch(
                epoch, model,
                trs_test_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
                train_args)
        else:
            raise NotImplementedError
            # cf_train_loss = train_cf_single_epoch(
            #     epoch, model,
            #     trs_train_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
            #     train_args)
            # cf_val_loss = val_cf_single_epoch(
            #     epoch, model,
            #     trs_test_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
            #     train_args)

        cf_train_losses.append(cf_train_loss)
        cf_val_losses.append(cf_val_loss)
        hrs.append(hr)
        ndcgs.append(ndcg)

        # Log the training history
        eval_info = {'cf_train_loss': cf_train_loss, 'cf_val_loss': cf_val_loss, 'hr': hr, 'ndcg': ndcg}
        logger.add_scalars('run{}'.format(run), eval_info, epoch)
        logger.close()

        # Perform early stopping
        early_stopping = train_args.get('early_stopping', None)
        if early_stopping is not None and early_stopping > 0 and epoch > train_args['epochs'] // 2:
            tmp = tensor(hrs[-(train_args['early_stopping'] + 1):])
            if hr > tmp.mean().item():
                print('Early stopped!')
                break

    # Store the weights
    weights_folder = train_args['weights_path']
    makedirs(weights_folder)
    filename = 'run{}_seed{}'.format(run, seed)
    weights_path = osp.join(weights_folder, filename + '_epoch{}'.format(epoch) + '.pkl')
    torch.save(model.state_dict(), weights_path)

    # Pick the best loss of this run
    cf_val_min_idx = np.argmin(hr)[0]

    cf_train_loss = cf_train_losses[cf_val_min_idx]
    cf_val_loss = cf_val_losses[cf_val_min_idx]
    cf_val_hr = hrs[cf_val_min_idx]
    cf_val_ndcg = ndcgs[cf_val_min_idx]

    if train_args['pretrain']:
        kg_val_min_idx = np.argmin(kg_train_losses)

        kg_train_loss = kg_train_losses[kg_val_min_idx]
        kg_val_loss = kg_train_losses[kg_val_min_idx]
        ret = kg_train_loss, cf_train_loss, kg_val_loss, cf_val_loss
    else:
        ret = cf_train_loss, cf_val_loss

    return ret


def path_single_run(run, model_class, model_args, dataset_args, train_args):
    seed = run + 2019
    torch.random.manual_seed(seed)

    # Get dataset
    dataset_args['seed'] = seed
    data = get_dataset(dataset_args).data.to(train_args['device'])
    if dataset_args['implicit']:
        trs_edge_iter, trs_train_rating_edge_iter, trs_test_rating_edge_iter = \
            get_implicit_iters(data, dataset_args, train_args)
    else:
        raise NotImplementedError
        # trs_edge_iter, trs_train_rating_edge_iter, trs_test_rating_edge_iter = \
        #     get_explicit_iters(data, train_args)

    # Init model
    model = model_class(data.num_nodes[0], data.num_relations[0], **model_args).to(train_args['device'])
    model.reset_parameters()

    # Init logger
    logger_path = osp.join(train_args['logger_path'], 'run{}seed{}'.format(run, seed))
    makedirs(logger_path)
    logger = SummaryWriter(logger_path)

    # Start training
    cf_train_losses = []
    cf_val_losses = []
    recalls = []
    ndcgs = []
    if train_args['pretrain']:
        kg_train_losses = []
        kg_val_losses = []
        for epoch in range(1, train_args['pretrain_epochs'] + 1):
            kg_train_loss = train_kg_single_epoch(epoch, model, trs_edge_iter, train_args)
            kg_val_loss = val_kg_single_epoch(epoch, model, trs_test_rating_edge_iter)
            kg_train_losses.append(kg_train_loss)
            kg_val_losses.append(kg_val_loss)
    for epoch in range(1, train_args['epochs'] + 1):
        if dataset_args['implicit']:
            cf_train_loss = train_im_cf_single_epoch(
                epoch, model,
                trs_train_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
                train_args)
            cf_val_loss, recall, ndcg = val_im_cf_single_epoch(
                epoch, model,
                trs_test_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
                train_args)
        else:
            raise NotImplementedError
            # cf_train_loss = train_cf_single_epoch(
            #     epoch, model,
            #     trs_train_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
            #     train_args)
            # cf_val_loss = val_cf_single_epoch(
            #     epoch, model,
            #     trs_test_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
            #     train_args)

        cf_train_losses.append(cf_train_loss)
        cf_val_losses.append(cf_val_loss)
        recalls.append(recall)
        ndcgs.append(ndcg)

        # Log the training history
        eval_info = {'cf_train_loss': cf_train_loss, 'cf_val_loss': cf_val_loss, 'hr': hr, 'ndcg': ndcg}
        logger.add_scalars('run{}'.format(run), eval_info, epoch)
        logger.close()

        # Perform early stopping
        early_stopping = train_args.get('early_stopping', None)
        if early_stopping is not None and early_stopping > 0 and epoch > train_args['epochs'] // 2:
            tmp = tensor(recalls[-(train_args['early_stopping'] + 1):])
            if recall > tmp.mean().item():
                print('Early stopped!')
                break

    # Store the weights
    weights_folder = train_args['weights_path']
    makedirs(weights_folder)
    filename = 'run{}_seed{}'.format(run, seed)
    weights_path = osp.join(weights_folder, filename + '_epoch{}'.format(epoch) + '.pkl')
    torch.save(model.state_dict(), weights_path)

    # Pick the best loss of this run
    cf_val_min_idx = np.argmin(recall)[0]

    cf_train_loss = cf_train_losses[cf_val_min_idx]
    cf_val_loss = cf_val_losses[cf_val_min_idx]
    cf_val_hr = recalls[cf_val_min_idx]
    cf_val_ndcg = ndcgs[cf_val_min_idx]

    if train_args['pretrain']:
        kg_val_min_idx = np.argmin(kg_train_losses)

        kg_train_loss = kg_train_losses[kg_val_min_idx]
        kg_val_loss = kg_train_losses[kg_val_min_idx]
        ret = kg_train_loss, cf_train_loss, kg_val_loss, cf_val_loss
    else:
        ret = cf_train_loss, cf_val_loss

    return ret
