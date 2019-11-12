from __future__ import division

import os.path as osp
import numpy as np
import torch
from torch import tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data.makedirs import makedirs
from torch_geometric.datasets import MovieLens


from .utils import get_iters, cleardir
from .train_eval_kg import train_kg_single_epoch, val_kg_single_epoch
from .train_eval_cf import train_cf_single_epoch, train_sec_order_cf_single_epoch, \
    val_cf_single_epoch, val_sec_order_cf_single_epoch


def single_run_with_kg(run, model_class, model_args, dataset_args, train_args):
    seed = run + 2019
    torch.random.manual_seed(seed)

    dataset_args['seed'] = seed
    data = MovieLens(**dataset_args).data.to(train_args['device'])
    model = model_class(
        model_args['hidden_size'],
        model_args['emb_dim'], model_args['repr_dim'],
        data.num_nodes[0], data.num_relations[0]
    ).to(train_args['device'])
    trs_edge_iter, trs_train_rating_edge_iter, trs_test_rating_edge_iter = \
        get_iters(data, batch_size=train_args['batch_size'])

    filename = 'run{}_seed{}_hs{}_lr{}_core{}_reprdim{}'.format(
        run, seed, model_args['hidden_size'],
    train_args['lr'], dataset_args['num_core'], model_args['repr_dim'])
    debug = '' if not train_args['debug'] else '_debug{}'.format(train_args['debug'])
    logger_path = osp.join(train_args['logger_folder'], filename + debug)
    cleardir(logger_path)
    logger = SummaryWriter(logger_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    kg_train_losses = []
    cf_train_losses = []
    kg_val_losses = []
    cf_val_losses = []
    for epoch in range(1, train_args['epochs'] + 1):
        kg_train_loss = train_kg_single_epoch(epoch, model, trs_edge_iter, train_args)
        kg_val_loss = val_kg_single_epoch(epoch, model, trs_test_rating_edge_iter)
        cf_train_loss = train_cf_single_epoch(
            epoch, model,
            trs_train_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
            train_args)
        cf_val_loss = val_cf_single_epoch(
            epoch, model,
            trs_test_rating_edge_iter, data.edge_index[:, data.train_edge_mask],
            train_args)

        kg_train_losses.append(kg_train_loss)
        kg_val_losses.append(kg_val_loss)
        cf_train_losses.append(cf_train_loss)
        cf_val_losses.append(cf_val_loss)

        # Log the training history
        eval_info = {'kg_train_loss': kg_train_loss, 'kg_val_loss': kg_val_loss,
                     'cf_train_loss': cf_train_loss, 'cf_val_loss': cf_val_loss}
        logger.add_scalars('run: {}'.format(run), eval_info, epoch)

        # Perform early stopping
        early_stopping = train_args.get('early_stopping', None)
        if early_stopping is not None and early_stopping > 0 and epoch > train_args['epochs'] // 2:
            tmp = tensor(cf_val_losses[-(train_args['early_stopping'] + 1):])
            if cf_val_loss > tmp.mean().item():
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Store the weights
    weights_folder = train_args['weights_folder']
    makedirs(weights_folder)
    weights_path = osp.join(weights_folder, filename + '_epoch{}'.format(epoch) + debug + '.pkl')
    torch.save(model.state_dict(), weights_path)

    # Pick the best loss of this run
    kg_train_loss = np.min(kg_train_losses)
    kg_val_loss = np.min(kg_val_losses)
    cf_train_loss = np.min(cf_train_losses)
    cf_val_loss = np.min(cf_val_losses)

    return kg_train_loss, cf_train_loss, kg_val_loss, cf_val_loss


def sec_order_single_run_with_kg(model, data, loss_func, train_args):
    data.to(train_args['device'])
    model.to(train_args['device']).reset_parameters()

    edge_iter, train_rating_edge_iter, test_rating_edge_iter = \
        get_iters(data, batch_size=train_args['batch_size'])

    opt = Adam(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])

    train_sec_order_edge_index = data.train_sec_order_edge_index[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    best_kg_val_loss = float('inf')
    best_cf_val_loss = float('inf')
    kg_train_losses = []
    kg_val_losses = []
    cf_train_losses = []
    cf_val_losses = []
    for epoch in range(1, train_args['epochs'] + 1):
        kg_train_loss = train_kg_single_epoch(epoch, model, edge_iter, opt)
        kg_val_loss = val_kg_single_epoch(epoch, model, test_rating_edge_iter)

        cf_train_loss = train_sec_order_cf_single_epoch(
            epoch,
            model,
            train_rating_edge_iter, train_sec_order_edge_index, data,
            loss_func, opt,
            train_args
        )
        cf_val_loss = val_sec_order_cf_single_epoch(
            epoch, model,
            test_rating_edge_iter, train_sec_order_edge_index, data,
            loss_func, train_args
        )

        if kg_val_loss < best_kg_val_loss:
            best_kg_val_loss = kg_val_loss
        if cf_val_loss < best_cf_val_loss:
            best_cf_val_loss = cf_val_loss

        eval_info = {'epoch': epoch, 'kg_train_loss': kg_train_loss, 'kg_val_loss': kg_val_loss,
                     'cf_train_loss': cf_train_loss, 'cf_val_loss': cf_val_loss}

        if logger is not None:
            train_args['logger'](eval_info)

        kg_train_losses.append(kg_train_loss)
        kg_val_losses.append(kg_val_loss)
        cf_train_losses.append(cf_train_loss)
        cf_val_losses.append(cf_val_loss)

        early_stopping = train_args.get('early_stopping', None)
        if early_stopping is not None and early_stopping > 0 and epoch > train_args['epochs'] // 2:
            tmp = tensor(cf_val_losses[-(train_args['early_stopping'] + 1):])
            if cf_val_loss > tmp.mean().item():
                break

    weights_path = osp.expanduser(osp.normpath(train_args['weights_path']))
    makedirs(weights_path)
    file_name = 'model_weights'
    weights_path = osp.join(weights_path, file_name)
    torch.save(model.state_dict(), weights_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


