from __future__ import division

import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data.makedirs import makedirs

import os
import tqdm
import numpy as np
from torch.optim import Adam


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_iters(data, batch_size=128):
    edge_iter = DataLoader(
        TensorDataset(
            data.edge_index.t()[data.train_edge_mask],
            data.edge_attr[data.train_edge_mask],
        ),
        batch_size=batch_size,
        shuffle=True
    )

    train_rating_edge_iter = DataLoader(
        TensorDataset(
            data.edge_index.t()[data.train_edge_mask * data.rating_edge_mask],
            data.edge_attr[data.train_edge_mask * data.rating_edge_mask],
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_rating_edge_iter = DataLoader(
        TensorDataset(
            data.edge_index.t()[data.test_edge_mask * data.rating_edge_mask],
            data.edge_attr[data.test_edge_mask * data.rating_edge_mask],
        ),
        batch_size=batch_size,
        shuffle=True
    )

    return edge_iter, train_rating_edge_iter, test_rating_edge_iter


def train_kg_single_epoch(epoch, edge_iter, data, loss_func, opt_kg, task_args):
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(edge_iter, total=len(edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        r_idx = edge_attr[:, 0]
        x = data.x
        r_emb = data.r_emb[r_idx]
        r_proj = data.r_proj[r_idx].reshape(-1, task_args['emb_dim'], task_args['repr_dim'])
        proj_head = torch.matmul(x[edge_index[:, :1]], r_proj).reshape(-1, task_args['repr_dim'])
        proj_tail = torch.matmul(x[edge_index[:, 1:2]], r_proj).reshape(-1, task_args['repr_dim'])

        loss_t = loss_func(r_emb + proj_head, proj_tail)

        opt_kg.zero_grad()
        loss_t.backward()
        opt_kg.step()

        losses.append(float(loss_t.detach()))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train KG loss: {:.3f}'.format(epoch, loss))
    return loss


def val_kg_single_epoch(epoch, test_rating_edge_iter, data, loss_func, task_args):
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        r_idx = edge_attr[:, 0]
        x = data.x
        r_emb = data.r_emb[r_idx]
        r_proj = data.r_proj[r_idx].reshape(-1, task_args['emb_dim'], task_args['repr_dim'])
        proj_head = torch.matmul(x[edge_index[:, :1]], r_proj).reshape(-1, task_args['repr_dim'])
        proj_tail = torch.matmul(x[edge_index[:, 1:2]], r_proj).reshape(-1, task_args['repr_dim'])

        loss_t = loss_func(r_emb + proj_head, proj_tail)

        losses.append(float(loss_t.detach()))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train KG loss: {:.3f}'.format(epoch, loss))
    return loss


def train_cf_single_epoch(epoch, model, train_rating_edge_iter, data, loss_func, opt_cf):
    model.training = True
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(
            data.x,
            data.edge_index[:, data.train_edge_mask],
        )
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]

        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        opt_cf.zero_grad()
        loss_t.backward()
        opt_cf.step()

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def eval_cf_single_epoch(epoch, model, test_rating_edge_iter, data, loss_func):
    model.training = False
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(
            data.x,
            data.edge_index[:, data.train_edge_mask],
        )
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]
        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def train_sec_order_cf_single_epoch(
        epoch, model,
        train_rating_edge_iter, train_sec_order_edge_index, data,
        loss_func, opt_cf,
        train_args):
    n_train_sec_order_edge_index = train_sec_order_edge_index.shape[1]

    model.training = True
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        batch_train_sec_order_edge_index = \
            train_sec_order_edge_index[:, np.random.choice(n_train_sec_order_edge_index, train_args['batch_size'])]
        x = model(
            data.x,
            data.edge_index[:, data.train_edge_mask],
            torch.from_numpy(batch_train_sec_order_edge_index)
        )
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]

        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        opt_cf.zero_grad()
        loss_t.backward()
        opt_cf.step()

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def eval_sec_order_cf_single_epoch(
        epoch, model,
        test_rating_edge_iter, train_sec_order_edge_index, data,
        loss_func):
    model.training = False
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(
            data.x,
            data.edge_index[:, data.train_edge_mask],
            torch.from_numpy(train_sec_order_edge_index)
        )
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]
        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def single_run_with_kg(data, model, loss_func, train_args, task_args):
    data.to(device)
    model.to(device).reset_parameters()
    if device == 'cuda':
        data.x = data.x.requires_grad_()
        data.r_proj = data.r_proj.requires_grad_()
        data.r_emb = data.r_emb.requires_grad_()

    edge_iter, train_rating_edge_iter, test_rating_edge_iter = \
        get_iters(data, batch_size=train_args['batch_size'])

    opt_kg = Adam([data.x, data.r_proj, data.r_emb], lr=1e-3, weight_decay=train_args['weight_decay'])

    params = [param for param in model.parameters()]
    opt_cf = Adam(params + [data.x, data.r_proj, data.r_emb], lr=1e-3, weight_decay=train_args['weight_decay'])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    best_kg_val_loss = float('inf')
    best_cf_val_loss = float('inf')
    kg_train_losses = []
    kg_val_losses = []
    cf_train_losses = []
    cf_val_losses = []
    for epoch in range(1, train_args['epochs'] + 1):
        kg_train_loss = train_kg_single_epoch(epoch, edge_iter, data, loss_func, opt_kg, task_args)
        kg_val_loss = val_kg_single_epoch(epoch, test_rating_edge_iter, data, loss_func, task_args)
        cf_train_loss = train_cf_single_epoch(epoch, model, train_rating_edge_iter, data, loss_func, opt_cf)
        cf_val_loss = eval_cf_single_epoch(epoch, model, test_rating_edge_iter, data, loss_func)

        if kg_val_loss < best_kg_val_loss:
            best_kg_val_loss = kg_val_loss
        if cf_val_loss < best_cf_val_loss:
            best_cf_val_loss = cf_val_loss

        eval_info = {'epoch': epoch, 'kg_train_loss': kg_train_loss, 'kg_val_loss': kg_val_loss,
                     'cf_train_loss': cf_train_loss, 'cf_val_loss': cf_val_loss}

        if train_args.get('logger', None) is not None:
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

    weights_path = os.path.expanduser(os.path.normpath(train_args['weights_path']))
    makedirs(weights_path)
    weights_path = os.path.join(weights_path, 'model_weights.pth')
    torch.save(model.state_dict(), weights_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sec_order_single_run_with_kg(data, model, loss_func, train_args, task_args):
    data.to(device)
    model.to(device).reset_parameters()
    if device == 'cuda':
        data.x = data.x.requires_grad_()
        data.r_proj = data.r_proj.requires_grad_()
        data.r_emb = data.r_emb.requires_grad_()

    edge_iter, train_rating_edge_iter, test_rating_edge_iter = \
        get_iters(data, batch_size=train_args['batch_size'])

    opt_kg = Adam([data.x, data.r_proj, data.r_emb], lr=1e-3, weight_decay=train_args['weight_decay'])
    params = [param for param in model.parameters()]
    opt_cf = Adam(params + [data.x, data.r_proj, data.r_emb], lr=1e-3, weight_decay=train_args['weight_decay'])

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
        kg_train_loss = train_kg_single_epoch(epoch, edge_iter, data, loss_func, opt_kg, task_args)
        kg_val_loss = val_kg_single_epoch(epoch, test_rating_edge_iter, data, loss_func, task_args)
        cf_train_loss = train_sec_order_cf_single_epoch(
            epoch,
            model,
            train_rating_edge_iter, train_sec_order_edge_index, data,
            loss_func, opt_cf,
            train_args
        )
        cf_val_loss = eval_sec_order_cf_single_epoch(
            epoch, model,
            test_rating_edge_iter, train_sec_order_edge_index, data,
            loss_func,
        )

        if kg_val_loss < best_kg_val_loss:
            best_kg_val_loss = kg_val_loss
        if cf_val_loss < best_cf_val_loss:
            best_cf_val_loss = cf_val_loss

        eval_info = {'epoch': epoch, 'kg_train_loss': kg_train_loss, 'kg_val_loss': kg_val_loss,
                     'cf_train_loss': cf_train_loss, 'cf_val_loss': cf_val_loss}

        if train_args.get('logger', None) is not None:
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

    weights_path = osp.join(train_args['weights_path'], 'model_weights')
    torch.save(model.state_dict(), weights_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
