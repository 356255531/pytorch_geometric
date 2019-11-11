from __future__ import division

import torch

import tqdm
import numpy as np


def train_cf_single_epoch(
        epoch, model,
        train_rating_edge_iter, data,
        loss_func, opt_cf):
    model.training = True
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(
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


def val_cf_single_epoch(epoch, model, test_rating_edge_iter, data, loss_func):
    model.training = False
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(
            data.edge_index[:, data.train_edge_mask],
        )
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]
        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Val CF loss: {:.3f}'.format(epoch, loss))
    return loss


def train_sec_order_cf_single_epoch(
        epoch, model,
        train_rating_edge_iter, train_sec_order_edge_index, data,
        loss_func, opt,
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
        est_rating = model.predict_(
            model(
                data.edge_index[:, data.train_edge_mask],
                torch.from_numpy(batch_train_sec_order_edge_index).to(train_args['device'])),
            edge_index
        )
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        opt.zero_grad()
        loss_t.backward()
        opt.step()

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def val_sec_order_cf_single_epoch(
        epoch, model,
        test_rating_edge_iter, train_sec_order_edge_index, data,
        loss_func,
        train_args):
    n_train_sec_order_edge_index = train_sec_order_edge_index.shape[1]

    model.training = False
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        sec_order_edge_index_idx = np.isin[n_train_sec_order_edge_index[:, 0], edge_index.cpu().numpy()[:, 0]]
        batch_train_sec_order_edge_index = train_sec_order_edge_index[sec_order_edge_index_idx]
        est_rating = model.predict_(
            model(
                data.edge_index[:, data.train_edge_mask],
                batch_train_sec_order_edge_index.to(train_args['device'])),
            edge_index
        )
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Val CF loss: {:.3f}'.format(epoch, loss))
    return loss
