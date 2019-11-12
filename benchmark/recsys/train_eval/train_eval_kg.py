from __future__ import division

import tqdm
import numpy as np

from .utils import get_opt


def train_kg_single_epoch(epoch, model, edge_iter, train_args):
    kg_opt = get_opt(train_args['kg_opt'], model, lr=train_args['lr'], weight_decay=train_args['weight_decay'])

    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(edge_iter, total=len(edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        loss_t = model.get_kg_loss(edge_index, edge_attr)

        kg_opt.zero_grad()
        loss_t.backward()
        kg_opt.step()

        losses.append(float(loss_t.detach()))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train KG loss: {:.3f}'.format(epoch, loss))
    return loss


def val_kg_single_epoch(epoch, kg_model, test_rating_edge_iter):
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        loss_t = kg_model.get_kg_loss(edge_index, edge_attr)

        losses.append(float(loss_t.detach()))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Val KG loss: {:.3f}'.format(epoch, loss))
    return loss

