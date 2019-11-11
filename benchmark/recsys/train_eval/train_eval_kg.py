from __future__ import division

import tqdm
import numpy as np


def train_kg_single_epoch(epoch, model, edge_iter, opt_kg):
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(edge_iter, total=len(edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        loss_t = model.get_kg_loss(edge_index, edge_attr)

        opt_kg.zero_grad()
        loss_t.backward()
        opt_kg.step()

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

