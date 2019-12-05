from __future__ import division

import torch

import tqdm
import numpy as np

from .utils import get_loss_func, get_opt, compute_hr, compute_ndcg


def train_im_cf_single_epoch(epoch, model, train_rating_edge_iter, train_edge_index, train_args):
    loss_func = get_loss_func(train_args['cf_loss'])
    opt = get_opt(train_args['cf_opt'], model, train_args['lr'], train_args['weight_decay'])

    model.training = True
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        u_nids, pos_i_nids, neg_i_nids = batch[0].t()
        u_nids = u_nids.to(train_args['device'])
        pos_i_nids = pos_i_nids.to(train_args['device'])
        neg_i_nids = neg_i_nids.to(train_args['device'])
        x = model(train_edge_index)
        u_emb = x[u_nids]
        pos_i_emb = x[pos_i_nids]
        neg_i_emb = x[neg_i_nids]

        pos_dist = torch.sum(u_emb * pos_i_emb, dim=-1)
        neg_dist = torch.sum(u_emb * neg_i_emb, dim=-1)
        loss_t = - loss_func(pos_dist, neg_dist)

        opt.zero_grad()
        loss_t.backward()
        opt.step()

        losses.append(float(loss_t.detach()))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def train_cf_single_epoch(epoch, model, train_rating_edge_iter, train_edge_index, train_args):
    loss_func = get_loss_func(train_args['cf_loss'])
    opt = get_opt(train_args['cf_opt'], model, train_args['lr'], train_args['weight_decay'])

    model.training = True
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        batch_edge_index_t, batch_edge_attr = batch
        est_rating = model.ex_predict(train_edge_index, batch_edge_index_t.t(), batch_edge_attr)
        rating = batch_edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        opt.zero_grad()
        loss_t.backward()
        opt.step()

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(epoch, loss))
    return loss


def val_im_cf_single_epoch(epoch, model, test_rating_edge_iter, train_edge_index, train_args):
    loss_func = get_loss_func(train_args['cf_loss'])

    model.training = False
    loss = float('inf')
    losses = []
    ndcgs = []
    hrs = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        u_nids, pos_i_nids, neg_i_nids = batch[0].t()
        u_nids = u_nids.to(train_args['device'])
        pos_i_nids = pos_i_nids.to(train_args['device'])
        neg_i_nids = neg_i_nids.to(train_args['device'])
        x = model(train_edge_index)
        u_emb = x[u_nids]
        pos_i_emb = x[pos_i_nids]
        neg_i_emb = x[neg_i_nids]

        pos_dist = torch.sum(u_emb * pos_i_emb, dim=-1)
        neg_dist = torch.sum(u_emb * neg_i_emb, dim=-1)
        target = torch.zeros(batch[0].shape[0]).to(train_args['device'])
        loss_t = loss_func(pos_dist, target) - loss_func(neg_dist, target)
        losses.append(float(loss_t.detach()))
        loss = np.mean(losses)

        # Compute NDCG and HR
        u_nids = test_rating_edge_iter.u_nids
        for u_nid in u_nids:
            test_pos_pairs_df = test_rating_edge_iter.test_pos_pairs_df
            test_pos_pairs_np = test_pos_pairs_df[test_pos_pairs_df.u_nid == u_nid].to_numpy()
            test_neg_pairs_df = test_rating_edge_iter.test_neg_pairs_df
            test_neg_pairs_np = test_neg_pairs_df[test_neg_pairs_df.u_nid == u_nid].to_numpy()
            if test_pos_pairs_np.shape[0] == 0 or test_pos_pairs_np.shape[0] == 0:
                continue
            pos_idx = np.random.choice(range(test_pos_pairs_np.shape[0]), train_args['pos_samples'], replace=True)
            neg_idx = np.random.choice(range(test_neg_pairs_np.shape[0]), train_args['neg_samples'], replace=True)
            test_pos_pair = test_pos_pairs_np[pos_idx]
            test_neg_pair = test_neg_pairs_np[neg_idx]
            u_nids, i_nids = torch.from_numpy(np.concatenate((test_pos_pair, test_neg_pair), axis=0)).t()
            u_emb = x[u_nids]
            i_emb = x[i_nids]
            dist = torch.sum(u_emb * i_emb, dim=-1).cpu().detach().numpy()
            hr = compute_hr(dist, train_args['pos_samples'], train_args['num_recs'])
            ndcg = compute_ndcg(dist, train_args['pos_samples'], train_args['num_recs'])
            hrs.append(hr)
            ndcgs.append(ndcg)
        hr = np.mean(hrs)
        ndcg = np.mean(ndcgs)
        pbar.set_description('Epoch: {}, Val CF loss: {:.3f}, Hit rate: {:.3f}, NDCG: {:.3f}'.format(epoch, loss, hr, ndcg))
    return loss, hr, ndcg


def val_cf_single_epoch(epoch, model, test_rating_edge_iter, train_edge_index, train_args):
    loss_func = get_loss_func(train_args['cf_loss'])

    model.training = False
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        batch_edge_index_t, batch_edge_attr = batch
        est_rating = model.ex_predict(train_edge_index, batch_edge_index_t.t(), batch_edge_attr)
        rating = batch_edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Val CF loss: {:.3f}'.format(epoch, loss))
    return loss


def train_sec_order_cf_single_epoch(
        epoch, model,
        trs_train_rating_edge_iter, train_edge_index, train_sec_order_edge_index,
        train_args):
    n_train_sec_order_edge_index = train_sec_order_edge_index.shape[1]
    loss_func = get_loss_func(train_args['cf_loss'])
    opt = get_opt(train_args['cf_opt'], model, train_args['lr'], train_args['weight_decay'])

    model.training = True
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(trs_train_rating_edge_iter, total=len(trs_train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        batch_train_sec_order_edge_index = \
            train_sec_order_edge_index[:, np.random.choice(n_train_sec_order_edge_index, train_args['sec_order_batch_size'])]
        est_rating = model.ex_predict(
                train_edge_index,
                edge_index,
                torch.from_numpy(batch_train_sec_order_edge_index).to(train_args['device']),
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
        test_rating_edge_iter, train_edge_index, train_sec_order_edge_index,
        train_args):
    n_train_sec_order_edge_index = train_sec_order_edge_index.shape[1]
    loss_func = get_loss_func(train_args['cf_loss'])

    model.training = False
    loss = float('inf')
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        sec_order_edge_index_idx = np.random.choice(n_train_sec_order_edge_index, train_args['sec_order_batch_size'])
        batch_train_sec_order_edge_index = train_sec_order_edge_index[:, sec_order_edge_index_idx]
        est_rating = model.ex_predict(
            train_edge_index,
            edge_index,
            torch.from_numpy(batch_train_sec_order_edge_index).to(train_args['device']),
        )
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss_t = loss_func(est_rating, rating)

        losses.append(np.sqrt(float(loss_t.detach()) * 25))
        loss = np.mean(losses)
        pbar.set_description('Epoch: {}, Val CF loss: {:.3f}'.format(epoch, loss))
    return loss
