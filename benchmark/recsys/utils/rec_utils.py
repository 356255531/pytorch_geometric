import numpy as np
import torch
import tqdm
import pandas as pd


def hit(hit_vec):
    if hit_vec.sum() > 0:
        return 1
    else:
        return 0


def ndcg(hit_vec):
    ndcg_vec = [np.reciprocal(np.log2(idx+2)) for idx, if_hit in enumerate(hit_vec.cpu().numpy()) if if_hit]
    return np.sum(ndcg_vec)


def metrics(
        epoch,
        model, x, meta_paths,
        test_pos_unid_inid_map, neg_unid_inid_map,
        rec_args):
    HR, NDCG, losses = [], [], []
    propagated_node_emb = model(x, meta_paths)

    u_nids = list(test_pos_unid_inid_map.keys())
    test_bar = tqdm.tqdm(u_nids, total=len(u_nids))
    for u_idx, u_nid in enumerate(test_bar):
        pos_i_nids = test_pos_unid_inid_map[u_nid]
        neg_i_nids = neg_unid_inid_map[u_nid]
        if len(pos_i_nids) == 0 or len(neg_i_nids) == 0:
            continue
        pos_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
        neg_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(neg_i_nids))], 'neg_i_nid': neg_i_nids})
        pos_neg_pair_np = pd.merge(pos_i_nid_df, neg_i_nid_df, how='inner', on='u_nid').to_numpy()

        u_node_emb = propagated_node_emb[pos_neg_pair_np[:, 0]]
        pos_i_node_emb = propagated_node_emb[pos_neg_pair_np[:, 1]]
        neg_i_node_emb = propagated_node_emb[pos_neg_pair_np[:, 2]]
        pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
        pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)

        loss = - (pred_pos - pred_neg).sigmoid().log().mean().item()

        u_node_emb = propagated_node_emb[u_nid]
        pos_i_node_emb = propagated_node_emb[pos_i_nids]
        neg_i_node_emb = propagated_node_emb[neg_i_nids]
        pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
        pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)

        _, indices = torch.topk(torch.cat([pred_pos, pred_neg]), rec_args['num_recs'])
        hit_vec = indices < len(pos_i_nids)

        HR.append(hit(hit_vec))
        NDCG.append(ndcg(hit_vec))
        losses.append(loss)
        test_bar.set_description('Epoch: {:.4f} loss: {:.4f} HR: {:.4f} NDCG: {:.4f}'.format(epoch, np.mean(losses), np.mean(HR), np.mean(NDCG)))

    return np.mean(HR), np.mean(NDCG), np.mean(losses)