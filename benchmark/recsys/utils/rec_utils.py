import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score


def hit(hit_vec_np):
    if np.sum(hit_vec_np) > 0:
        return 1
    else:
        return 0


def ndcg(hit_vec_np):
    num_recs = hit_vec_np.shape[0]
    return ndcg_score(hit_vec_np.reshape(1, -1), np.ones((1, num_recs)))


def auc(hit_vec_np):
    num_recs = hit_vec_np.shape[0]
    return roc_auc_score(hit_vec_np, np.ones(num_recs))

