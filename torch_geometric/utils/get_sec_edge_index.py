import numpy as np
import scipy
import torch
import tqdm


def to_coo_sparse_adj_mat_np(edge_index, m, n):
    adj_mat_row, adj_mat_col = edge_index
    data = np.ones(edge_index.shape[1])
    sparse_adj_mat = scipy.sparse.coo_matrix((data, (adj_mat_row, adj_mat_col)), shape=(m, n))
    return sparse_adj_mat


def get_sec_order_edge(edge_index, x):
    """without self loop"""
    # Prepare the input for sparse matrix
    adj_mat_dim = x.shape[0]
    edge_index = edge_index.cpu().detach().numpy()

    # compute two-hop neighbour
    sparse_adj_mat = to_coo_sparse_adj_mat_np(edge_index, adj_mat_dim, adj_mat_dim)
    adj_mat = np.array(sparse_adj_mat.todense())
    sparse_adj_mat.setdiag(0)
    sparse_adj_mat.eliminate_zeros()
    sparse_adj_mat.sum_duplicates()
    sparse_adj_mat = sparse_adj_mat.tocsc()
    sparse_adj_mat[sparse_adj_mat > 1] = 1

    sparse_sec_order_adj_mat = sparse_adj_mat.dot(sparse_adj_mat).tocoo().tocsr()
    sparse_sec_order_adj_mat.setdiag(0)
    sparse_sec_order_adj_mat.eliminate_zeros()
    sparse_sec_order_adj_mat = sparse_sec_order_adj_mat.tocoo()
    sec_order_adj_mat_row = sparse_sec_order_adj_mat.row.astype(np.int64)
    sec_order_adj_mat_col = sparse_sec_order_adj_mat.col.astype(np.int64)
    sec_order_adj_mat_data = sparse_sec_order_adj_mat.data.astype(np.int64)

    num_sec_order_edge = int(np.sum(np.sum(sparse_sec_order_adj_mat)))
    sec_order_edge_index_index = np.zeros(num_sec_order_edge, dtype=np.int64)
    sec_order_mid = np.zeros(num_sec_order_edge, dtype=np.int64)
    sec_order_edge_index_acc = 0
    pbar = tqdm.tqdm(sec_order_adj_mat_data, total=sec_order_adj_mat_data.shape[0])
    for i, value in enumerate(pbar):
        sec_order_edge_index_index[sec_order_edge_index_acc: sec_order_edge_index_acc + value] = i
        head = sec_order_adj_mat_row[i]
        head_row = adj_mat[head, :].reshape(-1) > 0
        tail = sec_order_adj_mat_col[i]
        tail_col = adj_mat[:, tail].reshape(-1) > 0
        sec_order_mid[sec_order_edge_index_acc: sec_order_edge_index_acc + value] = np.where(head_row & tail_col)[0]
        sec_order_edge_index_acc += value
    sec_order_head = sec_order_adj_mat_row[sec_order_edge_index_index].reshape(1, -1)
    sec_order_mid = sec_order_mid.reshape(1, -1)
    sec_order_tail = sec_order_adj_mat_col[sec_order_edge_index_index].reshape(1, -1)
    sec_order_edge_index = np.concatenate(
        (sec_order_head, sec_order_mid, sec_order_tail),
        axis=0)

    return sec_order_edge_index
