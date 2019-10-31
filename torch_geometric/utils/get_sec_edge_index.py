import numpy as np
import scipy
import torch


def check_mat(sparse_mat):
    """
    set diagnal to zero, trucate all value to 1, update coo representation
    :param sparse_mat:
    :return:
    """
    sparse_mat = sparse_mat.tocsr()
    sparse_mat.setdiag(0)
    sparse_mat.eliminate_zeros()
    sparse_mat.sum_duplicates()
    sparse_mat = sparse_mat.tocoo()
    assert (abs(sparse_mat - sparse_mat.T) > 1e-10).nnz == 0
    return sparse_mat


def to_coo_sparse_adj_mat_np(edge_index, m, n):
    adj_mat_row, adj_mat_col = edge_index
    data = np.ones(edge_index.shape[1])
    sparse_adj_mat = scipy.sparse.coo_matrix((data, (adj_mat_row, adj_mat_col)), shape=(m, n))
    return sparse_adj_mat


def get_sec_order_edge(edge_index, x, tensor_type):
    """without self loop"""
    float_tensor, long_tensor, byte_tensor = tensor_type

    # Prepare the input for sparse matrix
    adj_mat_dim = x.shape[0]
    edge_index = edge_index.cpu().detach().numpy()

    # compute two-hop neighbour
    coo_sparse_adj_mat = to_coo_sparse_adj_mat_np(edge_index, adj_mat_dim, adj_mat_dim)
    coo_sparse_adj_mat.setdiag(0)
    coo_sparse_adj_mat.eliminate_zeros()
    coo_sparse_adj_mat.sum_duplicates()
    coo_sparse_adj_mat = coo_sparse_adj_mat.tocsc()
    coo_sparse_adj_mat[coo_sparse_adj_mat > 1] = 1
    coo_sparse_adj_mat = coo_sparse_adj_mat.tocoo()
    adj_mat_row = coo_sparse_adj_mat.row.astype(np.int64)
    adj_mat_col = coo_sparse_adj_mat.col.astype(np.int64)

    coo_sparse_sec_order_adj_mat = coo_sparse_adj_mat.dot(coo_sparse_adj_mat).tocoo().tocsr().tocoo()
    coo_sparse_sec_order_adj_mat.setdiag(0)
    coo_sparse_sec_order_adj_mat.eliminate_zeros()
    sec_order_adj_mat_row = coo_sparse_sec_order_adj_mat.row.astype(np.int64)
    sec_order_adj_mat_col = coo_sparse_sec_order_adj_mat.col.astype(np.int64)
    sec_order_adj_mat_data = coo_sparse_sec_order_adj_mat.data.astype(np.int64)

    num_sec_order_edge = int(np.sum(np.sum(coo_sparse_sec_order_adj_mat)))
    sec_order_edge_index_index = np.zeros(num_sec_order_edge, dtype=np.int64)
    sec_order_mid = np.zeros(num_sec_order_edge, dtype=np.int64)
    sec_order_edge_index_acc = 0
    for i, value in enumerate(sec_order_adj_mat_data):
        sec_order_edge_index_index[sec_order_edge_index_acc: sec_order_edge_index_acc + value] = i
        head = sec_order_adj_mat_row[i]
        tail = sec_order_adj_mat_col[i]
        head_row_adj_mat_value = adj_mat_col[adj_mat_row == head]
        tail_col_adj_mat_value = adj_mat_row[adj_mat_col == tail]
        sec_order_mid[sec_order_edge_index_acc: sec_order_edge_index_acc + value] = \
            np.intersect1d(head_row_adj_mat_value, tail_col_adj_mat_value)
        sec_order_edge_index_acc += value
    sec_order_head = sec_order_adj_mat_row[sec_order_edge_index_index].reshape(1, -1)
    sec_order_mid = sec_order_mid.reshape(1, -1)
    sec_order_tail = sec_order_adj_mat_col[sec_order_edge_index_index].reshape(1, -1)
    sec_order_edge_index = torch.from_numpy(
        np.concatenate((sec_order_head, sec_order_mid, sec_order_tail), axis=0)
    ).type(long_tensor)

    return sec_order_edge_index
