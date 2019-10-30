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


def get_sec_order_edge(edge_index, x, tensor_type):
    float_tensor, long_tensor, byte_tensor = tensor_type

    adj_mat_dim = x.shape[0]
    edge_index = edge_index.cpu().detach().numpy()
    adj_mat_row, adj_mat_col = edge_index
    value = np.ones(edge_index.shape[1])
    sparse_adj_mat = scipy.sparse.coo_matrix((value, (adj_mat_row, adj_mat_col)), shape=(adj_mat_dim, adj_mat_dim))
    sparse_adj_mat = check_mat(sparse_adj_mat)
    adj_mat_row, adj_mat_col = sparse_adj_mat.row.astype(np.int64), sparse_adj_mat.col.astype(np.int64)
    adj_mat_data = sparse_adj_mat.data.astype(np.int64)
    adj_mat_data[adj_mat_data > 1] = 1
    sparse_adj_mat = scipy.sparse.coo_matrix((adj_mat_data, (adj_mat_row, adj_mat_col)), shape=(adj_mat_dim, adj_mat_dim))

    sparse_sec_order_adj_mat = sparse_adj_mat.dot(sparse_adj_mat).tocoo().tocsr().tocoo()
    sparse_sec_order_adj_mat = sparse_sec_order_adj_mat.tocsr()
    sparse_sec_order_adj_mat.setdiag(0)
    sparse_sec_order_adj_mat.eliminate_zeros()
    sparse_sec_order_adj_mat = sparse_sec_order_adj_mat.tocoo()
    sec_order_adj_mat_row = sparse_sec_order_adj_mat.row.astype(np.int64)
    sec_order_adj_mat_col = sparse_sec_order_adj_mat.col.astype(np.int64)
    sec_order_mat_adj_value = sparse_sec_order_adj_mat.data.astype(np.int64)

    sec_order_edge_index_index = np.zeros(np.sum(sec_order_mat_adj_value), dtype=np.int64)
    middle = np.zeros(np.sum(sec_order_mat_adj_value), dtype=np.int64)
    sec_order_edge_index_acc = 0
    for i, value in enumerate(sec_order_mat_adj_value):
        sec_order_edge_index_index[sec_order_edge_index_acc: sec_order_edge_index_acc + value] = i
        head = sec_order_adj_mat_row[i]
        tail = sec_order_adj_mat_col[i]
        head_row_adj_mat_value = adj_mat_col[adj_mat_row == head]
        tail_col_adj_mat_value = adj_mat_row[adj_mat_col == tail]
        middle[sec_order_edge_index_acc: sec_order_edge_index_acc + value] = \
            np.intersect1d(head_row_adj_mat_value, tail_col_adj_mat_value)

        sec_order_edge_index_acc += value
    sec_order_row = sec_order_adj_mat_row[sec_order_edge_index_index].reshape(1, -1)
    sec_order_col = sec_order_adj_mat_col[sec_order_edge_index_index].reshape(1, -1)
    sec_order_edge_index = torch.from_numpy(np.concatenate((sec_order_row, sec_order_col), axis=0)).type(long_tensor)
    middle = torch.from_numpy(middle).type(long_tensor)

    return sec_order_edge_index, middle