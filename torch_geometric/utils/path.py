import numpy as np
import pandas as pd
from functools import reduce


def filter_path(path):
    return path


def join(edge_index_np, target=None, path_length=2):
    mask = np.where(edge_index_np[0, :] != edge_index_np[1, :])[0]
    edge_index_np = edge_index_np[:, mask]

    if target is not None:
        edge_index_suf_idx = np.isin(edge_index_np[1, :], target)
        path_index_df = pd.DataFrame(
            edge_index_np[:, edge_index_suf_idx].T,
            columns=[str(path_length - 1), str(path_length)]
        )
    else:
        path_index_df = pd.DataFrame(
            edge_index_np.T,
            columns=[str(path_length - 1), str(path_length)]
        )
    for i in range(path_length - 1, 0, -1):
        edge_index_df = pd.DataFrame(
            edge_index_np.T, columns=[str(i - 1), str(i)]
        )
        path_index_df = pd.merge(edge_index_df, path_index_df, on=str(i), how='inner')
        masks = [
            path_index_df.iloc[:, 0] != path_index_df.iloc[:, i]
            for i in range(1, path_index_df.shape[1])
        ]
        mask = reduce(lambda x, y: x * y, masks)
        path_index_df = path_index_df[mask]
    return path_index_df.to_numpy().T

def create_path(edge_index, step_length):
    edge_indices = [edge_index for i in range(step_length)]
    return join(*edge_indices)
