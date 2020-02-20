import numpy as np
import pandas as pd
from functools import reduce

def filter_path(path):
    return path


def join(*edge_indices):
    step_length = len(edge_indices) - 1
    source, target = edge_indices[0].cpu().detach().numpy()
    not_mask = np.where(source != target)
    source, target = source[not_mask], target[not_mask]
    df = pd.DataFrame({'0': source, '1': target})
    for i in range(step_length):
        source_, target_ = edge_indices[i + 1].cpu().detach().numpy()
        not_mask_ = np.where(source_ != target_)
        source_, target_ = source_[not_mask_], target_[not_mask_]
        df_ = pd.DataFrame({str(i + 1): source_, str(i + 2): target_})
        df = df.join(df_.set_index(str(i + 1)), on=str(i + 1))
        masks = [df.iloc[:, i] != df.iloc[:, -1] for i in range(df.shape[1] - 1)]
        mask = reduce(lambda x, y: x * y, masks)
        df = df[mask]
    return df.T.values


def create_path(edge_index, step_length):
    edge_indices = [edge_index for i in range(step_length)]
    return join(*edge_indices)
