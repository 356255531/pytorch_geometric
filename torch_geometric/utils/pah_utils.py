import numpy as np
import pandas as pd


def filter_path(path):
    return path


def create_path(edge_index, step_length):
    source, target = edge_index.cpu().detach().numpy()
    not_mask = np.where(source != target)
    source, target = source[not_mask], target[not_mask]
    df = pd.DataFrame({'1': source, '2': target})
    for i in range(step_length):
        df_ = pd.DataFrame({str(i + 2): source, str(i + 3): target})
        df = df.join(df_.set_index(str(i + 2)), on=str(i + 2))
        print(df)
        df.apply(lambda x: x.drop_duplicates(), axis=1)
        df = df.dropna()
    return df.T.values
