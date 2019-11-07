import numpy as np
import pandas as pd


def get_sec_order_edge(edge_index):
    source, target = edge_index.cpu().detach().numpy()
    not_mask = np.where(source != target)
    source, target = source[not_mask], target[not_mask]
    df1 = pd.DataFrame({'heads': source, 'middles': target})
    df2 = pd.DataFrame({'middles': source, 'tails': target})
    path = df1.join(df2.set_index('middles'), on='middles')
    path = path[path.heads != path.tails].values.T
    return path
