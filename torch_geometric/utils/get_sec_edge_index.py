import numpy as np
import pandas as pd


def get_sec_order_edge(edge_index):
    edge_index = edge_index.numpy()
    source, target = edge_index
    sources_dict = defaultdict(list)
    targets = []
    repeated_edge_index = []
    for i, t in enumerate(source):
        sources_dict[t].append(i)

    for i in range(len(target)):
        relevant_edges = sources_dict[target[i].item()]
        two_hop_targets = target[relevant_edges]
        repeated_edge_index.append(np.repeat(edge_index[:, i:i + 1], two_hop_targets.shape[0], axis=1))
        targets.append(two_hop_targets)

    targets = np.concatenate(targets)
    repeated_edge_index = np.concatenate(repeated_edge_index, axis=1)
    return np.concatenate([repeated_edge_index, targets[None, :]], axis=0)

def get_sec_order_edge(edge_index):
    source, target = edge_index.cpu().detach().numpy()
    not_mask = np.where(source != target)
    source, target = source[not_mask], target[not_mask]
    df1 = pd.DataFrame({'heads': source, 'middles': target})
    df2 = pd.DataFrame({'middles': source, 'tails': target})
    path = df1.join(df2.set_index('middles'), on='middles')
    path = path[path.heads != path.tails].values.T
    return path
