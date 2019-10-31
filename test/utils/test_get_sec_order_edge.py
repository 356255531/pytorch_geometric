import torch
from torch_geometric.utils import get_sec_order_edge
import numpy as np


def test_get_sec_order_edge():
    edge_index = torch.from_numpy(np.array([[0,1,1,1,2,2,3,3], [1,0,2,3,1,3,1,2]]))
    x = np.zeros((4, 1))
    print(get_sec_order_edge(edge_index, x))


if __name__ == '__main__':
    test_get_sec_order_edge()
