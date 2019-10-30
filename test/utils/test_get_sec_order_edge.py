import torch
from torch_geometric.utils import get_sec_order_edge
import numpy as np


def test_get_sec_order_edge():
    torch.random.manual_seed(2019)

    if torch.cuda.is_available():
        float_tensor = torch.cuda.FloatTensor
        long_tensor = torch.cuda.LongTensor
        byte_tensor = torch.cuda.ByteTensor
    else:
        float_tensor = torch.FloatTensor
        long_tensor = torch.LongTensor
        byte_tensor = torch.ByteTensor
    tensor_type = (float_tensor, long_tensor, byte_tensor)

    edge_index = torch.from_numpy(np.array([[0,1,1,1,2,2,3,3], [1,0,2,3,1,3,1,2]]))
    x = np.zeros((4, 1))
    print(get_sec_order_edge(edge_index, x, tensor_type))


if __name__ == '__main__':
    test_get_sec_order_edge()
