import torch
from torch_geometric.nn import SAGEConv


def test_sage_conv():
    in_channels, out_channels = (16, 32)
    edge_index_1 = torch.tensor([[0, 0, 0], [1, 2, 3]])
    edge_index_2 = torch.tensor([[0, 0], [1, 2]])
    num_nodes = edge_index_1.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = SAGEConv(in_channels, out_channels)
    print(conv(x, edge_index_1))
    print(conv(x, edge_index_2))
    assert conv.__repr__() == 'SAGEConv(16, 32)'
    assert conv(x, edge_index_1).size() == (num_nodes, out_channels)
    assert conv((x, None), edge_index_1).size() == (num_nodes, out_channels)


if __name__ == '__main__':
    test_sage_conv()