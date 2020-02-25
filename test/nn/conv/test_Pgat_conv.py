import torch
from torch_geometric.nn import PGATConv


def test_pgat_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = PGATConv(in_channels, out_channels, heads=2, dropout=0.5)
    assert conv.__repr__() == 'PGATConv(16, 32, heads=2)'
    assert conv(x, edge_index).size() == (num_nodes, 2 * out_channels)
    assert conv((x, None), edge_index).size() == (num_nodes, 2 * out_channels)

    conv = PGATConv(in_channels, out_channels, heads=2, concat=False)
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv((x, None), edge_index).size() == (num_nodes, out_channels)


if __name__ == '__main__':
    test_pgat_conv()