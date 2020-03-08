import torch
from torch_geometric.utils.path import join


def test_path():
    x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ]
    )
    y = torch.tensor([
        [4, 5, 7, 4, 5],
        [1, 2, 3, 2, 1]
    ])
    path = join(x, y)
    print(path)
    assert path == 1


if __name__ == '__main__':
    test_path()