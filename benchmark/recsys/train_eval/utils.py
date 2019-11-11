from torch.utils.data import TensorDataset, DataLoader


def get_iters(data, batch_size=128):
    edge_iter = DataLoader(
        TensorDataset(
            data.edge_index.t()[data.train_edge_mask],
            data.edge_attr[data.train_edge_mask],
        ),
        batch_size=batch_size,
        shuffle=True
    )

    train_rating_edge_iter = DataLoader(
        TensorDataset(
            data.edge_index.t()[data.train_edge_mask * data.rating_edge_mask],
            data.edge_attr[data.train_edge_mask * data.rating_edge_mask],
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_rating_edge_iter = DataLoader(
        TensorDataset(
            data.edge_index.t()[data.test_edge_mask * data.rating_edge_mask],
            data.edge_attr[data.test_edge_mask * data.rating_edge_mask],
        ),
        batch_size=batch_size,
        shuffle=True
    )

    return edge_iter, train_rating_edge_iter, test_rating_edge_iter
