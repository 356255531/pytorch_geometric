import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import GCNConv

import tqdm
import numpy as np

epochs = 20
emb_dim = 300
repr_dim = 64
batch_size = 128
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')
data = MovieLens(path, '1m', train_ratio=0.8).data


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(300, 16, cached=True)
        self.conv2 = GCNConv(16, 64, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)

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

loss_func = torch.nn.MSELoss()
opt_kg = torch.optim.SGD([data.x, data.r_proj, data.r_emb], lr=1e-3)

model_parameters = [param for param in model.parameters()]
opt_cf = torch.optim.Adam(model_parameters+[data.x, data.r_proj, data.r_emb], lr=1e-5)

for i in range(epochs):
    losses = []
    pbar = tqdm.tqdm(edge_iter, total=len(edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        r_idx = edge_attr[:, 0]
        x = data.x
        r_emb = data.r_emb[r_idx]
        r_proj = data.r_proj[r_idx].reshape(-1, emb_dim, repr_dim)
        proj_head = torch.matmul(x[edge_index[:, :1]], r_proj).reshape(-1, repr_dim)
        proj_tail = torch.matmul(x[edge_index[:, 1:2]], r_proj).reshape(-1, repr_dim)

        loss = loss_func(r_emb + proj_head, proj_tail)

        opt_kg.zero_grad()
        loss.backward()
        opt_kg.step()

        losses.append(float(loss.detach()))
        pbar.set_description('Epoch: {}, Train KG loss: {}'.format(i, np.mean(losses)))

    model.training = True
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(data.x, data.edge_index[:, data.train_edge_mask])
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]
        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss = loss_func(est_rating, rating)

        opt_cf.zero_grad()
        loss.backward()
        opt_cf.step()

        losses.append(float(loss.detach()))
        pbar.set_description('Epoch: {}, Train CF loss: {}'.format(i, np.mean(losses)))

    model.training = False
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    with torch.no_grad():
        for batch in pbar:
            edge_index, edge_attr = batch
            x = model(data.x, data.edge_index[:, data.train_edge_mask])
            head = x[edge_index[:, 0]]
            tail = x[edge_index[:, 1]]
            est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
            rating = edge_attr[:, 1:2].float().detach() / 5
            loss = loss_func(est_rating, rating)

            losses.append(float(loss.detach()))
            pbar.set_description('Epoch: {}, Validation loss: {}'.format(i, np.mean(losses)))
