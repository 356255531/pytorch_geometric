import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import GCNConv, PAConv, GATConv

import tqdm
import numpy as np
import itertools
import pdb

torch.random.manual_seed(2019)

if torch.cuda.is_available():
    float_tensor = torch.cuda.FloatTensor
    long_tensor = torch.cuda.LongTensor
    byte_tensor = torch.cuda.ByteTensor
    device = 'cuda'
else:
    float_tensor = torch.FloatTensor
    long_tensor = torch.LongTensor
    byte_tensor = torch.ByteTensor
    device = 'cpu'
tensor_type = (float_tensor, long_tensor, byte_tensor)

epochs = 40
emb_dim = 300
repr_dim = 64
kg_batch_size = 1024
cf_batch_size = 1024
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')
data = MovieLens(path, '1m', tensor_type, train_ratio=0.8).data


def get_sec_order_edge(edge_index, long_tensor):
    heads = edge_index[0, :].cpu().detach().numpy()
    tails = edge_index[1, :].cpu().detach().numpy()

    unique, counts = np.unique(heads, return_counts=True)
    head_cnt_dict = dict(zip(unique, counts))

    n_occ_head_cnt = [head_cnt_dict[tail] for tail in tails.tolist()]
    n_occ_head = np.zeros(np.sum(n_occ_head_cnt))
    idx = 0
    for i, xs in enumerate(n_occ_head_cnt):
        n_occ_head[idx: idx + xs] = i
        idx += xs

    n_occ_tail = [np.where(heads == tail)[0].tolist() for tail in tails.tolist()]
    n_occ_tail = np.array(list(itertools.chain.from_iterable(n_occ_tail)))

    heads = heads[n_occ_head].reshape(1, -1)
    tails = tails[n_occ_tail].reshape(1, -1)

    sec_order_edge_index = np.concatenate((heads, tails), axis=0)
    sec_order_edge_index = torch.from_numpy(sec_order_edge_index).type(long_tensor)

    return sec_order_edge_index


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(300, 128, cached=True)
        self.conv2 = GCNConv(128, 64, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(300, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            64, 64, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class PACNet(torch.nn.Module):
    def __init__(self):
        super(PACNet, self).__init__()
        self.conv1 = GATConv(300, 16, heads=4, dropout=0.6)
        self.conv2 = GCNConv(64, 32, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def get_attention(self):
        pass

    def forward(self, x, edge_index, sec_order_edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, sec_order_edge_index)
        return x


model = PACNet().to(device)

edge_iter = DataLoader(
    TensorDataset(
        data.edge_index.t()[data.train_edge_mask],
        data.edge_attr[data.train_edge_mask],
    ),
    batch_size=kg_batch_size,
    shuffle=True
)

train_rating_edge_iter = DataLoader(
    TensorDataset(
        data.edge_index.t()[data.train_edge_mask * data.rating_edge_mask],
        data.edge_attr[data.train_edge_mask * data.rating_edge_mask],
    ),
    batch_size=cf_batch_size,
    shuffle=True
)

test_rating_edge_iter = DataLoader(
    TensorDataset(
        data.edge_index.t()[data.test_edge_mask * data.rating_edge_mask],
        data.edge_attr[data.test_edge_mask * data.rating_edge_mask],
    ),
    batch_size=cf_batch_size,
    shuffle=True
)

loss_func = torch.nn.MSELoss()
opt_kg = torch.optim.Adam([data.x, data.r_proj, data.r_emb], lr=1e-3)

params = [param for param in model.parameters()]
opt_cf = torch.optim.Adam(params + [data.x, data.r_proj, data.r_emb], lr=1e-3)

for i in range(epochs):
    # losses = []
    # pbar = tqdm.tqdm(edge_iter, total=len(edge_iter))
    # for batch in pbar:
    #     edge_index, edge_attr = batch
    #     r_idx = edge_attr[:, 0]
    #     x = data.x
    #     r_emb = data.r_emb[r_idx]
    #     r_proj = data.r_proj[r_idx].reshape(-1, emb_dim, repr_dim)
    #     proj_head = torch.matmul(x[edge_index[:, :1]], r_proj).reshape(-1, repr_dim)
    #     proj_tail = torch.matmul(x[edge_index[:, 1:2]], r_proj).reshape(-1, repr_dim)
    #
    #     loss = loss_func(r_emb + proj_head, proj_tail)
    #
    #     opt_kg.zero_grad()
    #     loss.backward()
    #     opt_kg.step()
    #
    #     losses.append(float(loss.detach()))
    #     pbar.set_description('Epoch: {}, Train KG loss: {:.3f}'.format(i + 1, np.mean(losses)))

    sec_order_edge_index = get_sec_order_edge(data.edge_index[:, data.train_edge_mask], long_tensor)
    model.training = True
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        x = model(data.x, data.edge_index[:, data.train_edge_mask], sec_order_edge_index)
        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]
        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss = loss_func(est_rating, rating)

        opt_cf.zero_grad()
        loss.backward()
        opt_cf.step()

        losses.append(float(loss.detach()))
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(i + 1, np.mean(losses)))

    sec_order_edge_index = get_sec_order_edge(data.edge_index[:, data.test_edge_mask], long_tensor)
    model.training = False
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    with torch.no_grad():
        for batch in pbar:
            edge_index, edge_attr = batch
            x = model(data.x, data.edge_index[:, data.train_edge_mask], sec_order_edge_index)
            head = x[edge_index[:, 0]]
            tail = x[edge_index[:, 1]]
            est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
            rating = edge_attr[:, 1:2].float().detach() / 5
            loss = loss_func(est_rating, rating)

            losses.append(float(loss.detach()))
            pbar.set_description('Epoch: {}, Validation loss: {:.3f}'.format(i + 1, np.mean(losses)))
