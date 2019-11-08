import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torch_geometric.datasets import MovieLens
from torch_geometric.nn import GCNConv, PAConv, GATConv

import tqdm
import numpy as np

torch.random.manual_seed(2019)

if torch.cuda.is_available():
    float_tensor = torch.cuda.FloatTensor
    long_tensor = torch.cuda.LongTensor
    bool_tensor = torch.cuda.BoolTensor
    device = 'cuda'
else:
    float_tensor = torch.FloatTensor
    long_tensor = torch.LongTensor
    bool_tensor = torch.BoolTensor
    device = 'cpu'
tensor_type = (float_tensor, bool_tensor, long_tensor)

epochs = 40
emb_dim = 300
repr_dim = 64
kg_batch_size = 1024
cf_batch_size = 1024
sec_order_batch_size = 1024
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1m')
data = MovieLens(path, '1m', tensor_type=tensor_type, train_ratio=0.8, sec_order=True, debug=0.01).data


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
        self.conv2 = PAConv(64, 8, heads=4, dropout=0.6)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def get_attention(self):
        pass

    def forward(self, x, edge_index, sec_order_edge_index):
        '''

        :param x:
        :param edge_index: np.array, [2, N]
        :param sec_order_edge_index: [3, M]
        :return:
        '''
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

train_sec_order_edge_index = data.train_sec_order_edge_index[0]
n_train_sec_order_edge_index = train_sec_order_edge_index.shape[1]

loss_func = torch.nn.MSELoss()
opt_kg = torch.optim.Adam([data.x, data.r_proj, data.r_emb], lr=1e-3)

params = [param for param in model.parameters()]
opt_cf = torch.optim.Adam(params + [data.x, data.r_proj, data.r_emb], lr=1e-3)

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
        pbar.set_description('Epoch: {}, Train KG loss: {:.3f}'.format(i + 1, np.mean(losses)))

    model.training = True
    losses = []
    pbar = tqdm.tqdm(train_rating_edge_iter, total=len(train_rating_edge_iter))
    for batch in pbar:
        edge_index, edge_attr = batch
        batch_train_sec_order_edge_index = \
            train_sec_order_edge_index[:, np.random.choice(n_train_sec_order_edge_index, sec_order_batch_size)]
        x = model(
            data.x,
            data.edge_index[:, data.train_edge_mask],
            torch.from_numpy(batch_train_sec_order_edge_index).type(long_tensor)
        )

        head = x[edge_index[:, 0]]
        tail = x[edge_index[:, 1]]
        est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
        rating = edge_attr[:, 1:2].float().detach() / 5
        loss = loss_func(est_rating, rating)
        opt_cf.zero_grad()
        loss.backward()
        opt_cf.step()

        losses.append(np.sqrt(float(loss.detach()) * 25))
        pbar.set_description('Epoch: {}, Train CF loss: {:.3f}'.format(i + 1, np.mean(losses)))

    model.training = False
    losses = []
    pbar = tqdm.tqdm(test_rating_edge_iter, total=len(test_rating_edge_iter))
    with torch.no_grad():
        for batch in pbar:
            edge_index, edge_attr = batch
            batch_test_sec_order_edge_index = \
                train_sec_order_edge_index[:, np.random.choice(n_train_sec_order_edge_index, sec_order_batch_size)]
            x = model(
                data.x,
                data.edge_index[:, data.train_edge_mask],
                torch.from_numpy(batch_test_sec_order_edge_index).type(long_tensor)
            )
            head = x[edge_index[:, 0]]
            tail = x[edge_index[:, 1]]
            est_rating = torch.sum(head * tail, dim=1).reshape(-1, 1)
            rating = edge_attr[:, 1:2].float().detach() / 5
            loss = loss_func(est_rating, rating)

            losses.append(np.sqrt(float(loss.detach()) * 25))
            pbar.set_description('Epoch: {}, Validation loss: {:.3f}'.format(i + 1, np.mean(losses)))
