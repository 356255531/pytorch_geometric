import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import MovieLens
import os.path as osp
import tqdm
import numpy as np

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

emb_dim = 300
repr_dim = 64
batch_size = 128

root = osp.join('.', 'tmp', 'ml')
dataset = MovieLens(root, '1m', tensor_type, train_ratio=0.8, debug=True, sec_order=True)
data = dataset.data
import pdb
pdb.set_trace()
edge_iter = DataLoader(TensorDataset(data.edge_index.t()[data.train_edge_mask], data.edge_attr[data.train_edge_mask],), batch_size=batch_size, shuffle=True)

loss_func = torch.nn.MSELoss()
opt = torch.optim.SGD([data.x, data.r_proj, data.r_emb], lr=1e-3)

for i in range(20):
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

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(float(loss.detach()))
        pbar.set_description('loss: {}'.format(np.mean(losses)))
