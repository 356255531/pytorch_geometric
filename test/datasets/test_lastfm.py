import torch
from torch_geometric.datasets import LastFM
import os.path as osp

torch.random.manual_seed(2019)

emb_dim = 300
repr_dim = 64
batch_size = 128

root = osp.join('.', 'tmp')
dataset = LastFM(root, '2k', train_ratio=0.8, sec_order=False)
data = dataset.data
import pdb
pdb.set_trace()
