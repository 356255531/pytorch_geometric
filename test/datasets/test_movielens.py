import torch
from torch_geometric.datasets import MovieLens
import os.path as osp

torch.random.manual_seed(2019)

if torch.cuda.is_available():
    float_tensor = torch.cuda.FloatTensor
    long_tensor = torch.cuda.LongTensor
    byte_tensor = torch.cuda.ByteTensor
else:
    float_tensor = torch.FloatTensor
    long_tensor = torch.LongTensor
    byte_tensor = torch.ByteTensor
tensor_type = (float_tensor, byte_tensor, long_tensor)

emb_dim = 300
repr_dim = 64
batch_size = 128

root = osp.join('.', 'tmp', 'ml')
dataset = MovieLens(root, '1m', tensor_type, train_ratio=0.8, debug=True, sec_order=True)
data = dataset.data

import pdb
pdb.set_trace()
