import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import pandas as pd
import numpy as np

from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(dataset.num_features, args.hidden)
        self.conv3 = GCNConv(2 * args.hidden, dataset.num_classes)
        self.sec_edge_index = None

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        if self.sec_edge_index is None:
            row, col = data.edge_index.cpu().numpy()
            df1 = pd.DataFrame({'head': row, 'middle': col})
            df1 = df1[df1['head'] != df1['middle']]
            df2 = pd.DataFrame({'middle': row, 'tail': col})
            df2 = df2[df2['middle'] != df2['tail']]
            df = pd.merge(df1, df2, on='middle')
            df = df[df['head'] != df['tail']][['head', 'tail']]

            # df3 = pd.DataFrame({'middle2': row, 'tail': col})
            # df3 = df3[df3['middle2'] != df3['tail']]
            # df = pd.merge(df, df3, on='middle2')
            # df = df[df['head'] != df['tail']]
            # df = df[df['middle2'] != df['tail']][['head', 'tail']]

            row = df['head'].to_numpy().reshape(1, -1)
            col = df['tail'].to_numpy().reshape(1, -1)
            self.sec_edge_index = torch.from_numpy(np.concatenate((row, col), axis=0)).to('cuda')
        x, edge_index = data.x, data.edge_index
        x = F.relu(torch.cat((self.conv1(x, edge_index), self.conv2(x, self.sec_edge_index)), dim=-1))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
