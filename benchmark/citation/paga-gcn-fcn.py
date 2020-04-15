import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import functools

from datasets import get_planetoid_dataset
from train_eval import run, random_planetoid_splits

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):
    def __init__(self, input_channel):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(input_channel, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, input_channel, 1)
        self.score_pool3 = nn.Conv2d(256, input_channel, 1)
        self.score_pool4 = nn.Conv2d(512, input_channel, 1)

        self.upscore2 = nn.ConvTranspose2d(
            input_channel, input_channel, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            input_channel, input_channel, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            input_channel, input_channel, 4, stride=2, bias=False)

        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return self.sigmoid(h)


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.size = dataset.data.x.shape[0]

        self.att_blocks = nn.ModuleList([FCN8s(1)])

        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        for att_block in self.att_blocks:
            att_block.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def path_transform(self, meta_path_edge_indicis, att_block):
        dense_meta_path_adjs = [
            torch.sparse.FloatTensor(
                adj,
                torch.ones(adj.shape[1], device=adj.device),
                torch.Size([self.size, self.size])
            ).to_dense().unsqueeze(0)
            for adj in meta_path_edge_indicis
        ]
        meta_path_tensor = torch.cat(dense_meta_path_adjs, dim=0)

        att = att_block(meta_path_tensor.unsqueeze(0))[0]
        dense_attended_adjs = meta_path_tensor * att

        sparse_adapted_adj = torch.eye(self.size).to_sparse()
        for i in range(dense_attended_adjs.shape[0]):
            sparse_adapted_adj = torch.sparse.mm(sparse_adapted_adj, dense_attended_adjs[i, :, :]).to_sparse()
            edge_index, norm = self.norm(sparse_adapted_adj.indices(), self.size, edge_weight=sparse_adapted_adj.values())
            sparse_adapted_adj = torch.sparse.FloatTensor(
                edge_index,
                norm,
                torch.Size([self.size, self.size])
            ).coalesce()

        return sparse_adapted_adj.indices(), sparse_adapted_adj.values()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        meta_path = [[data.edge_index]]

        att_adjs = [self.path_transform(meta_path, self.att_blocks[meta_path_idx]) for meta_path_idx, meta_path in enumerate(meta_path)]
        att_adjs = [
            torch.sparse.FloatTensor(meta_edge_index, meta_edge_weight, torch.Size([x.shape[0], x.shape[0]]))
            for meta_edge_index, meta_edge_weight in att_adjs
        ]
        adp_adj = functools.reduce(lambda x, y: x + y, att_adjs) / len(att_adjs)
        adp_adj = adp_adj.coalesce()
        adapt_edge_index, adp_edge_weight = adp_adj.indices(), adp_adj.values()

        x = F.relu(self.conv1(x, adapt_edge_index, edge_weight=adp_edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, adapt_edge_index, edge_weight=adp_edge_weight)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
Net(dataset)(dataset.data)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)