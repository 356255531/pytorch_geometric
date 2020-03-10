import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from ..inits import glorot, zeros


class MeanPathEncoder(object):
    def __init__(self, heads, out_channels):
        self.heads = heads
        self.out_channels = out_channels

    def __call__(self, x, path_index_without_target):
        x_path = x[path_index_without_target.T].mean(dim=1)
        return x_path.view(-1, self.heads, self.out_channels)


class PAGATConv(MessagePassing):
    r"""The path aware graph attention operator

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels, out_channels,
                 path_encoder=None,
                 heads=2, concat=True,
                 negative_slope=0.2,
                 dropout=0.5,
                 bias=True, **kwargs):
        super(PAGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.path_encoder = path_encoder if path_encoder is not None else MeanPathEncoder(heads, out_channels)
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # Transformer encoder
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, path):
        """
        :param x:
        :param path:
        :param size:
        :return:
        """
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            raise AttributeError('x must be tensor!')

        # Remove self loop
        mask = path[0, :] != path[-1, :]
        path = path[:, mask]

        edge_index = path[-2:] if self.flow == 'source_to_target' else path[:2]
        path_index_without_target = path[:-1] if self.flow == 'source_to_target' else path[1:]

        return self.propagate(edge_index, size=x.shape[0], x=x, path_index_without_target=path_index_without_target)

    def message(self, x, size_i, edge_index_i, path_index_without_target):
        # Compute attention coefficients.
        x_path = self.path_encoder(x, path_index_without_target)
        alpha = (x_path * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_path * alpha.view(-1, self.heads, 1), alpha.mean(dim=-1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
