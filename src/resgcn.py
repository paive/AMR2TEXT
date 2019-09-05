'''
@Author: Neo
@Date: 2019-09-02 19:02:52
@LastEditTime: 2019-09-04 19:59:16
'''

import torch
import torch.nn as nn

from attention import BilinearAttention
from utils import get_acti_fun
import constants as C


def get_resgcn(config):
    gcn = ResGCNCell(input_dim=config.input_dim,
                     output_dim=config.output_dim,
                     directions=config.directions,
                     num_layers=config.num_layers,
                     num_heads=config.num_heads,
                     activation=config.activation,
                     dropout=config.dropout)
    return gcn


class ResDCGCNConfig:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 directions: int,
                 num_layers: int,
                 num_heads: int,
                 activation: str,
                 dropout: float):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.directions = directions
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout

    def __str__(self):
        return "\tInput dim:".ljust(C.PRINT_SPACE) + str(self.input_dim) + "\n" + \
               "\tOutput dim".ljust(C.PRINT_SPACE) + str(self.output_dim) + "\n" + \
               "\tNum layers:".ljust(C.PRINT_SPACE) + str(self.num_layers) + "\n" + \
               "\tNum heads".ljust(C.PRINT_SPACE) + str(self.num_heads) + "\n" + \
               "\tActivation".ljust(C.PRINT_SPACE) + str(self.activation) + "\n" + \
               "\tDropout".ljust(C.PRINT_SPACE) + str(self.dropout) + "\n"


class ResGCNCell(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers,
                 num_heads,
                 directions,
                 activation,
                 dropout):
        super(ResGCNCell, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._layers = nn.ModuleList()

        if self._input_dim != self._output_dim:
            self.input_fc = nn.Linear(self._input_dim, self._output_dim)

        for i in range(num_layers):
            self._layers.append(GraphConvolution(heads=num_heads[0],
                                                 output_dim=self._output_dim,
                                                 dropout=dropout,
                                                 activation=activation,
                                                 directions=directions))
            self._layers.append(GraphConvolution(heads=num_heads[1],
                                                 output_dim=self._output_dim,
                                                 dropout=dropout,
                                                 activation=activation,
                                                 directions=directions))

    def forward(self, adj, h):
        if self._input_dim != self._output_dim:
            h = self.input_fc(h)

        for i, layer in enumerate(self._layers):
            h = layer(adj=adj, inputs=h)
        return h


class GraphConvolution(nn.Module):
    def __init__(self,
                 heads,
                 output_dim,
                 dropout,
                 activation,
                 directions):
        super(GraphConvolution, self).__init__()
        self._heads = heads
        self._output_dim = output_dim
        self._directions = directions
        self._dropout = dropout
        self._activation = activation

        self.head_conv_attns = nn.ModuleList()
        self.head_conv_aggres = nn.ParameterList()
        self.head_conv_actis = nn.ModuleList()
        self.head_conv_norms = nn.ModuleList()

        self.head_fcs = nn.ModuleList()
        self.head_actis = nn.ModuleList()
        self.head_norms = nn.ModuleList()

        self.head_dropout = nn.ModuleList()

        for i in range(self._heads):
            self.head_conv_attns.append(BilinearAttention(self._output_dim, 'leakyrelu'))
            self.head_conv_aggres.append(nn.Parameter(torch.zeros(1, self._directions)))
            self.head_conv_actis.append(get_acti_fun(self._activation))
            self.head_conv_norms.append(nn.LayerNorm(self._output_dim))

            self.head_fcs.append(nn.Linear(self._output_dim, self._output_dim))
            self.head_actis.append(get_acti_fun(self._activation))
            self.head_norms.append(nn.LayerNorm(self._output_dim))

            self.head_dropout.append(nn.Dropout(self._dropout))

    def forward(self, adj, inputs):
        h = inputs
        for i in range(self._heads):
            h = self._convolve(adj, h, i)
            h = self._fc(h, i)
            h = self.head_dropout[i](h)
        h = h + inputs
        return h

    def _fc(self, h, i):
        h = self.head_fcs[i](h)
        h = self.head_actis[i](h)
        h = self.head_norms[i](h)
        return h

    def _convolve(self, adj, h, i):
        direct_list = []
        for j in range(self._directions):
            label = j + 1
            mask = torch.ones_like(adj) * label
            adji = (mask == adj)
            adji_max, _ = torch.max(adji, dim=-1)
            pos = (adji_max == 0)
            pos = torch.diag_embed(pos)
            adji[pos] = 1
            h, attn = self.head_conv_attns[i](h, h, adji)
            direct_list.append(h)

        output = torch.stack(direct_list, dim=2)        # B x N x D x H
        weight = torch.softmax(self.head_conv_aggres[i], dim=-1)    # B x 1 x D
        output = torch.matmul(weight, output).squeeze(2)    # B x N x H
        output = self.head_conv_actis[i](output)
        output = self.head_conv_norms[i](output)
        return output


if __name__ == '__main__':
    from embeder import EmbederConfig
    from embeder import Embeder

    nlabel = torch.LongTensor([[1, 2, 3, 4], [1, 2, 3, 0]])
    npos = torch.LongTensor([[1, 2, 3, 4], [1, 2, 1, 0]])
    adj = torch.LongTensor([[[3, 2, 1, 4], [0, 3, 1, 2], [0, 2, 3, 0], [1, 2, 0, 3]],
                            [[3, 2, 4, 0], [2, 3, 0, 0], [1, 2, 3, 0], [0, 0, 0, 3]]])

    nembedder_config = EmbederConfig(17775, 360, 0, True, 0.5)
    nembedder = Embeder(nembedder_config)

    pembedder_config = EmbederConfig(200, 300, None, True, 0.5)
    pembedder = Embeder(pembedder_config)

    config = ResDCGCNConfig(660, 360, 4, 4, [3, 2], 'relu', 0.1)
    dcgcn = get_resgcn(config)

    optimizer = torch.optim.Adam([{"params": nembedder.parameters()},
                                  {"params": pembedder.parameters()},
                                  {"params": dcgcn.parameters()}])

    inputs = torch.cat((nembedder(nlabel), pembedder(npos)), dim=-1)
    print(adj.size())
    print(inputs.size())
    out = dcgcn(adj, inputs)
    print(out.size())
    print(out)
    print(adj)
