'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-26 21:58:34
@LastEditTime: 2019-09-02 11:18:27
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn

from attention import BilinearAttention
from utils import get_acti_fun
import constants as C


def get_dcgcn(config):
    gcn = DCGCNModel(input_dim=config.input_dim,
                     output_dim=config.output_dim,
                     num_layers=config.num_layers,
                     activation=config.activation,
                     bidirection=config.bidirection,
                     dropout=config.dropout)
    return gcn


class DCGCNConfig:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation: str,
                 bidirection: bool,
                 dropout: float):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.bidirection = bidirection
        self.dropout = dropout

    def __str__(self):
        return "\tInput dim:".ljust(C.PRINT_SPACE) + str(self.input_dim) + "\n" + \
               "\tOutput dim".ljust(C.PRINT_SPACE) + str(self.output_dim) + "\n" + \
               "\tNum layers:".ljust(C.PRINT_SPACE) + str(self.num_layers) + "\n" + \
               "\tActivation".ljust(C.PRINT_SPACE) + str(self.activation) + "\n" + \
               "\tDropout".ljust(C.PRINT_SPACE) + str(self.dropout) + "\n"


class DCGCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, activation, bidirection, dropout):
        super(DCGCNModel, self).__init__()
        self.bidirection = bidirection
        if self.bidirection:
            self.directed_gcn = GCNCell(input_dim, output_dim//2, num_layers, [1, 3, 4], activation, dropout)
            self.reversed_gcn = GCNCell(input_dim, output_dim//2, num_layers, [2, 3, 4], activation, dropout)
        else:
            self.gcn_cell = GCNCell(input_dim, output_dim, num_layers, [1, 2, 3, 4], activation, dropout)

    def forward(self, adj, h):
        if self.bidirection:
            directed_hid = self.directed_gcn(adj, h)
            reversed_hid = self.reversed_gcn(adj, h)
            h = torch.cat((directed_hid, reversed_hid), dim=-1)
            return h
        else:
            h = self.gcn_cell(adj, h)
            return h


class GCNCell(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers,
                 directions,
                 activation,
                 dropout):
        super(GCNCell, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._layers = nn.ModuleList()

        if self._input_dim != self._output_dim:
            self.input_fc = nn.Linear(self._input_dim, self._output_dim)

        for i in range(num_layers):
            self._layers.append(GraphConvolution(heads=6,
                                                 output_dim=self._output_dim,
                                                 dropout=dropout,
                                                 activation=activation,
                                                 directions=directions))
            self._layers.append(GraphConvolution(heads=3,
                                                 output_dim=self._output_dim,
                                                 dropout=dropout,
                                                 activation=activation,
                                                 directions=directions))

        # self.aggregate_factor = nn.Parameter(torch.zeros(1, num_layers * 2))

    def forward(self, adj, h):
        if self._input_dim != self._output_dim:
            h = self.input_fc(h)

        for i, layer in enumerate(self._layers):
            h = layer(adj=adj, inputs=h)
        return h

    # def forward(self, adj, h):
    #     if self._input_dim != self._output_dim:
    #         h = self.input_fc(h)

    #     layer_list = []
    #     for i, layer in enumerate(self._layers):
    #         h = layer(adj=adj, inputs=h)
    #         layer_list.append(h)
    #     combination = torch.stack(layer_list, dim=2)            # B x N x L x H
    #     layer_weight = torch.softmax(self.aggregate_factor, dim=-1)
    #     combination = torch.matmul(layer_weight, combination).squeeze(2)
    #     return combination


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
        self._hidden_dim = self._output_dim // self._heads
        self._dropout = dropout
        self._activation = activation
        self._directions = directions

        self.head_fcs = nn.ModuleList()
        self.head_attns = nn.ModuleList()
        self.head_actis = nn.ModuleList()
        for i in range(heads):
            head_input_dim = self._output_dim + self._hidden_dim * i
            self.head_fcs.append(nn.Linear(head_input_dim, self._hidden_dim))
            # self.head_attns.append(ScaledDotProductAttention(math.sqrt(head_input_dim), self._dropout))
            self.head_attns.append(BilinearAttention(self._hidden_dim, 'leakyrelu'))
            self.head_actis.append(get_acti_fun(self._activation))

        # Linear transform
        self.layer_fc = nn.Linear(self._output_dim, self._output_dim)
        self.layer_acti = get_acti_fun(self._activation)
        # Linear norm
        self.layer_norm = nn.LayerNorm(self._output_dim)
        # Dropout layer
        self.dropout_layer = nn.Dropout(self._dropout)

    def forward(self, adj, inputs):
        h = inputs
        cache_list = [h]
        for i in range(self._heads):
            convolved = self._convolve(adj, h, i)
            cache_list.append(convolved)
            h = torch.cat(cache_list, dim=2)

        outputs = torch.cat(cache_list[1:], dim=2)
        outputs = self.layer_fc(outputs)
        outputs = self.layer_acti(outputs)
        outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.dropout_layer(outputs)
        return outputs

    def _convolve(self, adj, h, i):
        h = self.head_fcs[i](h)
        mask = torch.zeros_like(adj)
        for direction in self._directions:
            mask = mask + (adj == direction).long()
        h, attn = self.head_attns[i](h, h, mask)
        h = self.head_actis[i](h)
        return h


if __name__ == '__main__':
    from embeder import EmbederConfig
    from embeder import Embeder

    nlabel = torch.LongTensor([[1, 2, 3, 4], [1, 2, 3, 0]])
    npos = torch.LongTensor([[1, 2, 3, 4], [1, 2, 1, 0]])
    adj = torch.LongTensor([[[3, 2, 1, 4], [0, 3, 1, 2], [0, 2, 3, 0], [1, 2, 0, 3]],
                            [[3, 2, 4, 0], [2, 3, 0, 0], [1, 2, 3, 0], [0, 0, 0, 3]]])

    nembedder_config = EmbederConfig(17775, 360, 0, True)
    nembedder = Embeder(nembedder_config)

    pembedder_config = EmbederConfig(200, 300, None, True)
    pembedder = Embeder(pembedder_config)

    config = DCGCNConfig(660, 360, 3, 'relu', False, 0.1)
    dcgcn = get_dcgcn(config)

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
