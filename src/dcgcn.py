'''
@Author: Neo
@Date: 2019-09-02 19:02:52
@LastEditTime: 2019-09-04 15:35:13
'''

import torch
import torch.nn as nn

from attention import BilinearAttention
from utils import get_acti_fun
import constants as C


def get_dcgcn(config):
    gcn = GCNCell(input_dim=config.input_dim,
                  output_dim=config.output_dim,
                  directions=config.directions,
                  num_layers=config.num_layers,
                  num_heads=config.num_heads,
                  activation=config.activation,
                  dropout=config.dropout)
    return gcn


class DCGCNConfig:
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


class GCNCell(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers,
                 num_heads,
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

        self.aggregate_fc = nn.Linear(num_layers * 2 * self._output_dim, self._output_dim)

    def forward(self, adj, h):
        if self._input_dim != self._output_dim:
            h = self.input_fc(h)

        layer_list = []
        for i, layer in enumerate(self._layers):
            h = layer(adj=adj, inputs=h)
            layer_list.append(h)
        output = torch.cat(layer_list, dim=-1)
        output = self.aggregate_fc(output)
        return output


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
        self._hidden_dim = self._output_dim // self._heads
        self._dropout = dropout
        self._activation = activation

        self.head_ins = nn.ModuleList()
        self.head_in_actis = nn.ModuleList()
        self.head_conv_attns = nn.ModuleList()
        self.head_se_fc = nn.ModuleList()
        self.head_out_actis = nn.ModuleList()
        for i in range(self._heads):
            head_input_dim = self._output_dim + self._hidden_dim * i
            self.head_ins.append(nn.Linear(head_input_dim, self._hidden_dim))
            self.head_in_actis.append(get_acti_fun(self._activation))
            for j in range(self._directions):
                self.head_conv_attns.append(BilinearAttention(self._hidden_dim, 'leakyrelu'))
            self.head_se_fc.append(nn.Linear(self._hidden_dim, 1))
            self.head_out_actis.append(get_acti_fun(self._activation))

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
        outputs = self.dropout_layer(outputs)
        outputs = outputs + inputs
        outputs = self.layer_fc(outputs)
        outputs = self.layer_acti(outputs)
        outputs = self.layer_norm(outputs)
        return outputs

    def _convolve(self, adj, h, i):
        direct_list = []
        output = self.head_ins[i](h)
        output = self.head_in_actis[i](output)

        for j in range(self._directions):
            k = i * self._directions + j
            label = j + 1
            mask = torch.ones_like(adj) * label
            adji = (mask == adj)
            adji_max, _ = torch.max(adji, dim=-1)
            pos = (adji_max == 0)
            pos = torch.diag_embed(pos)
            adji[pos] = 1

            # adji = adji / (torch.sum(adji, dim=-1, keepdim=True))
            # output = torch.matmul(adji, output)
            output, attn = self.head_conv_attns[k](output, output, adji)
            direct_list.append(output)

        output = torch.stack(direct_list, dim=2)        # B x N x D x H
        se = self.head_se_fc[i](output).squeeze(-1)     # B x N x D
        se = torch.softmax(se, dim=-1).unsqueeze(2)     # B x N x 1 x D
        output = torch.matmul(se, output).squeeze(2)    # B x N x H
        output = self.head_out_actis[i](output)
        return output


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
