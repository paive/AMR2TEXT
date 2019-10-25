'''
@Author: Neo
@Date: 2019-09-02 19:02:52
@LastEditTime: 2019-09-15 17:20:32
'''

import torch
import torch.nn as nn

from utils import get_acti_fun
from attention import MultiHeadAttention
import constants as C


def get_transfomergcn(config):
    gcn = TransformerGCN(
        hid_dim=config.hid_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        directions=config.directions,
        activation=config.activation,
        dropout=config.dropout,
        param_sharing=config.param_sharing)
    return gcn


class TransformerGCNConfig:
    def __init__(self,
                 hid_dim: int,
                 num_layers: int,
                 num_heads: int,
                 directions: int,
                 activation: str,
                 dropout: float,
                 param_sharing: str):
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.directions = directions
        self.activation = activation
        self.dropout = dropout
        self.param_sharing = param_sharing

    def __str__(self):
        return "\tHid dim:".ljust(C.PRINT_SPACE) + str(self.hid_dim) + "\n" + \
               "\tNum layers:".ljust(C.PRINT_SPACE) + str(self.num_layers) + "\n" + \
               "\tNum heads:".ljust(C.PRINT_SPACE) + str(self.num_heads) + "\n" + \
               "\tDirections:".ljust(C.PRINT_SPACE) + str(self.directions) + '\n' + \
               "\tActivation".ljust(C.PRINT_SPACE) + str(self.activation) + "\n" + \
               "\tDropout".ljust(C.PRINT_SPACE) + str(self.dropout) + "\n" + \
               "\tParam sharing".ljust(C.PRINT_SPACE) + str(self.param_sharing) + "\n"


class TransformerGCN(nn.Module):
    def __init__(self,
                 hid_dim,
                 num_layers,
                 num_heads,
                 directions,
                 activation,
                 dropout,
                 param_sharing):
        super(TransformerGCN, self).__init__()
        self._hid_dim = hid_dim
        self._num_heads = num_heads
        self._directions = directions
        self._activation = activation
        self._dropout = dropout

        if param_sharing == 'conv':
            self.conv = GCNConvolution(self._hid_dim, self._num_heads, self._directions, self._dropout, self._activation)
            self._layers = nn.ModuleList([
                Block(hid_dim=self._hid_dim,
                      num_heads=self._num_heads,
                      directions=self._directions,
                      dropout=self._dropout,
                      activation=self._activation,
                      convolution=self.conv) for i in range(num_layers)])
        elif param_sharing == 'inter':
            self.inter = Intermediate(self._hid_dim, self._activation, self._dropout)
            self._layers = nn.ModuleList([
                Block(hid_dim=self._hid_dim,
                      num_heads=self._num_heads,
                      directions=self._directions,
                      dropout=self._dropout,
                      activation=self._activation,
                      intermediate=self.inter) for i in range(num_layers)])
        else:
            assert param_sharing is None
            self._layers = nn.ModuleList([
                Block(hid_dim=self._hid_dim,
                      num_heads=self._num_heads,
                      directions=self._directions,
                      dropout=self._dropout,
                      activation=self._activation) for i in range(num_layers)])
        self.layer_weight = nn.Parameter(torch.zeros(1, num_layers))

    def forward(self, adj, h):
        layer_list = []
        for i, layer in enumerate(self._layers):
            h = layer(adj=adj, inputs=h)
            layer_list.append(h)
        output = torch.stack(layer_list, dim=2)
        weight = torch.softmax(self.layer_weight, dim=-1)
        output = torch.matmul(weight, output).squeeze(2)
        return output


class Intermediate(nn.Module):
    def __init__(self, hid_dim, activation, dropout):
        super(Intermediate, self).__init__()
        self._hid_dim = hid_dim
        self._activation = activation
        self._dropout = dropout

        self.fc1 = nn.Linear(self._hid_dim, 4 * self._hid_dim)
        self.fc1_acti = get_acti_fun(self._activation)
        self.fc2 = nn.Linear(4 * self._hid_dim, self._hid_dim)
        self.fc_dropout = nn.Dropout(self._dropout)
        self.fc_norm = nn.LayerNorm(self._hid_dim)

    def forward(self, hid):
        residual = hid
        output = self.fc1(hid)
        output = self.fc1_acti(output)
        output = self.fc2(output)
        output = self.fc_dropout(output)
        output = self.fc_norm(output + residual)
        return output


class GCNConvolution(nn.Module):
    def __init__(self, hid_dim, num_heads, directions, dropout, activation):
        super(GCNConvolution, self).__init__()
        self._num_heads = num_heads
        self._directions = directions
        self._hid_dim = hid_dim
        self._dropout = dropout
        self._activation = activation

        self.conv_acti = get_acti_fun(self._activation)
        self.conv_norm = nn.LayerNorm(self._hid_dim)
        self.direct_fc = nn.Linear(self._directions * self._hid_dim, self._hid_dim)

    def forward(self, adj, hid):
        residual = hid
        direct_list = []
        for j in range(self._directions):
            label = j + 1
            mask = (adj == label).float()
            weight = mask / (torch.sum(mask, dim=-1, keepdim=True) + C.EPSILON)
            output = torch.matmul(weight, hid)
            direct_list.append(output)
        output = torch.cat(direct_list, dim=-1)
        output = self.direct_fc(output)
        output = self.conv_acti(output)
        output = output + residual
        output = self.conv_norm(output)
        return output


class AttentionConvolution(nn.Module):
    def __init__(self, hid_dim, num_heads, directions, dropout, activation):
        super(AttentionConvolution, self).__init__()
        self._num_heads = num_heads
        self._directions = directions
        self._hid_dim = hid_dim
        self._dropout = dropout
        self._activation = activation

        self.directed_attention = MultiHeadAttention(self._hid_dim, self._hid_dim, self._hid_dim, self._dropout, self._num_heads)
        self.reversed_attention = MultiHeadAttention(self._hid_dim, self._hid_dim, self._hid_dim, self._dropout, self._num_heads)

        self.direct_fc = nn.Linear(3*self._hid_dim, self._hid_dim)
        self.conv_acti = get_acti_fun(self._activation)
        self.conv_norm = nn.LayerNorm(self._hid_dim)

    def forward(self, adj, hid):
        residual = hid
        direct_list = [hid]

        mask = (adj == C.DIRECTED_EDGE_ID) + (adj == C.GLOGAL_EDGE_ID)
        directed_output, similarity = self.directed_attention(hid, hid, mask=mask)
        # weight = mask / (torch.sum(mask, dim=-1, keepdim=True) + C.EPSILON)
        # directed_output = torch.matmul(weight, hid)
        direct_list.append(directed_output)

        mask = (adj == C.REVERSE_EDGE_ID)
        reversed_output, similarity = self.directed_attention(hid, hid, mask=mask)
        # weight = mask / (torch.sum(mask, dim=-1, keepdim=True) + C.EPSILON)
        # reversed_output = torch.matmul(weight, hid)
        direct_list.append(reversed_output)

        output = torch.cat(direct_list, dim=-1)
        output = self.direct_fc(output)
        output = self.conv_acti(output)
        output = output + residual
        output = self.conv_norm(output)
        return output


class GatedConvolution(nn.Module):
    def __init__(self, hid_dim, directions, dropout, activation):
        super(GatedConvolution, self).__init__()
        self._hid_dim = hid_dim
        self._directions = directions
        self._dropout = dropout
        self._activation = activation

        self.direct_fc = nn.Linear(self._directions * self._hid_dim, self._hid_dim)
        self.conv_acti = get_acti_fun(self._activation)

        self.ri = nn.Linear(self._hid_dim, self._hid_dim)
        self.rh = nn.Linear(self._hid_dim, self._hid_dim)

        self.zi = nn.Linear(self._hid_dim, self._hid_dim)
        self.zh = nn.Linear(self._hid_dim, self._hid_dim)

        self.ni = nn.Linear(self._hid_dim, self._hid_dim)
        self.nh = nn.Linear(self._hid_dim, self._hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(self._hid_dim)

    def forward(self, adj, hid):
        # residual = hid
        direct_list = []
        for j in range(self._directions):
            label = j + 1
            mask = (adj == label).float()
            weight = mask / (torch.sum(mask, dim=-1, keepdim=True) + C.EPSILON)
            output = torch.matmul(weight, hid)
            direct_list.append(output)
        output = torch.cat(direct_list, dim=-1)
        output = self.direct_fc(output)
        output = self.conv_acti(output)

        r = self.ri(output) + self.rh(hid)
        r = torch.sigmoid(r)

        z = self.zi(output) + self.zh(hid)
        z = torch.sigmoid(z)

        n = self.ni(output) + r * self.nh(hid)
        n = torch.tanh(n)

        output = (1 - z) * n + z * hid
        output = self.dropout(output)
        output = self.conv_norm(output)
        return output


class Block(nn.Module):
    def __init__(self,
                 hid_dim,
                 num_heads,
                 directions,
                 dropout,
                 activation,
                 convolution=None,
                 intermediate=None):
        super(Block, self).__init__()
        if convolution is None:
            # self.convolution = GCNConvolution(hid_dim, num_heads, directions, dropout, activation)
            self.convolution = AttentionConvolution(hid_dim, num_heads, directions, dropout, activation)
            # self.convolution = GatedConvolution(hid_dim, directions, dropout, activation)
        else:
            self.convolution = convolution

        if intermediate is None:
            self.intermediate = Intermediate(hid_dim, activation, dropout)
        else:
            self.intermediate = intermediate

    def forward(self, adj, inputs):
        h = self.convolution(adj, inputs)
        h = self.intermediate(h)
        return h


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

    config = TransformerGCNConfig(660, 512, 4, 8, 'prelu', 0.1)
    dcgcn = get_transfomergcn(config)

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
