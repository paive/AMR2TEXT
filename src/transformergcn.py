import torch
import torch.nn as nn

from utils import get_acti_fun
import constants as C
from embeder import RelativePosEmbder
from graphconvolution import get_graph_convolution


def get_transfomergcn(config):
    gcn = TransformerGCN(
        hid_dim=config.hid_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        directions=config.directions,
        activation=config.activation,
        dropout=config.dropout,
        stadia=config.stadia,
        bigcn=config.bigcn)
    return gcn


class TransformerGCNConfig:
    def __init__(self,
                 hid_dim: int,
                 num_layers: int,
                 num_heads: int,
                 directions: int,
                 activation: str,
                 dropout: float,
                 stadia: int,
                 bigcn: bool):
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.directions = directions
        self.activation = activation
        self.dropout = dropout
        self.stadia = stadia
        self.bigcn = bigcn

    def __str__(self):
        return "\tHid dim:".ljust(C.PRINT_SPACE) + str(self.hid_dim) + "\n" + \
               "\tNum layers:".ljust(C.PRINT_SPACE) + str(self.num_layers) + "\n" + \
               "\tNum heads:".ljust(C.PRINT_SPACE) + str(self.num_heads) + "\n" + \
               "\tDirections:".ljust(C.PRINT_SPACE) + str(self.directions) + '\n' + \
               "\tActivation".ljust(C.PRINT_SPACE) + str(self.activation) + "\n" + \
               "\tDropout".ljust(C.PRINT_SPACE) + str(self.dropout) + "\n" + \
               "\tStadia".ljust(C.PRINT_SPACE) + str(self.stadia) + "\n" + \
               "\tBigcn".ljust(C.PRINT_SPACE) + str(self.bigcn) + "\n"


class TransformerGCN(nn.Module):
    def __init__(self,
                 hid_dim,
                 num_layers,
                 num_heads,
                 directions,
                 activation,
                 dropout,
                 stadia,
                 bigcn):
        super(TransformerGCN, self).__init__()
        self._hid_dim = hid_dim
        self._num_heads = num_heads
        self._directions = directions
        self._activation = activation
        self._dropout = dropout
        self._stadia = stadia
        self._conv_name = 'AttentionGCNConv'
        self._bigcn = bigcn

        self._layers = nn.ModuleList([
            Block(hid_dim=self._hid_dim,
                  num_heads=self._num_heads,
                  directions=self._directions,
                  dropout=self._dropout,
                  activation=self._activation,
                  stadia=self._stadia,
                  conv_name=self._conv_name,
                  bigcn=self._bigcn) for i in range(num_layers)])
        self.layer_weight = nn.Parameter(torch.zeros(1, num_layers))

    def forward(self, adj, relative_pos, h):
        layer_list = []
        for i, layer in enumerate(self._layers):
            h = layer(adj=adj, relative_pos=relative_pos, inputs=h)
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


class Block(nn.Module):
    def __init__(self,
                 hid_dim,
                 num_heads,
                 directions,
                 dropout,
                 activation,
                 stadia,
                 conv_name,
                 convolution=None,
                 intermediate=None,
                 bigcn=True):
        super(Block, self).__init__()
        if convolution is None:
            self.convolution = get_graph_convolution(
                conv_name=conv_name,
                hid_dim=hid_dim,
                num_heads=num_heads,
                directions=directions,
                dropout=dropout,
                activation=activation,
                stadia=stadia,
                bigcn=bigcn)
        else:
            self.convolution = convolution

        if intermediate is None:
            self.intermediate = Intermediate(hid_dim, activation, dropout)
        else:
            self.intermediate = intermediate

    def forward(self, adj, relative_pos, inputs):
        h = self.convolution(adj, relative_pos, inputs)
        h = self.intermediate(h)
        return h
