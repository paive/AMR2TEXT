import torch
import torch.nn as nn

from attention import MultiHeadAttention
import constants as C
from utils import get_acti_fun
from embeder import RelativePosEmbder


def get_graph_convolution(conv_name, hid_dim, num_heads, directions, dropout, activation, stadia, relative_pos_embder):
    config = GraphConvolutionConfig(hid_dim, num_heads, directions, dropout, activation, stadia)
    if conv_name == 'GCNConv':
        return GCNConvolution(config)
    elif conv_name == 'GatedGCNConv':
        return GatedConvolution(config)
    elif conv_name == 'AttentionGCNConv':
        return AttentionConvolution(config, relative_pos_embder=relative_pos_embder)
    else:
        raise NameError("{} doesn't exists".format(conv_name))


class GraphConvolutionConfig:
    def __init__(self, hid_dim, num_heads, directions, dropout, activation, stadia):
        self._hid_dim = hid_dim
        self._num_heads = num_heads
        self._directions = directions
        self._dropout = dropout
        self._activation = activation
        self._stadia = stadia

    def __str__(self):
        return "\tStadia:".ljust(C.PRINT_SPACE) + str(self.stadia) + "\n" + \
               "\tHid dim".ljust(C.PRINT_SPACE) + str(self.hid_dim) + "\n" + \
               "\tDirections".ljust(C.PRINT_SPACE) + str(self._directions) + "\n" + \
               "\tDropout".ljust(C.PRINT_SPACE) + str(self._dropout) + "\n" + \
               "\tActivation".ljust(C.PRINT_SPACE) + str(self._activation) + "\n"


class GCNConvolution(nn.Module):
    def __init__(self, config):
        super(GCNConvolution, self).__init__()
        self.config = config

        self.conv_acti = get_acti_fun(self.config._activation)
        self.conv_norm = nn.LayerNorm(self.config._hid_dim)
        self.direct_fc = nn.Linear(self.config._directions * self.config._hid_dim, self.config._hid_dim)

    def forward(self, adj, relative_pos, hid):
        residual = hid
        direct_list = []
        for j in range(self.config._directions):
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


class GatedConvolution(nn.Module):
    def __init__(self, config):
        super(GatedConvolution, self).__init__()
        self.config = config

        self.direct_fc = nn.Linear(self.config._directions * self.config._hid_dim, self.config._hid_dim)
        self.conv_acti = get_acti_fun(self.config._activation)

        self.ri = nn.Linear(self.config._hid_dim, self.config._hid_dim)
        self.rh = nn.Linear(self.config._hid_dim, self.config._hid_dim)

        self.zi = nn.Linear(self.config._hid_dim, self.config._hid_dim)
        self.zh = nn.Linear(self.config._hid_dim, self.config._hid_dim)

        self.ni = nn.Linear(self.config._hid_dim, self.config._hid_dim)
        self.nh = nn.Linear(self.config._hid_dim, self.config._hid_dim)

        self.dropout = nn.Dropout(self.config._dropout)
        self.conv_norm = nn.LayerNorm(self.config._hid_dim)

    def forward(self, adj, relative_pos, hid):
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


class AttentionConvolution(nn.Module):
    def __init__(self, config, relative_pos_embder=None):
        super(AttentionConvolution, self).__init__()
        self.config = config

        if relative_pos_embder is None:
            self.relative_pos_embder = RelativePosEmbder(stadia=self.config._stadia)
        else:
            self.relative_pos_embder = relative_pos_embder

        self.attention = MultiHeadAttention(
            query_dim=self.config._hid_dim,
            key_dim=self.config._hid_dim,
            num_units=self.config._hid_dim,
            dropout_p=self.config._dropout,
            h=self.config._num_heads)

        self.conv_acti = get_acti_fun(self.config._activation)
        self.conv_norm = nn.LayerNorm(self.config._hid_dim)

    def forward(self, adj, relative_pos, hid):
        residual = hid

        relative_embedding = self.relative_pos_embder(relative_pos)     # B x N x N x 1
        relative_embedding = relative_embedding.squeeze(-1)

        mask = adj != 0
        output, similarity = self.attention(hid, hid, mask=mask, relative_embedding=relative_embedding)
        output = self.conv_acti(output)
        output = output + residual
        output = self.conv_norm(output)
        return output


# class AttentionConvolution(nn.Module):
#     def __init__(self, config):
#         super(AttentionConvolution, self).__init__()
#         self.config = config

#         self.directed_attention = MultiHeadAttention(
#             self.config._hid_dim, self.config._hid_dim, self.config._hid_dim, self.config._dropout, self.config._num_heads)
#         self.reversed_attention = MultiHeadAttention(
#             self.config._hid_dim, self.config._hid_dim, self.config._hid_dim, self.config._dropout, self.config._num_heads)

#         self.direct_fc = nn.Linear(3*self.config._hid_dim, self.config._hid_dim)
#         self.conv_acti = get_acti_fun(self.config._activation)
#         self.conv_norm = nn.LayerNorm(self.config._hid_dim)

#     def forward(self, adj, relative_pos, hid):
#         residual = hid
#         direct_list = [hid]

#         mask = (adj == C.DIRECTED_EDGE_ID) + (adj == C.GLOGAL_EDGE_ID)
#         directed_output, similarity = self.directed_attention(hid, hid, mask=mask)
#         # weight = mask / (torch.sum(mask, dim=-1, keepdim=True) + C.EPSILON)
#         # directed_output = torch.matmul(weight, hid)
#         direct_list.append(directed_output)

#         mask = (adj == C.REVERSE_EDGE_ID)
#         reversed_output, similarity = self.directed_attention(hid, hid, mask=mask)
#         # weight = mask / (torch.sum(mask, dim=-1, keepdim=True) + C.EPSILON)
#         # reversed_output = torch.matmul(weight, hid)
#         direct_list.append(reversed_output)

#         output = torch.cat(direct_list, dim=-1)
#         output = self.direct_fc(output)
#         output = self.conv_acti(output)
#         output = output + residual
#         output = self.conv_norm(output)
#         return output
