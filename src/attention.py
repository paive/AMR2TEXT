'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-22 09:26:02
@LastEditTime: 2019-09-05 11:32:58
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_acti_fun
import constants as C


class MLPAttention(nn.Module):
    def __init__(self, hid_dim, activation, coverage):
        super(MLPAttention, self).__init__()
        self.query_fc = nn.Linear(hid_dim, hid_dim, bias=False)
        self.value_fc = nn.Linear(hid_dim, hid_dim, bias=False)
        if coverage:
            self.coverage_fc = nn.Linear(1, hid_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hid_dim))
        self.acti = get_acti_fun(activation)
        self.mlp = nn.Linear(hid_dim, 1)

    def forward(self, query, value, mask, cov_vec=None):
        """
        query: B x H
        value: B x N x H
        mask : B x N
        cov  : B x N
        """
        attn_hid = self.query_fc(query).unsqueeze(1)
        attn_hid = attn_hid + self.value_fc(value)
        if cov_vec is not None:
            attn_hid = attn_hid + self.coverage_fc(cov_vec.unsqueeze(-1))
        attn_hid = attn_hid + self.bias
        attn_hid = self.acti(attn_hid)
        e = self.mlp(attn_hid).squeeze(-1)          # B x N
        zero_vec = torch.ones_like(e) * C.MINIMUM_VALUE
        attn = torch.where(mask > 0, e, zero_vec)
        attn = torch.softmax(attn, dim=-1)          # B x N
        output = torch.matmul(attn.unsqueeze(1), value).squeeze(1)
        return output, attn


class BilinearAttention(nn.Module):
    def __init__(self, hid_dim, activation):
        super(BilinearAttention, self).__init__()
        self._weight_matrix = nn.Parameter(torch.randn(hid_dim, hid_dim))
        self._activation = get_acti_fun(activation)

    def forward(self, query, value, mask, cov=None):
        """
        query: B x M x H
        value: B x N x H
        mask : B x N x N
        cov  : B x N
        """
        intermediate = torch.matmul(query, self._weight_matrix)
        e = self._activation(intermediate.bmm(value.transpose(1, 2)))
        zero_vec = torch.ones_like(e) * C.MINIMUM_VALUE
        attn = torch.where(mask > 0, e, zero_vec)
        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, value)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        zero_vec = torch.ones_like(attn) * (C.MINIMUM_VALUE)
        attn = torch.where(mask > 0, attn, zero_vec)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim, requires_grad=False).float()

        self.dropout_layer = nn.Dropout(dropout_p)
        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)

    def forward(self, query, keys, mask):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attn = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attn = attn / torch.sqrt(self._key_dim).cuda()

        if mask is not None:
            mask = mask.repeat(self._h, 1, 1)
            zero_vec = torch.ones_like(mask) * C.MINIMUM_VALUE
            attn = torch.where(mask > 0, attn, zero_vec)
            # attn.masked_fill_(~mask, -float('inf'))

        similarity = F.softmax(attn, dim=-1)

        # apply dropout
        drop_sim = self.dropout_layer(similarity)
        # multiplyt it with V
        output = torch.matmul(drop_sim, V)
        # convert attention back to its input original size
        restore_chunk_size = int(output.size(0) / self._h)

        output = torch.cat(
            output.split(split_size=restore_chunk_size, dim=0), dim=2)
        return output, similarity
