'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-22 09:26:02
@LastEditTime: 2019-09-02 09:40:19
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
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
