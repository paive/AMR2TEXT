import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, hid_dim, num_layers, dropout):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.LSTM(
            hid_dim, hid_dim // 2, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, hid):
        out, _ = self.rnn(hid)
        return out
