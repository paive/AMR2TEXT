import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, hid_dim, num_layers, dropout):
        super(RNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.rnn = nn.LSTM(
            hid_dim, hid_dim // 2, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, hid):
        out, state = self.rnn(hid)
        # state = (state[0].transpose(0, 1).contiguous().view(-1, self.num_layers, self.hid_dim),
        #          state[1].transpose(0, 1).contiguous().view(-1, self.num_layers, self.hid_dim))
        # state = (state[0][:, -1, :], state[1][:, -1, :])
        return out
