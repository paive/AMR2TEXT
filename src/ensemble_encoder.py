import torch
import torch.nn as nn


class EnsembleEncoder(nn.Module):
    def __init__(self, gcn_encoder, rnn_encoder, hid_dim):
        super(EnsembleEncoder, self).__init__()
        self.gcn_encoder = gcn_encoder
        self.rnn_encoder = rnn_encoder
        # self.mlp = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, adjs, relative_pos, gh, lh, aligns):
        gv = self.gcn_encoder(adjs, relative_pos, gh)
        lv = self.rnn_encoder(lh)

        # gv: B x N x D
        # lv: B x L x D
        # aligns: B x N
        assert lv.size(1) >= gv.size(1)
        # B x N x D
        aligns = aligns.unsqueeze(-1).repeat(1, 1, gv.size(-1))
        lgv = torch.gather(lv, index=aligns, dim=1)

        # gamma = torch.cat((gv, lgv), dim=-1)
        # gamma = self.mlp(gamma)
        # gamma = torch.sigmoid(gamma)

        # value = gamma * gv + (1. - gamma) * lgv
        value = 0.8 * gv + 0.2 * lgv
        return value
