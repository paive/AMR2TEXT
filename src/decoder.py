'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-10 12:02:29
@LastEditTime: 2019-09-02 09:55:36
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import constants as C
from attention import MLPAttention


class DecoderConfig:
    def __init__(self, num_token, emb_dim, hid_dim, coverage, cell_type):
        self.num_token = num_token
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.coverage = coverage
        self.cell_type = cell_type

    def __str__(self):
        return "\tNum token:".ljust(C.PRINT_SPACE) + str(self.num_token) + "\n" + \
               "\tEmb dim".ljust(C.PRINT_SPACE) + str(self.emb_dim) + "\n" + \
               "\tHid dim:".ljust(C.PRINT_SPACE) + str(self.hid_dim) + "\n" + \
               "\tCoverage".ljust(C.PRINT_SPACE) + str(self.coverage) + "\n" + \
               "\tCell".ljust(C.PRINT_SPACE) + str(self.cell_type) + "\n"


def get_rnn_cell(cell_type, input_size, hidden_size):
    if cell_type == 'GRU':
        cell = nn.GRUCell
    elif cell_type == 'LSTM':
        cell = nn.LSTMCell
    return cell(input_size=input_size,
                hidden_size=hidden_size)


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super(Decoder, self).__init__()
        self.config = config

        self.cell = get_rnn_cell(cell_type=self.config.cell_type,
                                 input_size=self.config.emb_dim + self.config.hid_dim,
                                 hidden_size=self.config.hid_dim)

        self.attention = MLPAttention(hid_dim=self.config.hid_dim,
                                      activation='relu',
                                      coverage=self.config.coverage)

    def _step(self, emb, value, value_mask, cov_vec=None, state=None, c1=None):
        # caculate attention
        query = state   # B x H
        context, attn = self.attention(query, value, value_mask, cov_vec)   # B x N
        if cov_vec is not None:
            cov_vec = cov_vec + attn
        inp = torch.cat((emb, context), dim=-1)           # B x (GH+DH)
        if self.config.cell_type == 'GRU':
            state = self.cell(inp, state)
            return state, cov_vec, attn
        elif self.config.cell_type == 'LSTM':
            state, c1 = self.cell(inp, (state, c1))
            return state, c1, cov_vec, attn

    def forward(self, embeddings, value, value_mask, state):
        """
        embeddings:     B x L x H
        value:          B x N x H
        value_mask:     B x N
        state:          B x H
        """
        if self.config.cell_type == 'LSTM':
            c1 = torch.zeros_like(state)

        batch_size, length = embeddings.size()[0:2]
        if self.config.coverage:
            cov_vec = torch.zeros_like(value_mask).float()
        else:
            cov_vec = None

        outputs = []
        attns = []
        for tid in range(0, length):
            if self.config.cell_type == 'GRU':
                state, cov_vec, similarity = self._step(embeddings[:, tid], value, value_mask, cov_vec, state)
            elif self.config.cell_type == 'LSTM':
                state, c1, cov_vec, similarity = self._step(embeddings[:, tid], value, value_mask, cov_vec, state, c1)
            outputs.append(state)
            attns.append(similarity)
        outputs = torch.stack(outputs, dim=1)
        attns = torch.stack(attns, dim=1)
        return outputs, attns


if __name__ == '__main__':
    def test_decoder():
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm
        # torch.autograd.set_detect_anomaly(True)
        config = DecoderConfig(num_token=20000,
                               emb_dim=300,
                               hid_dim=300,
                               enc_dim=360,
                               dropout=0.1,
                               coverage=True,
                               init_param=True)
        tembedder = nn.Embedding(20000, 300)
        decoder = Decoder(config, tembedder).cuda()

        optimizer = optim.Adam(params=decoder.parameters())
        generator = tqdm(range(10000))
        for idx in generator:
            tokens = torch.randint(10000, (16, 20)).cuda()
            mask = torch.ones_like(tokens)
            value = torch.rand(16, 30, 360).cuda()
            value_mask = torch.ones(16, 30).float().cuda()
            state = torch.rand(16, 360).cuda()
            _, loss = decoder(tokens, mask, value, value_mask, state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_decoder()
