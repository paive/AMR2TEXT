'''
@Author: Neo
@Date: 2019-09-02 15:24:04
@LastEditTime: 2019-09-02 19:02:31
'''

import torch.nn as nn
import constants as C
import numpy as np
import torch


class EmbederConfig:
    def __init__(self, num_emb, emb_dim, hid_dim, padding_idx, scale_grad_by_freq, dropout):
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq
        self.dropout = dropout

    def __str__(self):
        return "\tNum emb:".ljust(C.PRINT_SPACE) + str(self.num_emb) + "\n" + \
               "\tEmb dim".ljust(C.PRINT_SPACE) + str(self.emb_dim) + "\n" + \
               "\tHid dim".ljust(C.PRINT_SPACE) + str(self.hid_dim) + "\n" + \
               "\tPad idx:".ljust(C.PRINT_SPACE) + str(self.padding_idx) + "\n" + \
               "\tScale factor".ljust(C.PRINT_SPACE) + str(self.scale_grad_by_freq) + "\n" + \
               "\tDropout:".ljust(C.PRINT_SPACE) + str(self.dropout) + "\n"


class Embeder(nn.Module):
    def __init__(self, config):
        super(Embeder, self).__init__()
        self.config = config
        self.embeder = nn.Embedding(self.config.num_emb,
                                    self.config.emb_dim,
                                    padding_idx=self.config.padding_idx,
                                    scale_grad_by_freq=self.config.scale_grad_by_freq)
        self.matrix = nn.Parameter(torch.Tensor(self.config.emb_dim, self.config.hid_dim))
        nn.init.xavier_uniform_(self.matrix)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, data):
        embedding = self.embeder(data)
        embedding = torch.matmul(embedding, self.matrix)
        embedding = self.dropout(embedding)
        return embedding

    @property
    def embedding_size(self):
        return self.config.emb_dim


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=0):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class PosEmbederConfig:
    def __init__(self, max_seq_len, emb_dim, padding_idx=0):
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx

    def __str__(self):
        return "\tMax seq len:".ljust(C.PRINT_SPACE) + str(self.max_seq_len) + "\n" + \
               "\tEmb dim".ljust(C.PRINT_SPACE) + str(self.emb_dim) + "\n" + \
               "\tPad idx:".ljust(C.PRINT_SPACE) + str(self.padding_idx) + "\n"


class PosEmbeder(nn.Module):
    def __init__(self, config):
        super(PosEmbeder, self).__init__()
        self.config = config
        self.embeder = nn.Embedding.from_pretrained(
            embeddings=get_sinusoid_encoding_table(n_position=self.config.max_seq_len,
                                                   d_hid=self.config.emb_dim,
                                                   padding_idx=self.config.padding_idx),
            padding_idx=self.config.padding_idx,
            freeze=True)

    def forward(self, data):
        embedding = self.embeder(data)
        return embedding

    @property
    def embedding_size(self):
        return self.config.emb_dim


if __name__ == '__main__':
    pass
    # config = EmbederConfig(4, 8, 0, True)
    # embeder = Embeder(config)
    # embedding = embeder(torch.LongTensor([1,2,3,0]))
    # loss = torch.mean(embedding)
    # loss.backward()
    # print(embeder.embeder.weight)
    # print(embeder.embeder.weight.grad)
