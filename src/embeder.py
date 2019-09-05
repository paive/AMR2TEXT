'''
@Author: Neo
@Date: 2019-09-02 15:24:04
@LastEditTime: 2019-09-02 19:02:31
'''

import torch.nn as nn
import constants as C


class EmbederConfig:
    def __init__(self, num_emb, emb_dim, padding_idx, scale_grad_by_freq, dropout):
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq
        self.dropout = dropout

    def __str__(self):
        return "\tNum emb:".ljust(C.PRINT_SPACE) + str(self.num_emb) + "\n" + \
               "\tEmb dim".ljust(C.PRINT_SPACE) + str(self.emb_dim) + "\n" + \
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
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, data):
        embedding = self.embeder(data)
        embedding = self.dropout(embedding)
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
