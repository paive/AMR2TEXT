from typing import Iterator
from copy import deepcopy
import constants as C
import torch
from graphviz import Digraph
import math


def visualization_graph(_id, nlabels, adj, tokens, inverse_vocab):
    sen = " ".join([vocab_index_to_word(inverse_vocab, t) for t in tokens])
    dot = Digraph(comment=sen)

    for idx in range(len(nlabels)):
        dot.node(str(idx), vocab_index_to_word(inverse_vocab, nlabels[idx]))

    node_num = len(adj)
    for ids in range(node_num):
        for idd in range(node_num):
            if adj[ids][idd] != 0:
                dot.edge(str(ids), str(idd))
    dot.render(f'./figures/{_id}', view=False)


def vocab_index_word(vocab, word):
    if word in vocab:
        return vocab[word]
    else:
        return vocab[C.UNK_SYMBOL]


def vocab_index_to_word(inverse_vocab, index):
    if index in inverse_vocab:
        return inverse_vocab[index]
    else:
        return inverse_vocab[1]


def id2sentence(tokens, inverse_vocab):
    """Translate token ids to tokens, until encounter end_index"""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy().tolist()
    sentences = []
    for idx in range(len(tokens)):
        sen = []
        for t in tokens[idx]:
            if t == C.END_ID:
                break
            sen.append(vocab_index_to_word(inverse_vocab, t))
        sentences.append(sen)
    return sentences


def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def get_acti_fun(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif activation == 'gelu':
        return gelu
    elif activation == 'prelu':
        return torch.nn.PReLU()


def deprecated(func):
    def _wrapped(*args, **kwargs):
        print('*WARN* Keyword "%s" will deprecated' % func.__name__)
        return func(*args, **kwargs)
    _wrapped.func_name = func.__name__
    return _wrapped


@deprecated
def pad_2D_list(input_list, PAD, requires_mask=False):
    result = deepcopy(input_list)
    if requires_mask:
        mask = []
    length = [len(sublist) for sublist in result]
    max_len = max(length)
    for idx in range(len(result)):
        if requires_mask:
            mask.append([1] * length[idx])
            mask[idx].extend([0] * (max_len-length[idx]))
        result[idx].extend([PAD] * (max_len - length[idx]))
    if requires_mask:
        return result, mask
    return result


@deprecated
def pad_square_matrix(input_list, PAD):
    result = deepcopy(input_list)
    length = [len(adj) for adj in result]
    max_len = max(length)

    for idx in range(len(result)):
        pl = max_len - len(result[idx])
        if pl > 0:
            for ids in range(len(result[idx])):
                result[idx][ids].extend([PAD] * pl)
            for i in range(pl):
                result[idx].append([PAD] * max_len)
    return result


@deprecated
def ubatch_and_pad_graph_value(value, gsize):
    """Pad attention values with 0"""
    batch_size = len(gsize)
    assert sum(gsize) == value.size(0)
    value_dim = value.size(-1)
    batch_value = []
    gnode_value = []
    cur = 0
    for idx in range(batch_size):
        batch_value.append(value[cur: cur+gsize[idx]])
        gnode_value.append(value[cur+gsize[idx]-1].unsqueeze(0))
        cur += gsize[idx]

    max_len = max(gsize)
    mask = torch.ones(batch_size, max_len, device=value.device)
    for idx in range(batch_size):
        if max_len != gsize[idx]:
            pad = torch.zeros(max_len-gsize[idx], value_dim, device=value.device)
            batch_value[idx] = torch.cat((batch_value[idx], pad), dim=0)
        batch_value[idx] = batch_value[idx].unsqueeze(0)
        mask[idx, gsize[idx]:] = 0

    batch_value = torch.cat(batch_value, dim=0)
    gnode_value = torch.cat(gnode_value, 0)
    return batch_value, mask, gnode_value


if __name__ == '__main__':
    def test_ubatch_and_pad_graph_value():
        value = torch.rand(12, 4, requires_grad=True)
        gsize = [2, 3, 3, 4]

        bvalue, mask, gnode_value = ubatch_and_pad_graph_value(value, gsize)
        print(value)
        print(bvalue)
        print(gnode_value)

    def test_pad_square_matrix():
        value = [[[1, 2], [2, 1]], [[1]], [[1, 2, 0], [2, 1, 0], [1, 1, 0]]]
        print(pad_square_matrix(value, 0))

    def test_pad_2d_list():
        value = [[1, 2, 3], [0, 1]]
        value, mask = pad_2D_list(value, 0, True)
        print(value)
        print(mask)
    test_pad_2d_list()
