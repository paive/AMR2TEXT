import torch
from queue import Queue
import numpy as np


class BeamNode:
    def __init__(self, hidden, previous_node, input_token, attn, log_prob, length, cell=None, coverage=None):
        self.hidden = hidden
        self.cell = cell
        self.previous_node = previous_node
        self.input_token = input_token
        self.attn = attn
        self.coverage = coverage
        self.log_prob = log_prob
        self.length = length


class BeamSearch:
    def __init__(self, model, bos, eos, max_step, beam_size):
        self.model = model
        self.bos = bos
        self.eos = eos
        self.max_step = max_step
        self.beam_size = beam_size

    def advance(self, nlabel, npos, adjs, relative_pos, node_mask):
        h = self.model.embedding_graph(nlabel, npos)
        value, state = self.model.encode_graph(adjs, relative_pos, h, node_mask)
        if self.model.decoder.config.cell_type == 'LSTM':
            cell = torch.zeros_like(state)
        else:
            cell = None
        if self.model.decoder.config.coverage:
            cov_vec = torch.zeros(value.size(0), value.size(1))
            cov_vec = cov_vec.to(value.device)
        else:
            cov_vec = None

        input_token = torch.tensor(self.bos).long().to(value.device)
        root = BeamNode(hidden=state,
                        previous_node=None,
                        input_token=input_token,
                        attn=None,
                        log_prob=0.,
                        length=1,
                        cell=cell,
                        coverage=cov_vec)

        node_queue = Queue()
        node_queue.put(root)

        end_nodes = []
        while not node_queue.empty():
            candidates = []

            batch_tokens = []
            batch_state = []
            batch_cell = []
            batch_coverage = []
            batch_node = []
            for idx in range(node_queue.qsize()):
                node = node_queue.get()
                batch_node.append(node)
                if int(node.input_token) == self.eos or node.length >= self.max_step:
                    end_nodes.append(node)
                    continue
                batch_tokens.append(node.input_token)
                batch_state.append(node.hidden)
                if node.cell is not None:
                    batch_cell.append(node.cell)
                if node.coverage is not None:
                    batch_coverage.append(node.coverage)
            if len(batch_tokens) == 0:
                break
            batch_tokens = torch.stack(batch_tokens, dim=0)
            batch_state = torch.cat(batch_state, dim=0)
            if len(batch_cell) > 0:
                batch_cell = torch.cat(batch_cell, dim=0)
            if len(batch_coverage) > 0:
                batch_coverage = torch.cat(batch_coverage, dim=0)
            batch_size = len(batch_tokens)
            emb = self.model.token_embeder(batch_tokens)

            if self.model.decoder.config.cell_type == 'GRU':
                state, cov_vec, similarity = self.model.decoder._step(emb=emb, value=value.repeat(batch_size, 1, 1), value_mask=node_mask.repeat(batch_size, 1), state=batch_state, cov_vec=batch_coverage)
                cell = None
            elif self.model.decoder.config.cell_type == 'LSTM':
                state, cell, cov_vec, similarity = self.model.decoder._step(emb=emb, value=value.repeat(batch_size, 1, 1), value_mask=node_mask.repeat(batch_size, 1), state=batch_state, c1=batch_cell, cov_vec=batch_coverage)
            logit = self.model.projector(state)
            vocab_log_prob = torch.log_softmax(logit, dim=-1)       # B x V
            topk_log_prob, t = torch.topk(vocab_log_prob, k=self.beam_size, dim=-1)

            for idx in range(batch_size):
                for k in range(self.beam_size):
                    index = t[idx, k]
                    log_p = float(topk_log_prob[idx, k])
                    child = BeamNode(hidden=state[idx].unsqueeze(0), previous_node=batch_node[idx], input_token=index,
                                     attn=similarity[idx].unsqueeze(0), log_prob=batch_node[idx].log_prob+log_p, length=batch_node[idx].length+1, cell=cell[idx].unsqueeze(0), coverage=cov_vec[idx].unsqueeze(0))
                    candidates.append(child)

                # emb = self.model.token_embeder(node.input_token)
                # if self.model.decoder.config.cell_type == 'GRU':
                #     state, cov_vec, similarity = self.model.decoder._step(emb=emb, value=value, value_mask=node_mask, state=node.hidden, cov_vec=node.coverage)
                #     cell = None
                # elif self.model.decoder.config.cell_type == 'LSTM':
                #     state, cell, cov_vec, similarity = self.model.decoder._step(emb=emb, value=value, value_mask=node_mask, state=node.hidden, c1=node.cell, cov_vec=node.coverage)
                # logit = self.model.projector(state)
                # vocab_log_prob = torch.log_softmax(logit, dim=-1)
                # topk_log_prob, t = torch.topk(vocab_log_prob, k=self.beam_size, dim=-1)

                # for k in range(self.beam_size):
                #     index = t[0, k].unsqueeze(0)
                #     log_p = float(topk_log_prob[0, k])
                #     child = BeamNode(hidden=state, previous_node=node, input_token=index,
                #                      attn=similarity, log_prob=node.log_prob+log_p, length=node.length+1,
                #                      cell=cell, coverage=cov_vec)
                #     candidates.append(child)

            candidates = sorted(candidates, key=lambda x: x.log_prob, reverse=True)
            length = min(len(candidates), self.beam_size)
            for i in range(length):
                node_queue.put(candidates[i])

        end_nodes = sorted(end_nodes, key=endnode_sort_key, reverse=True)

        node = end_nodes[0]
        log_prob = node.log_prob
        predictions = []
        while node is not None:
            predictions.insert(0, int(node.input_token))
            node = node.previous_node
        if predictions[0] == self.bos:
            del(predictions[0])
        if predictions[-1] == self.eos:
            del(predictions[-1])

        return [predictions], log_prob


def endnode_sort_key(x, max_step=200):
    lenth_term = np.exp(x.length/max_step)
    logp_term = np.exp(x.log_prob/x.length)
    return lenth_term / (-logp_term)
