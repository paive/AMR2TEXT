import torch
from queue import Queue


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

        input_token = torch.LongTensor([self.bos]).to(value.device)
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
            for idx in range(node_queue.qsize()):
                node = node_queue.get()
                if int(node.input_token[0]) == self.eos or node.length >= self.max_step:
                    end_nodes.append(node)
                    continue

                emb = self.model.token_embeder(node.input_token)
                if self.model.decoder.config.cell_type == 'GRU':
                    state, cov_vec, similarity = self.model.decoder._step(emb=emb, value=value, value_mask=node_mask, state=node.hidden, cov_vec=node.coverage)
                    cell = None
                elif self.model.decoder.config.cell_type == 'LSTM':
                    state, cell, cov_vec, similarity = self.model.decoder._step(emb=emb, value=value, value_mask=node_mask, state=node.hidden, c1=node.cell, cov_vec=node.coverage)
                logit = self.model.projector(state)
                vocab_log_prob = torch.log_softmax(logit, dim=-1)
                topk_log_prob, t = torch.topk(vocab_log_prob, k=self.beam_size, dim=-1)

                for k in range(self.beam_size):
                    index = t[0, k].unsqueeze(0)
                    log_p = float(topk_log_prob[0, k])
                    child = BeamNode(hidden=state, previous_node=node, input_token=index,
                                     attn=similarity, log_prob=node.log_prob+log_p, length=node.length+1,
                                     cell=cell, coverage=cov_vec)
                    candidates.append(child)
            candidates = sorted(candidates, key=lambda x: x.log_prob, reverse=True)
            length = min(len(candidates), self.beam_size)
            for i in range(length):
                node_queue.put(candidates[i])

        end_nodes = sorted(end_nodes, key=lambda x: x.log_prob, reverse=True)
        end_nodes = end_nodes[: self.beam_size]

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
