from instance import Instance
from instance import pad_instance
import random
import math


class IteratorBase:
    def __init__(self, vocab, edge_vocab, batch_size, amr_path, grp_path, linear_amr_path, snt_path, stadia, max_src_len, max_tgt_len, keep_ratio):
        with open(amr_path, 'r') as f:
            amr_lines = f.readlines()
        with open(grp_path, 'r') as f:
            grp_lines = f.readlines()
        with open(linear_amr_path, 'r') as f:
            linear_amrs = f.readlines()
        with open(snt_path, 'r') as f:
            snt_lines = f.readlines()
        assert len(amr_lines) == len(grp_lines)
        assert len(grp_lines) == len(snt_lines)
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.stadia = stadia

        self.instances = []
        self.depracated_instances = []
        for idx in range(len(amr_lines)):
            ins = Instance(amr_lines[idx], grp_lines[idx], linear_amrs[idx], snt_lines[idx], stadia)
            ins.index(vocab, edge_vocab)
            ins.set_id(idx)
            if max_src_len is not None and max_tgt_len is not None:
                if len(ins.indexed_node) > max_src_len or len(ins.indexed_token) > max_tgt_len:
                    self.depracated_instances.append(ins)
                else:
                    self.instances.append(ins)
            else:
                self.instances.append(ins)

        if keep_ratio is not None:
            num = len(self.instances) * keep_ratio
            num = math.ceil(num)
            random.shuffle(self.instances)
            self.instances = self.instances[: num]


class Iterator(IteratorBase):
    def __init__(self, vocab, edge_vocab, batch_size, amr_path, grp_path, linear_amr_path, snt_path, stadia,
                 max_src_len=None, max_tgt_len=None, keep_ratio=None):
        super().__init__(vocab, edge_vocab, batch_size, amr_path, grp_path, linear_amr_path, snt_path, stadia, max_src_len, max_tgt_len, keep_ratio=keep_ratio)
        self.cur = 0

    def next(self, raw_snt=False):
        if self.cur == 0:
            random.shuffle(self.instances)
        r = min(self.cur + self.batch_size, len(self.instances))
        batch_instances = self.instances[self.cur: r]
        batch_dict = self._prepare(batch_instances)
        self.cur = r

        return_content = []
        if self.cur >= len(self.instances):
            self.cur = 0
            return_content = [batch_dict, True]
        else:
            return_content = [batch_dict, False]
        if raw_snt:
            sentences = self.get_raw_sen(batch_instances)
            return_content.append(sentences)
        return return_content

    def get_raw_sen(self, batch_instances):
        sentences = []
        for ins in batch_instances:
            sentences.append(ins.snt.tokens[1:-1])
        return sentences

    def _prepare(self, batch_instances):
        src_len = 0
        linear_amr_len = 0
        tgt_len = 0
        tokens = []
        token_mask = []
        nodes = []
        node_mask = []
        poses = []
        adjs = []
        relative_pos = []
        linear_amr = []
        linear_amr_mask = []
        aligns = []
        for ins in batch_instances:
            src_len = max(src_len, len(ins.indexed_node))
            linear_amr_len = max(linear_amr_len, len(ins.indexed_linear_amr))
            tgt_len = max(tgt_len, len(ins.indexed_token))
        for ins in batch_instances:
            new_ins = pad_instance(ins, src_len, linear_amr_len, tgt_len, self.stadia)
            tokens.append(new_ins.indexed_token)
            token_mask.append(new_ins.token_mask)
            nodes.append(new_ins.indexed_node)
            node_mask.append(new_ins.node_mask)
            poses.append(new_ins.graph_pos)
            adjs.append(new_ins.adj)
            relative_pos.append(new_ins.relative_pos)
            linear_amr.append(new_ins.indexed_linear_amr)
            linear_amr_mask.append(new_ins.linear_amr_mask)
            aligns.append(new_ins.aligns)

        return {"batch_nlabel": nodes,
                "batch_npos": poses,
                "batch_adjs": adjs,
                "node_mask": node_mask,
                "tokens": tokens,
                "token_mask": token_mask,
                'relative_pos': relative_pos,
                'linear_amr': linear_amr,
                'linear_amr_mask': linear_amr_mask,
                'aligns': aligns}


if __name__ == "__main__":
    from vocabulary import vocab_from_json
    # from vocabulary import vocab_to_json
    # from vocabulary import  build_from_paths
    from vocabulary import reverse_vocab
    # from utils import id2sentence
    from utils import visualization_graph

    dev_amr = './data/dev.amr'
    dev_snt = './data/dev.snt'
    dev_grh = "./data/dev.grh"
    dev_linear_amr = './data/dev.linear_amr'
    train_amr = './data/train.amr'
    train_snt = './data/train.snt'
    train_grh = './data/train.grh'
    train_linear_amr = './data/train.linear_amr'

    test_amr = './data/test.amr'
    test_snt = './data/test.snt'
    test_grh = "./data/test.grh"
    test_linear_amr = './data/test.linear_amr'

    # vocab = build_from_paths([train_amr, train_snt, dev_amr, dev_snt], 30000, 2)
    # vocab_to_json(vocab, "./data/new_vocab.json")
    # raise NotImplementedError

    vocab = vocab_from_json('./data/vocab.json')
    inverse_vocab = reverse_vocab(vocab)
    edge_vocab = vocab_from_json('./data/edge_vocab.json')

    # train_iter = BucketIterator(vocab, edge_vocab, 16, train_amr, train_grh, train_snt, 200, 200, 20, True)
    # dev_iter = BucketIterator(vocab, edge_vocab, 3, dev_amr, dev_grh, dev_snt, 200, 200, 10, True)
    # test_iter = BucketIterator(vocab, edge_vocab, 16, test_amr, test_grh, test_snt, 200, 200, 10, False)

    # train_iter = Iterator(vocab, edge_vocab, 16, train_amr, train_grh, train_snt, 3, 200, 200)
    dev_iter = Iterator(vocab, edge_vocab, 16, dev_amr, dev_grh, dev_linear_amr, dev_snt, 3, 200, 200)
    # test_iter = Iterator(vocab, edge_vocab, 16, test_amr, test_grh, test_snt, 3, 200, 200)

    i = 0
    while True:
        print(i)
        i += 1
        data, finish = dev_iter.next()
        print(data['batch_nlabel'][0])
        print(data['linear_amr'][0])
        print(data['aligns'][0])
        break

    # 可视化语义图
    # ins = dev_iter.instances[266]
    # visualization_graph(ins.id, ins.indexed_node, ins.adj, ins.indexed_token, inverse_vocab, edge_set=[1])
    # print(ins.graph_pos)
    # print(ins.adj)
    # print(ins.relative_pos)

    # 查看最大的儿子数
    # max_son_sum = 0
    # ins_id = -1
    # for idx, ins in enumerate(train_iter.instances):
    #     adj = ins.adj
    #     directed_edge = (adj == 1)
    #     ins_max_son_sum = np.max(np.sum(directed_edge, axis=1))
    #     if max_son_sum < ins_max_son_sum:
    #         max_son_sum = ins_max_son_sum
    #         ins_id = idx
    # print(max_son_sum)
    # ins = train_iter.instances[ins_id]
    # visualization_graph(ins.id, ins.indexed_node, ins.adj, ins.indexed_token, inverse_vocab)

    # 查看最大的深度
    # max_depth = 0
    # ins_id = -1
    # for idx, ins in enumerate(test_iter.instances):
    #     pos = ins.graph_pos
    #     ins_max_depth = np.max(ins.graph_pos)
    #     if max_depth < ins_max_depth:
    #         max_depth = ins_max_depth
    #         ins_id = idx
    # print(max_depth)
    # ins = test_iter.instances[ins_id]
    # visualization_graph(ins.id, ins.indexed_node, ins.adj, ins.indexed_token, inverse_vocab)
