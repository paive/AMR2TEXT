'''
@Author: Neo
@Date: 2019-09-02 19:20:08
@LastEditTime: 2019-09-05 16:16:11
'''

import os
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter

from vocabulary import build_from_paths
from vocabulary import vocab_from_json
from vocabulary import vocab_to_json
from vocabulary import reverse_vocab
from graph_reader import BucketIterator
from graph_reader import Iterator
from model import build_model
from arguments import get_arguments
from log import setup_main_logger
from utils import id2sentence
import constants as C


def save_model(model, model_dir, iter, val_loss=None):
    best_iter_file = os.path.join(model_dir, 'best_iter.txt')
    with open(best_iter_file, "w") as f:
        f.writelines([str(iter)+'\n', str(val_loss)])
    for fn in os.listdir(model_dir):
        if fn.endswith('th'):
            os.remove(os.path.join(model_dir, fn))
    model_dict_path = os.path.join(model_dir, f'model_{iter}.th')
    torch.save(model.state_dict(), model_dict_path)


def save_trained_iters(model_dir, iters):
    trained_iter_file = os.path.join(model_dir, 'trained_iters.txt')
    with open(trained_iter_file, "w") as f:
        f.write(str(iters))


def load_model(model, model_dir):
    trained_iters = 0
    trained_iters_file = os.path.join(model_dir, 'trained_iters.txt')
    if os.path.exists(trained_iters_file):
        with open(trained_iters_file) as f:
            lines = f.readlines()
        trained_iters = int(lines[0].strip())

    best_iter = -1
    best_loss = None
    best_iter_file = os.path.join(model_dir, 'best_iter.txt')
    if os.path.exists(best_iter_file):
        with open(best_iter_file) as f:
            lines = f.readlines()
        best_iter = int(lines[0].strip())
        best_loss = float(lines[1].strip())

    model_dict_path = os.path.join(model_dir, f'model_{best_iter}.th')
    if os.path.exists(model_dict_path):
        logger.info("Model state dict exists.")
        model.load_state_dict(torch.load(model_dict_path))
    return trained_iters, best_iter, best_loss


def build_vocab(args):
    if os.path.exists(args.vocab):
        vocab = vocab_from_json(args.vocab)
    else:
        vocab = build_from_paths([args.train_amr, args.train_snt, args.dev_amr, args.dev_snt],
                                 args.num_words, args.min_count)
        vocab_to_json(vocab, args.vocab)
    edge_vocab = vocab_from_json(args.edge_vocab)
    return vocab, edge_vocab

# def build_dataiters(args, vocab, edge_vocab):
#     train_iter = BucketIterator(
#         vocab, edge_vocab, args.batch_size, args.train_amr, args.train_grh, args.train_snt,
#         args.max_seq_len[0], args.max_seq_len[1], args.bucket_num[0], True)
#     dev_iter = BucketIterator(
#         vocab, edge_vocab, args.batch_size, args.dev_amr, args.dev_grh, args.dev_snt,
#         args.max_seq_len[0], args.max_seq_len[1], args.bucket_num[1], False)
#     test_iter = BucketIterator(
#         vocab, edge_vocab, args.batch_size, args.test_amr, args.test_grh, args.test_snt,
#         args.max_seq_len[0], args.max_seq_len[1], args.bucket_num[1], False)
#     return train_iter, dev_iter, test_iter


def build_dataiters(args, vocab, edge_vocab):
    train_iter = Iterator(
        vocab, edge_vocab, args.batch_size, args.train_amr, args.train_grh, args.train_snt,
        args.max_seq_len[0], args.max_seq_len[1])
    dev_iter = Iterator(
        vocab, edge_vocab, args.batch_size, args.dev_amr, args.dev_grh, args.dev_snt,
        args.max_seq_len[0], args.max_seq_len[1])
    test_iter = Iterator(
        vocab, edge_vocab, args.batch_size, args.test_amr, args.test_grh, args.test_snt,
        args.max_seq_len[0], args.max_seq_len[1])
    return train_iter, dev_iter, test_iter


def prepare_input_from_dicts(batch_dicts, cuda_device=None):
    nlabel = torch.LongTensor(batch_dicts['batch_nlabel'])
    npos = torch.LongTensor(batch_dicts['batch_npos'])
    adjs = torch.LongTensor(batch_dicts['batch_adjs'])
    node_mask = torch.LongTensor(batch_dicts['node_mask'])

    tokens = torch.LongTensor(batch_dicts['tokens'])
    token_mask = torch.LongTensor(batch_dicts['token_mask'])

    if cuda_device is not None:
        nlabel = nlabel.to(cuda_device)
        npos = npos.to(cuda_device)
        adjs = adjs.to(cuda_device)
        node_mask = node_mask.to(cuda_device)
        tokens = tokens.to(cuda_device)
        token_mask = token_mask.to(cuda_device)
    return nlabel, npos, adjs, node_mask, tokens, token_mask


class TrainConfig:
    def __init__(self, args):
        self.save_dir = args.save_dir
        self.iters = args.iters
        self.check_freq = args.checkpoint_frequency
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.momentum = args.momentum
        self.lr_reduce_factor = args.lr_reduce_factor
        self.lr_num_not_improved = args.lr_num_not_improved
        self.patience = args.patience

    def __str__(self):
        return "\tSave dir:".ljust(C.PRINT_SPACE) + str(self.save_dir) + "\n" + \
               "\tIters:".ljust(C.PRINT_SPACE) + str(self.iters) + "\n" + \
               "\tCheck frequency:".ljust(C.PRINT_SPACE) + str(self.check_freq) + "\n" + \
               "\tOptimizer:".ljust(C.PRINT_SPACE) + str(self.optimizer) + "\n" + \
               "\tLearning rate:".ljust(C.PRINT_SPACE) + str(self.lr) + "\n" + \
               "\tMomentum:".ljust(C.PRINT_SPACE) + str(self.momentum) + "\n" + \
               "\tlr reduce factor".ljust(C.PRINT_SPACE) + str(self.lr_reduce_factor) + "\n" + \
               "\tlr num not improved".ljust(C.PRINT_SPACE) + str(self.lr_num_not_improved) + "\n" + \
               "\tPatience".ljust(C.PRINT_SPACE) + str(self.patience) + "\n"


def train(config, model, train_iter, dev_iter, cuda_device, logger, writer):
    # load model
    trained_iters, best_iter, best_loss = load_model(model, config.save_dir)
    if best_loss is None:
        print("None model exists in {}, load failure.".format(config.save_dir))
    else:
        print("Load newest model successfully.")

    # optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_reduce_factor, patience=config.lr_num_not_improved)
    model.train()
    train_loss = []

    for iter_id in range(trained_iters + 1, config.iters + 1):
        batch_dicts, finish = train_iter.next()
        nlabel, npos, adjs, node_mask, tokens, token_mask = prepare_input_from_dicts(batch_dicts, cuda_device)
        loss = model(nlabel, npos, adjs, node_mask, tokens, token_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(float(loss))

        if iter_id % 50 == 0:
            logger.info(f"Iter {iter_id}, train loss {np.mean(train_loss)}, ppl {np.exp(np.mean(train_loss))}")
            train_loss = []
            # tensorboardX
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(name, param.data.cpu().numpy(), iter_id)

        # validation
        if iter_id % args.checkpoint_frequency == 0:
            val_loss = validation(model, dev_iter)
            logger.info(f"Iter {iter_id}, val loss {val_loss}, ppl {np.exp(val_loss)}")
            writer.add_scalar("val_ppl", np.exp(np.mean(val_loss)), iter_id)
            # save model
            if (best_loss is None) or (val_loss < best_loss):
                best_loss = val_loss
                save_model(model, config.save_dir, iter_id, best_loss)
                best_iter = iter_id
            save_trained_iters(config.save_dir, iter_id)
            scheduler.step(val_loss)
            print(f"Learning rate changes to {scheduler.optimizer.param_groups[0]['lr']}")

        # early stop
        if (iter_id - best_iter) >= config.check_freq * config.patience:
            break


def validation(model, dev_iter):
    model.eval()
    val_loss = []
    while True:
        batch_dicts, finish = dev_iter.next()
        nlabel, npos, adjs, node_mask, tokens, token_mask = prepare_input_from_dicts(batch_dicts, cuda_device)
        loss = model(nlabel, npos, adjs, node_mask, tokens, token_mask)
        val_loss.append(float(loss))
        if finish:
            break
    model.train()
    return np.mean(val_loss)


class TestConfig:
    def __init__(self, args):
        self.max_step = args.max_seq_len[1]
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir

    def __str__(self):
        return "\tSave dir:".ljust(C.PRINT_SPACE) + str(self.save_dir) + "\n" + \
               "\tResult dir:".ljust(C.PRINT_SPACE) + str(self.result_dir) + "\n" + \
               "\tMax step".ljust(C.PRINT_SPACE) + str(self.max_step) + "\n"


def test(config, model, train_iter, test_iter, inversed_vocab, cuda_device):
    # load model
    trained_iters, best_iter, best_loss = load_model(model, config.save_dir)
    if best_loss is None:
        print("None model exists in {}, load failure.".format(config.save_dir))
    else:
        print("Load newest model successfully.")
    model.eval()

    gold_snt = []
    pred_snt = []
    while True:
        batch_dicts, finish = test_iter.next()
        nlabel, npos, adjs, node_mask, tokens, token_mask = prepare_input_from_dicts(batch_dicts, cuda_device)
        predictions = model.predict(tokens[:, 0], nlabel, npos, adjs, node_mask, config.max_step)
        gold = id2sentence(tokens[:, 1:], inversed_vocab)
        pred = id2sentence(predictions, inversed_vocab)
        gold_snt.extend([g] for g in gold)
        pred_snt.extend(pred)
        if finish:
            break

    bleu = corpus_bleu(gold_snt, pred_snt)
    with open(os.path.join(config.result_dir, "output.txt"), 'w') as f:
        lines = ["Corpus bleu score: " + str(bleu) + "\n\n"]
        assert len(gold_snt) == len(pred_snt)
        for idx in range(len(gold_snt)):
            lines.append("snt:: " + " ".join(gold_snt[idx][0]) + "\n" +
                         "snt_out:: " + " ".join(pred_snt[idx]) + "\n\n")
        f.writelines(lines)
    print("Write output success!")


def main(args, logger, cuda_device, writer):
    logger.info("Load vocab and build data iterators...")
    vocab, edge_vocab = build_vocab(args)
    train_iter, dev_iter, test_iter = build_dataiters(args, vocab, edge_vocab)
    model = build_model(args, logger, cuda_device, vocab, edge_vocab)

    if args.mode == 'train':
        logger.info("Training...")
        train_config = TrainConfig(args)
        print('Train config:\n', train_config)
        train(train_config, model, train_iter, dev_iter, cuda_device, logger, writer)
    elif args.mode == 'test':
        logger.info('Test...')
        test_config = TestConfig(args)
        print('Test config:\n', test_config)
        inversed_vocab = reverse_vocab(vocab)
        test(test_config, model, train_iter, dev_iter, inversed_vocab, cuda_device)


if __name__ == "__main__":
    args = get_arguments()
    logger = setup_main_logger(__name__, file_logging=True, console=not args.quiet,
                               path=os.path.join(args.log_dir, C.LOG_NAME))
    cuda_device = torch.device(0)
    writer = SummaryWriter()
    main(args, logger, cuda_device, writer)
    writer.close()
