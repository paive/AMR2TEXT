import os
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter

from vocabulary import build_from_paths
from vocabulary import vocab_from_json
from vocabulary import vocab_to_json
from vocabulary import reverse_vocab
from graph_reader import Iterator
from model import build_model
from arguments import get_arguments
from log import setup_main_logger
from utils import id2sentence
from utils import vocab_index_word
import constants as C


def save_model(args, model, iter, val_loss=None):
    model_dir = os.path.join(args.save_dir, args.encoder_type, str(args.stadia))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    best_iter_file = os.path.join(model_dir, 'best_iter.txt')
    with open(best_iter_file, "w") as f:
        f.writelines([str(iter)+'\n', str(val_loss)])
    for fn in os.listdir(model_dir):
        if fn.endswith('th'):
            os.remove(os.path.join(model_dir, fn))
    model_dict_path = os.path.join(model_dir, f'model_{iter}.th')
    torch.save(model.state_dict(), model_dict_path)


def save_trained_iters(args, iters):
    model_dir = os.path.join(args.save_dir, args.encoder_type, str(args.stadia))
    trained_iter_file = os.path.join(model_dir, 'trained_iters.txt')
    with open(trained_iter_file, "w") as f:
        f.write(str(iters))


def load_model(args, model):
    model_dir = os.path.join(args.save_dir, args.encoder_type, str(args.stadia))
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
        vocab = build_from_paths([args.train_amr, args.train_snt, args.train_linear_amr, args.dev_amr, args.dev_snt, args.dev_linear_amr],
                                 args.num_words, args.min_count)
        vocab_to_json(vocab, args.vocab)
    edge_vocab = vocab_from_json(args.edge_vocab)
    return vocab, edge_vocab


def build_train_dataiters(args, vocab, edge_vocab):
    train_iter = Iterator(
        vocab, edge_vocab, args.batch_size, args.train_amr, args.train_grh, args.train_linear_amr, args.train_snt, stadia=args.stadia,
        max_src_len=args.max_seq_len[0], max_tgt_len=args.max_seq_len[1])
    dev_iter = Iterator(
        vocab, edge_vocab, args.batch_size, args.dev_amr, args.dev_grh, args.dev_linear_amr, args.dev_snt, stadia=args.stadia)
    return train_iter, dev_iter


def prepare_input_from_dicts(batch_dicts, cuda_device=None):
    nlabel = torch.LongTensor(batch_dicts['batch_nlabel'])
    npos = torch.LongTensor(batch_dicts['batch_npos'])
    adjs = torch.LongTensor(batch_dicts['batch_adjs'])
    relative_pos = torch.LongTensor(batch_dicts['relative_pos'])
    node_mask = torch.LongTensor(batch_dicts['node_mask'])

    linear_amr = torch.LongTensor(batch_dicts['linear_amr'])
    linear_amr_mask = torch.LongTensor(batch_dicts['linear_amr_mask'])
    aligns = torch.LongTensor(batch_dicts['aligns'])

    tokens = torch.LongTensor(batch_dicts['tokens'])
    token_mask = torch.LongTensor(batch_dicts['token_mask'])

    if cuda_device is not None:
        nlabel = nlabel.to(cuda_device)
        npos = npos.to(cuda_device)
        adjs = adjs.to(cuda_device)
        relative_pos = relative_pos.to(cuda_device)
        node_mask = node_mask.to(cuda_device)
        tokens = tokens.to(cuda_device)
        token_mask = token_mask.to(cuda_device)
        linear_amr = linear_amr.to(cuda_device)
        linear_amr_mask = linear_amr_mask.to(cuda_device)
        aligns = aligns.to(cuda_device)
    return nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns


class TrainConfig:
    def __init__(self, args):
        self.save_dir = args.save_dir
        self.iters = args.iters
        self.check_freq = args.checkpoint_frequency
        self.lr = args.lr
        self.lr_reduce_factor = args.lr_reduce_factor
        self.lr_num_not_improved = args.lr_num_not_improved
        self.patience = args.patience
        self.weight_decay = args.weight_decay

    def __str__(self):
        return "\tSave dir:".ljust(C.PRINT_SPACE) + str(self.save_dir) + "\n" + \
               "\tIters:".ljust(C.PRINT_SPACE) + str(self.iters) + "\n" + \
               "\tCheck frequency:".ljust(C.PRINT_SPACE) + str(self.check_freq) + "\n" + \
               "\tLearning rate:".ljust(C.PRINT_SPACE) + str(self.lr) + "\n" + \
               "\tWeight decay:".ljust(C.PRINT_SPACE) + str(self.weight_decay) + "\n" + \
               "\tlr reduce factor".ljust(C.PRINT_SPACE) + str(self.lr_reduce_factor) + "\n" + \
               "\tlr num not improved".ljust(C.PRINT_SPACE) + str(self.lr_num_not_improved) + "\n" + \
               "\tPatience".ljust(C.PRINT_SPACE) + str(self.patience) + "\n"


def train(args, config, model, train_iter, dev_iter, cuda_device, logger, writer):
    # load model
    trained_iters, best_iter, best_loss = load_model(args, model)
    if best_loss is None:
        print("None model exists in {}, load failure.".format(
            os.path.join(args.save_dir, args.encoder_type, str(args.stadia))))
    else:
        print("Load newest model successfully.")

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_reduce_factor, patience=config.lr_num_not_improved)
    model.train()
    train_loss = []

    for iter_id in range(trained_iters + 1, config.iters + 1):
        batch_dicts, finish = train_iter.next()
        nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns = prepare_input_from_dicts(batch_dicts, cuda_device)
        loss = model(nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns)
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
                save_model(args, model, iter_id, best_loss)
                best_iter = iter_id
            save_trained_iters(args, iter_id)
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
        nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns = prepare_input_from_dicts(batch_dicts, cuda_device)
        loss = model(nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns)
        val_loss.append(float(loss))
        if finish:
            break
    model.train()
    return np.mean(val_loss)


def build_test_dataiters(args, vocab, edge_vocab):
    test_iter = Iterator(
        vocab, edge_vocab, 64, args.test_amr, args.test_grh, args.test_linear_amr, args.test_snt, stadia=args.stadia)
    return test_iter


def test(args, model, test_iter, vocab, inversed_vocab, cuda_device):
    # load model
    trained_iters, best_iter, best_loss = load_model(args, model)
    if best_loss is None:
        print("None model exists in {}, load failure.".format(
            os.path.join(args.save_dir, args.encoder_type, str(args.stadia))))
    else:
        print("Load newest model successfully.")
    model.eval()

    gold_snt = []
    pred_snt = []

    # bos = vocab_index_word(vocab, C.BOS_SYMBOL)
    # eos = vocab_index_word(vocab, C.EOS_SYMBOL)
    # beam = BeamSearch(
    #     model=model, bos=bos, eos=eos,
    #     max_step=config.max_step, beam_size=config.beam_size)

    while True:
        batch_dicts, finish, raw_snt = test_iter.next(raw_snt=True)
        nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns = prepare_input_from_dicts(batch_dicts, cuda_device)
        predictions, _ = model.predict_with_beam_search(
            tokens[:, 0], nlabel, npos, adjs, relative_pos, node_mask, args.max_step, args.beam_size)
        # predictions, _ = beam.advance(nlabel, npos, adjs, relative_pos, node_mask)
        pred = id2sentence(predictions, inversed_vocab)
        gold_snt.extend(raw_snt)
        pred_snt.extend(pred)
        if finish:
            break
    result_dir = os.path.join(args.result_dir, args.encoder_type, str(args.stadia))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    bleu = corpus_bleu([[g] for g in gold_snt], pred_snt)
    with open(os.path.join(result_dir, "output.txt"), 'w') as f:
        lines = ["Corpus bleu score: " + str(bleu) + "\n\n"]
        assert len(gold_snt) == len(pred_snt)
        for idx in range(len(gold_snt)):
            lines.append("snt:: " + " ".join(gold_snt[idx]) + "\n" +
                         "snt_out:: " + " ".join(pred_snt[idx]) + "\n\n")
        f.writelines(lines)
    print("Write output success!")


def main(args, logger, cuda_device):
    logger.info("Load vocab and build data iterators...")
    vocab, edge_vocab = build_vocab(args)
    model = build_model(args, logger, cuda_device, vocab, edge_vocab)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    logger.info("Model size: {}".format(num_params / 1e6))

    if args.mode == 'train':
        logger.info("Training...")
        writer = SummaryWriter()
        train_config = TrainConfig(args)
        print('Train config:\n', train_config)
        train_iter, test_iter = build_train_dataiters(args, vocab, edge_vocab)
        train(args, train_config, model, train_iter, test_iter, cuda_device, logger, writer)
        writer.close()
    elif args.mode == 'test':
        logger.info('Test...')
        inversed_vocab = reverse_vocab(vocab)
        test_iter = build_test_dataiters(args, vocab, edge_vocab)
        with torch.no_grad():
            test(args, model, test_iter, vocab, inversed_vocab, cuda_device)


if __name__ == "__main__":
    args = get_arguments()
    log_dir = os.path.join(args.log_dir, args.encoder_type, str(args.stadia))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = setup_main_logger(__name__, file_logging=True, console=not args.quiet,
                               path=os.path.join(log_dir, C.LOG_NAME))
    if args.cuda_device is not None:
        cuda_device = torch.device(0)
    else:
        cuda_device = torch.device('cpu')
    main(args, logger, cuda_device)
