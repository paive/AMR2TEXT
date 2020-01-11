import argparse
import constants as C


def multiple_values(num_values=0, greater_or_equal=None, data_type=int):
    def parse(value_to_check):
        if ':' in value_to_check:
            expected_num_separators = num_values - 1 if num_values else 0
            if expected_num_separators > 0 and (value_to_check.count(':') != expected_num_separators):
                raise argparse.ArgumentTypeError("Expected either a single value or %d values separated by %s" %
                                                 (num_values, C.ARG_SEPARATOR))
            values = tuple(map(data_type, value_to_check.split(C.ARG_SEPARATOR, num_values - 1)))
        else:
            values = tuple([data_type(value_to_check)] * num_values)
        if greater_or_equal is not None:
            if any((value < greater_or_equal for value in values)):
                raise argparse.ArgumentTypeError("Must provide value greater or equal to %d" % greater_or_equal)
        return values

    return parse


def get_arguments():
    data_version = 'amr2.0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # log
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--quiet', type=bool, default=False)

    # data and input
    parser.add_argument('--train_amr', type=str, default=f'./data/{data_version}/train.amr')
    parser.add_argument('--train_grh', type=str, default=f'./data/{data_version}/train.grh')
    parser.add_argument('--train_snt', type=str, default=f'./data/{data_version}/train.snt')
    parser.add_argument('--train_linear_amr', type=str, default=f'./data/{data_version}/train.linear_amr')
    parser.add_argument('--dev_amr', type=str, default=f'./data/{data_version}/dev.amr')
    parser.add_argument('--dev_grh', type=str, default=f'./data/{data_version}/dev.grh')
    parser.add_argument('--dev_snt', type=str, default=f'./data/{data_version}/dev.snt')
    parser.add_argument('--dev_linear_amr', type=str, default=f'./data/{data_version}/dev.linear_amr')
    parser.add_argument('--test_amr', type=str, default=f'./data/{data_version}/test.amr')
    parser.add_argument('--test_grh', type=str, default=f'./data/{data_version}/test.grh')
    parser.add_argument('--test_snt', type=str, default=f'./data/{data_version}/test.snt')
    parser.add_argument('--test_linear_amr', type=str, default=f'./data/{data_version}/test.linear_amr')
    parser.add_argument('--max_seq_len', type=multiple_values(2, 1), default=(200, 200))
    parser.add_argument('--bucket_num', type=multiple_values(2, 1), default=(20, 10))
    parser.add_argument('--stadia', type=int, default=1)

    # vocabulary
    parser.add_argument('--num_words', type=int)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--vocab', type=str, default=f'./data/{data_version}/vocab.json')
    parser.add_argument('--edge_vocab', type=str, default=f'./data/{data_version}/edge_vocab.json')

    # model
    parser.add_argument('--encoder_type', type=str, default='gcn', choices=['gcn', 'rnn', 'both'])
    parser.add_argument('--bigcn', action='store_true')
    parser.add_argument('--emb_dim', type=int, default=360)
    parser.add_argument('--pos_emb_dim', type=int, default=300)
    parser.add_argument('--emb_dropout', type=multiple_values(2, 0), default=(.5, .5))
    parser.add_argument('--scale_grad_by_freq', type=bool, default=True)
    parser.add_argument('--weight_tying', type=bool, default=True)
    parser.add_argument('--model_dropout', type=multiple_values(2, 0), default=(0.5, 0.2))
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--gcn_layers', type=int, default=8)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--decoder_cell', type=str, default='LSTM')
    parser.add_argument('--coverage', type=bool, default=True)
    parser.add_argument('--init_param', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='./save')

    # predict
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_step', type=int, default=300)

    # train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iters', type=int, default=2000000)
    parser.add_argument('--checkpoint-frequency', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.8)
    parser.add_argument('--lr_num_not_improved', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args, _ = parser.parse_known_args()
    return args


def add_argument(params, name, dtype, default):
    params.add_argument("--" + name, type=dtype, default=default)
