import json
import logging
from collections import Counter
from contextlib import ExitStack
from itertools import chain, islice
from typing import Dict, Iterable, List, Optional

import constants as C
import utils

logger = logging.getLogger(__name__)


Vocab = Dict[str, int]
InverseVocab = Dict[int, str]


def build_from_paths(paths: List[str],
                     num_words: Optional[int] = None,
                     min_count: int = 1,
                     pad_to_multiple_of: Optional[int] = None) -> Vocab:
    """
    Creates vocabulary from paths to a file in sentence-per-line format. A sentence is just a whitespace delimited
    list of tokens. Note that special symbols like the beginning of sentence (BOS) symbol will be added to the
    vocabulary.

    :param paths: List of paths to files with one sentence per line.
    :param num_words: Optional maximum number of words in the vocabulary.
    :param min_count: Minimum occurrences of words to be included in the vocabulary.
    :param pad_to_multiple_of: If not None, pads the vocabulary to a size that is the next multiple of this int.
    :return: Word-to-id mapping.
    """
    with ExitStack() as stack:
        logger.info("Building vocabulary from dataset(s): %s", paths)
        files = (stack.enter_context(open(path, "rt")) for path in paths)
        return build_vocab(chain(*files), num_words, min_count, pad_to_multiple_of)


def build_vocab(data: Iterable[str],
                num_words: Optional[int] = None,
                min_count: int = 1,
                pad_to_multiple_of: Optional[int] = None) -> Vocab:
    """
    Creates a vocabulary mapping from words to ids. Increasing integer ids are assigned by word frequency,
    using lexical sorting as a tie breaker. The only exception to this are special symbols such as the padding symbol
    (PAD).

    :param data: Sequence of sentences containing whitespace delimited tokens.
    :param num_words: Optional maximum number of words in the vocabulary.
    :param min_count: Minimum occurrences of words to be included in the vocabulary.
    :param pad_to_multiple_of: If not None, pads the vocabulary to a size that is the next multiple of this int.
    :return: Word-to-id mapping.
    """
    vocab_symbols_set = set(C.VOCAB_SYMBOLS)
    raw_vocab = Counter(token for line in data for token in utils.get_tokens(line)
                        if token not in vocab_symbols_set)
    # For words with the same count, they will be ordered reverse alphabetically.
    # Not an issue since we only care for consistency
    pruned_vocab = [w for c, w in sorted(((c, w) for w, c in raw_vocab.items() if c >= min_count), reverse=True)]

    if num_words is not None:
        vocab = list(islice(pruned_vocab, num_words))
        num_words_log = str(num_words)
    else:
        vocab = pruned_vocab
        num_words_log = "None"

    if pad_to_multiple_of is not None:
        current_vocab_size = len(vocab) + len(C.VOCAB_SYMBOLS)
        rest = current_vocab_size % pad_to_multiple_of
        padded_vocab_size = current_vocab_size if rest == 0 else current_vocab_size + pad_to_multiple_of - rest
        logger.info("Padding vocabulary to a multiple of %d: %d -> %d",
                    pad_to_multiple_of, current_vocab_size, padded_vocab_size)
        pad_entries = [C.PAD_FORMAT % idx for idx in range(current_vocab_size, padded_vocab_size)]
        pad_to_multiple_log = str(pad_to_multiple_of)
    else:
        pad_entries = []
        pad_to_multiple_log = "None"

    word_to_id = {word: idx for idx, word in enumerate(chain(C.VOCAB_SYMBOLS, vocab, pad_entries))}
    logger.info("Vocabulary: types: %d/%d/%d/%d (initial/min_pruned/max_pruned/+special) " +
                "[min_frequency=%d, max_num_types=%s, pad_to_multiple_of=%s]",
                len(raw_vocab), len(pruned_vocab), len(vocab),
                len(word_to_id), min_count, num_words_log, pad_to_multiple_log)

    # Important: pad symbol becomes index 0
    assert word_to_id[C.PAD_SYMBOL] == C.PAD_ID
    return word_to_id


def vocab_to_json(vocab: Vocab, path: str):
    """
    Saves vocabulary in human-readable json.

    :param vocab: Vocabulary mapping.
    :param path: Output file path.
    """
    with open(path, "w", encoding=C.VOCAB_ENCODING) as out:
        json.dump(vocab, out, indent=4, ensure_ascii=False)
        logger.info('Vocabulary saved to "%s"', path)


def vocab_from_json(path: str, encoding: str = C.VOCAB_ENCODING) -> Vocab:
    """
    Saves vocabulary in json format.

    :param path: Path to json file containing the vocabulary.
    :param encoding: Vocabulary encoding.
    :return: The loaded vocabulary.
    """
    with open(path, encoding=encoding) as inp:
        vocab = json.load(inp)
        logger.info('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab


def reverse_vocab(vocab: Vocab) -> InverseVocab:
    """
    Returns value-to-key mapping from key-to-value-mapping.

    :param vocab: Key to value mapping.
    :return: A mapping from values to keys.
    """
    return {v: k for k, v in vocab.items()}


if __name__ == '__main__':
    dev_amr_file = './data/dev.amr'
    dev_snt_file = './data/dev.snt'
    train_amr_file = './data/train.amr'
    train_snt_file = './data/train.snt'

    # vocab = build_from_paths(paths, num_words=30000, min_count=1)
    # vocab_to_json(vocab, "./data/vocab.json")
