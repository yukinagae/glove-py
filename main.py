from argparse import ArgumentParser
import codecs
from collections import Counter
from scipy import sparse
import numpy as np
import logging
logger = logging.getLogger("glove-py")


def parse_args():
    parser = ArgumentParser(description=(
        'Build a GloVe vector-space model from the corpus'))
    parser.add_argument('corpus_path')

    vocab_parser = parser.add_argument_group('Vocabulary options')
    vocab_parser.add_argument('--vocab-path')

    cooccur_parser = parser.add_argument_group('Cooccurrence tracking options')
    cooccur_parser.add_argument('--cooccur-path')
    cooccur_parser.add_argument('-w', '--window-size', type=int, default=10)

    glove_parser = parser.add_argument_group('GloVe options')
    glove_parser.add_argument('-s', '--vector-size', type=int, default=100)
    glove_parser.add_argument('--iterations', type=int, default=25)
    glove_parser.add_argument('--learning-rate', type=float, default=0.05)

    return parser.parse_args()


def build_vocab(corpus_path):
    vocab = Counter()

    for line in codecs.open(corpus_path, encoding='utf-8'):
        tokens = line.strip().split()
        vocab.update(tokens)

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


def build_cooccur(vocab, corpus_path, window_size=10):

    #     vocab_size = len(vocab)
    vocab_size = 122
    cooccurrences = sparse.lil_matrix(
        (vocab_size, vocab_size), dtype=np.float64)

    for i, line in enumerate(codecs.open(corpus_path, encoding='utf-8')):
        logger.info(f"Building cooccurrence matrix: on line {i}")
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]
        logger.debug(token_ids)

        for center_i, center_id in enumerate(token_ids):

            # logger.debug(f"center_i: {center_i}, center_id: {center_id}")

            left_context_ids = token_ids[max(
                0, center_i - window_size): center_i]
            left_contexts_len = len(left_context_ids)

            # logger.debug(f"left_context_ids: {left_context_ids}")
            # logger.debug(f"left_contexts_len: {left_contexts_len}")

            for left_i, left_id in enumerate(left_context_ids):
                distance = left_contexts_len - left_i
            #     logger.debug(f"distance: {distance}")

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)
            #     logger.debug(f"increment: {increment}")

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    for i, (row, data) in enumerate(zip(cooccurrences.rows, cooccurrences.data)):
        for data_idx, j in enumerate(row):
            yield i, j, data[data_idx]


def main(args):

    corpus_path = args.corpus_path
    logger.debug(corpus_path)

    logger.info("Fetching vocab...")
    vocab = build_vocab(corpus_path)
    logger.info(f"Vocab has {len(vocab)} elements.")

    logger.info("Building cooccurrence...")
    cooccurrences = build_cooccur(vocab, corpus_path)
    logger.info(
        f"Cooccurrence list has {sum(1 for _ in cooccurrences)} pairs.")
    logger.debug(cooccurrences)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] %(message)s")
    main(parse_args())
