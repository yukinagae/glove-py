from argparse import ArgumentParser
import codecs
import logging
logger = logging.getLogger("glove-py")


def parse_args():
    parser = ArgumentParser(description=(
        'Build a GloVe vector-space model from the corpus'))
    parser.add_argument('corpus',
                        metavar='corpus_path',
                        type=lambda path: codecs.open(path, encoding='utf-8'))

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


def main(args):
    logger.debug(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] %(message)s")
    main(parse_args())
