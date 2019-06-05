from argparse import ArgumentParser
import codecs
import logging
logger = logging.getLogger("glove-py")


def parse_args():
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
                     'provided corpus'))
    parser.add_argument('corpus', metavar='corpus_path',
                        type=lambda path: codecs.open(path, encoding='utf-8'))

    vocab_parser = parser.add_argument_group('Vocabulary options')
    vocab_parser.add_argument('--vocab-path',
                              help=('Path to vocabulary file. If this path '
                                    'exists, the vocabulary will be loaded '
                                    'from the file. If it does not exist, '
                                    'the vocabulary will be written to this '
                                    'file.'))

    cooccur_parser = parser.add_argument_group('Cooccurrence tracking options')
    cooccur_parser.add_argument('--cooccur-path',
                                help=('Path to cooccurrence matrix file. If '
                                      'this path exists, the matrix will be '
                                      'loaded from the file. If it does not '
                                      'exist, the matrix will be written to '
                                      'this file.'))
    cooccur_parser.add_argument('-w', '--window-size', type=int, default=10,
                                help=('Number of context words to track to '
                                      'left and right of each word'))
    cooccur_parser.add_argument('--min-count', type=int, default=10,
                                help=('Discard cooccurrence pairs where at '
                                      'least one of the words occurs fewer '
                                      'than this many times in the training '
                                      'corpus'))

    glove_parser = parser.add_argument_group('GloVe options')
    glove_parser.add_argument('--vector-path',
                              help=('Path to which to save computed word '
                                    'vectors'))
    glove_parser.add_argument('-s', '--vector-size', type=int, default=100,
                              help=('Dimensionality of output word vectors'))
    glove_parser.add_argument('--iterations', type=int, default=25,
                              help='Number of training iterations')
    glove_parser.add_argument('--learning-rate', type=float, default=0.05,
                              help='Initial learning rate')
    glove_parser.add_argument('--save-often', action='store_true',
                              default=False,
                              help=('Save vectors after every training '))

    return parser.parse_args()


def main(args):
    logger.debug(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] %(message)s")
    main(parse_args())
