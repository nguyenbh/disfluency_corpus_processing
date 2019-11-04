"""Split Switchboard corpus."""
import argparse
from fnmatch import fnmatch
import logging
import os
import pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from disfluency_corpus_processing.corpus import Corpus


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def conll_format(segment):
    """Prepare data in CoNLL 2002 format."""
    sent = []
    for w in segment:
        fields = w.split('/')
        if len(fields) == 2:
            sent.append((fields[0], fields[1], 'O'))
        elif len(fields) == 3:
            sent.append((fields[0], fields[1], '@dis'))
        else:
            logging.error('Bad data! %s' % w)
            logging.error('Bad segment: %s' % segment)
            return None

    return sent


def johson_charniak_split_files(root):
    """Follow Johnson & Charniak, 2004 to get train and test split
    for disfluency detection task."""
    test_pattern = 'sw4[0-1]*.dps'
    train_pattern = '*.dps'

    test, train = [], []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, test_pattern):
                filename = os.path.join(path, name)
                logging.info(filename)
                test.append(filename)
            elif fnmatch(name, train_pattern):
                filename = os.path.join(path, name)
                logging.info(filename)
                train.append(filename)

    logging.info('Train: %d files; Test: %d files' % (len(train), len(test)))

    return train, test


def bitext_format(segment):
    """Prepare data in bitext format.

    bitext format is a tuple of (original text, disfluency removed text)
    """
    raw = []
    clean = []
    for w in segment:
        fields = w.split('/')
        if len(fields) == 2:
            raw.append(fields[0])
            clean.append(fields[0])
        elif len(fields) == 3:
            raw.append(fields[0])
        else:
            logging.error('Bad data! %s' % w)
            logging.error('Bad segment: %s' % segment)
            return None

    return (' '.join(raw), ' '.join(clean))


def get_data(filelist, filetype='conll'):
    conll = []
    for i, filename in enumerate(tqdm(filelist)):
        swbd = Corpus(filename,
                      'dps',
                      punctuation=True)
        for segment in swbd.parse():
            if filetype == 'bitext':
                output = bitext_format(segment)
            else:
                output = conll_format(segment)
            if output:
                conll.append(output)

    return conll


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, required=True,
                        help="The LDC Switchboard corpus directory which contains dysfl/ directory")

    args = parser.parse_args()

    # if args.d is None:
    #     switchboard_treebank_dir = '/Users/nguyen.bach/Desktop/disfluencies/treebank3/treebank_3/")'
    # else:
    #     switchboard_treebank_dir = args.d

    train, test = johson_charniak_split_files(args.d)

    conll_test = get_data(test)
    conll_train = get_data(train)

    # make train and validation set
    train, valid = train_test_split(conll_train, test_size=0.03, random_state=42)

    logging.info('Train: %d segments; Validation: %d; Test: %d segments' %
                 (len(conll_train), len(valid), len(conll_test)))

    with open('train.punct.pickle', 'wb') as fh:
        pickle.dump(train, fh, protocol=3)
    with open('validation.punct.pickle', 'wb') as fh:
        pickle.dump(valid, fh, protocol=3)
    with open('test.punct.pickle', 'wb') as fh:
        pickle.dump(conll_test, fh, protocol=3)

if __name__ == "__main__":
    main()
