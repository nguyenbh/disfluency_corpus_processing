# -*- coding: utf-8 -*-
import os
import pytest
from disfluency_corpus_processing.corpus import Corpus

__author__ = "Nguyen Bach"
__copyright__ = "Nguyen Bach"
__license__ = "mit"

def test_corpus_init():
    swbd = Corpus(input_file=os.path.abspath(__file__),
                  file_type='dps',
                  punctuation=True)
    print(type(swbd))
    assert isinstance(swbd, Corpus)


# implement more tests below