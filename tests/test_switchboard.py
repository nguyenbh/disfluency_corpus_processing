# -*- coding: utf-8 -*-
import os
import pytest
from disfluency_corpus_processing.switchboard import conll_format

__author__ = "Nguyen Bach"
__copyright__ = "Nguyen Bach"
__license__ = "mit"


def test_conll_format():
    segment = ['I/O', 'I/@dis/@dis', 'love/O', 'dog/0']

    out = conll_format(segment)
    expected = [('I', 'O', 'O'), ('I', '@dis', '@dis'),
                ('love', 'O', 'O'), ('dog', '0', 'O')]

    assert out == expected

# implement more tests below
