# -*- coding: utf-8 -*-

import pytest
from disfluency_corpus_processing.skeleton import fib

__author__ = "Nguyen Bach"
__copyright__ = "Nguyen Bach"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
