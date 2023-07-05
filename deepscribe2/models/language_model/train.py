from nltk.lm import MLE, Laplace
from typing import List, Dict, Any, Tuple
from itertools import chain
from nltk.util import everygrams


def unpadded_everygram(texts: List[List[str]], min_len: int = 1, max_len: int = -1):
    vocab = chain.from_iterable(texts)

    grams = [list(everygrams(seq, min_len=min_len, max_len=max_len)) for seq in texts]

    return grams, vocab
