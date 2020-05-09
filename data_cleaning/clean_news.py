import re
import string
from itertools import filterfalse
from typing import List

import nltk
import regex
from nltk import word_tokenize, WordPunctTokenizer


def remove_punctuation(text: str) -> str:
    """
    Take an input string, split into individual character, remove the punctuations and return a string

    Use regex \p{P} for any punctuation character to include special punctuations character (en/em dash, opening/closing quote)
    """
    text_no_punc = regex.sub("\p{P}+", "", text)
    return text_no_punc


def tokenize(text: str) -> List:
    """
    Take an input string, convert to lowercase, and return a list of words

    Choices:
    1. re.split('\W+', text.lower())
    2. word_tokenize(text.lower())
    """
    tk = WordPunctTokenizer()
    tokens = tk.tokenize(text.lower())
    return tokens


def remove_token_punctuation(tokenized_list: List) -> List:
    r = regex.compile("\p{P}+")
    tokenized_list_without_punc = list(filterfalse(r.match, tokenized_list))
    return tokenized_list_without_punc


def remove_stopwords(tokenized_list: List) -> List:
    """
    Take a list of words, remove the stopwords and return a list of words

    Notes:
    - nltk stopwords are all lowercases
    - nltk stopwords contain words with contractions (you're, that'll, won't) but not words without contractions (youre, thatll, wont)
    - nltk stopwords contain pure contractions (re, ll, t), but not contractions with punctuations ('re, 'll, 't)
    """
    stopwords = nltk.corpus.stopwords.words("english")
    tokenized_list_no_stopwords = [
        word for word in tokenized_list if word not in stopwords
    ]
    return tokenized_list_no_stopwords


def full_tokenize(text: str) -> List:
    tokens = tokenize(text)
    tokens_no_punc = remove_token_punctuation(tokens)
    tokens_no_stopwords = remove_stopwords(tokens_no_punc)
    return tokens_no_stopwords
