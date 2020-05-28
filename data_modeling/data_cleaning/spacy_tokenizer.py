from typing import Iterable, List

import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()


def spacy_tokenizer(col: Iterable) -> List:
    """
    1. Tokenize
    2. Remove stopwords
    3. Remove punctuations
    4. Remove whitespace characters (ex. \n, \t)
    5. Lemmatize
    6. Lowercase
    #TODO strip accents

    Args:
        col: An iterable of strings to pass into tokenizer

    Returns: A list of tokenized strings which are joined with whitespaces

    """
    tokens_col = []
    for doc in nlp.pipe(col, disable=["tagger", "parser", "ner"]):
        tokens_list = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        tokens_string = " ".join(tokens_list)
        tokens_col.append(tokens_string)
    return tokens_col
