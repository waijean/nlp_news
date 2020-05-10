from typing import List

from nltk import PorterStemmer, WordNetLemmatizer


def stemming(tokenized_list: List) -> List:
    """
    Take a list of words, perform stemming and return a list of stemmed words

    Faster but stemmed word might not be an actual word. For example, stemming announcements will return annouc
    """
    ps = PorterStemmer()
    tokenized_list_stemming = [ps.stem(word) for word in tokenized_list]
    return tokenized_list_stemming


def lemmatize(tokenized_list: List) -> List:
    """
    Take a list of words, lemamatize the words and return a list of lemmatized words

    Return an actual word but slower since it uses WordNet corpus
    """
    wn = WordNetLemmatizer()
    tokenized_list_lemmatize = [wn.lemmatize(word) for word in tokenized_list]
    return tokenized_list_lemmatize
