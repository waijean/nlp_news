from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data_modeling.data_cleaning.nltk_tokenizer import nltk_tokenize
from data_modeling.data_cleaning.normalize import lemmatize

from scipy.sparse import csr_matrix
import pandas as pd


class CustomVectorizer(CountVectorizer):
    """
    Defines a CustomVectorizer class which inherits from CountVectorizer in order to overwrite its build_analyzer
    method and use its _word_ngrams built in method to extract n-grams

    https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af
    """

    def build_analyzer(self):
        def analyser(text: str) -> List:
            """
            Create the analyzer that will be returned by build_analyzer method

            This analyser is split into three stages
            1. Tokenize
            2. Lemmatize
            3. Extract n-grams
            """
            tokens = nltk_tokenize(text)
            lemmatized_tokens = lemmatize(tokens)

            return self._word_ngrams(lemmatized_tokens)

        return analyser


def count_vectorize(
    text_col: pd.Series, ngram_range: tuple
) -> Tuple[CustomVectorizer, csr_matrix]:
    """
    Instantiate the Custom Vectorizer, fit_transform on the text column and return sparse matrix
    """
    count_vect = CustomVectorizer(ngram_range=ngram_range)
    X_count = count_vect.fit_transform(text_col)
    return count_vect, X_count


def transform_tfidf(X_count: csr_matrix) -> csr_matrix:
    """
    Transform a vectorized count matrix to a tf-idf matrix
    """
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_count)
    return X_tfidf
