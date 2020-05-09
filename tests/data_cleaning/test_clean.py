import nltk


def test_import_stopwords():
    stopword = nltk.corpus.stopwords.words("english")
    assert stopword
