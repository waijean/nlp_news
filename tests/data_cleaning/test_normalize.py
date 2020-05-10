from data_cleaning.normalize import stemming, lemmatize


def test_stemming(tokenized_list_no_stopwords, tokenized_list_stemming):
    actual_tokenized_list_stemming = stemming(tokenized_list_no_stopwords)
    assert actual_tokenized_list_stemming == tokenized_list_stemming


def test_lemmatize(tokenized_list_no_stopwords, tokenized_list_lemmatize):
    actual_tokenized_list_lemmatize = lemmatize(tokenized_list_no_stopwords)
    assert actual_tokenized_list_lemmatize == tokenized_list_lemmatize
