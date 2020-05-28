from data_modeling.data_cleaning.nltk_tokenizer import (
    remove_punctuation,
    remove_stopwords,
    tokenize,
    nltk_tokenize,
    remove_token_punctuation,
)


def test_remove_punctuation(input_text, input_text_no_punc):
    actual_string_no_punct = remove_punctuation(input_text)
    assert actual_string_no_punct == input_text_no_punc


def test_tokenize(input_text, tokenized_list):
    actual_tokenized_list = tokenize(input_text)
    assert actual_tokenized_list == tokenized_list


def test_remove_token_punctuation(tokenized_list, tokenized_list_no_punc):
    actual_tokenized_list_no_punc = remove_token_punctuation(tokenized_list)
    assert actual_tokenized_list_no_punc == tokenized_list_no_punc


def test_remove_stopwords(tokenized_list_no_punc, tokenized_list_no_stopwords):
    actual_tokenized_list_no_stopwords = remove_stopwords(tokenized_list_no_punc)
    assert actual_tokenized_list_no_stopwords == tokenized_list_no_stopwords


def test_full_tokenize(input_text, tokenized_list_no_stopwords):
    actual_tokenized_list_no_stopwords = nltk_tokenize(input_text)
    assert actual_tokenized_list_no_stopwords == tokenized_list_no_stopwords
