from data_modeling.data_cleaning.spacy_tokenizer import spacy_tokenizer
from utils.constants import COL_ARTICLE


def test_spacy_tokenizer(input_text_df, expected_spacy_tokenized_col):
    actual_tokens_col = spacy_tokenizer(input_text_df[COL_ARTICLE])
    assert actual_tokens_col == expected_spacy_tokenized_col
