from data_modeling.data_preprocessing.textblob_sentiment import (
    calculate_polarity,
    calculate_subjectivity,
    add_col_polarity,
    add_col_subjectivity,
)
import pandas as pd
import numpy as np

from utils.constants import COL_ARTICLE_POLARITY, COL_ARTICLE, COL_ARTICLE_SUBJECTIVITY


def test_calculate_polarity(input_text):
    polarity = calculate_polarity(input_text)
    assert polarity == -0.325


def test_calculate_subjectivity(input_text):
    subjectivity = calculate_subjectivity(input_text)
    assert subjectivity == 0.52


def test_add_col_polarity(input_text_df):
    actual_df = add_col_polarity(input_text_df, COL_ARTICLE)
    expected_col = pd.Series([-0.325], name=COL_ARTICLE_POLARITY)
    assert all(np.isclose(actual_df[COL_ARTICLE_POLARITY], expected_col))


def test_add_col_subjectivity(input_text_df):
    actual_df = add_col_subjectivity(input_text_df, COL_ARTICLE)
    expected_col = pd.Series([0.52], name=COL_ARTICLE_SUBJECTIVITY)
    assert all(np.isclose(actual_df[COL_ARTICLE_SUBJECTIVITY], expected_col))
