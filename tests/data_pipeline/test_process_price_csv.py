from data_pipeline.process_price_csv import (
    remove_zero_volume_rows,
    add_col_absolute_return,
    add_col_sign,
    remove_zero_return_rows,
)
from utils.constants import COL_ABSOLUTE_RETURN, COL_SIGN

import pandas as pd
import numpy as np


def test_remove_zero_volume_rows(input_price_df):
    expected_df = input_price_df.drop(3)
    actual_df = remove_zero_volume_rows(input_price_df)
    assert actual_df.equals(expected_df)


def test_add_col_absolute_return(input_price_df):
    expected_col = pd.Series([0.1, -0.1, 50.4, 0.0], name=COL_ABSOLUTE_RETURN)
    actual_df = add_col_absolute_return(input_price_df)
    assert all(np.isclose(actual_df[COL_ABSOLUTE_RETURN], expected_col))


def test_remove_zero_return_rows(input_price_df):
    df_with_absolute_return = add_col_absolute_return(input_price_df)
    actual_df = remove_zero_return_rows(df_with_absolute_return)
    expected_col = pd.Series([0.1, -0.1, 50.4], name=COL_ABSOLUTE_RETURN)
    assert all(np.isclose(actual_df[COL_ABSOLUTE_RETURN], expected_col))


def test_add_col_sign(input_price_df):
    df_with_absolute_return = add_col_absolute_return(input_price_df)
    actual_df = add_col_sign(df_with_absolute_return)
    expected_col = pd.Series([1, -1, 1, np.nan], name=COL_SIGN)
    assert actual_df[COL_SIGN].equals(expected_col)
