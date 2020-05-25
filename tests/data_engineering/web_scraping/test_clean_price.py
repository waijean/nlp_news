import pandas as pd
import numpy as np

from data_engineering.web_scraping.clean_price import drop_null_rows


def test_drop_null_rows(expected_price_df):
    na_row = pd.DataFrame({"Date": ["2020-01-04"], "Close": [np.nan],})
    price_df_with_na = expected_price_df.append(na_row)
    actual_df = drop_null_rows(price_df_with_na)
    assert actual_df.equals(expected_price_df)
