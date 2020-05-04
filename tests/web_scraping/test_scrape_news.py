import os

import pandas as pd

from utils.constants import COL_DATE
from web_scraping.scrape_news import create_df_from_list, add_date_col, export_df


def test_create_df(expected_news_df):
    news_list = [
        ["headline_1", "article_1", "category_1"],
        ["headline_2", "article_2", "category_2"],
    ]
    actual_news_df = create_df_from_list(news_list)
    assert actual_news_df.equals(expected_news_df)


def test_add_date_col(expected_news_df):
    actual_news_df_with_date = add_date_col(expected_news_df)
    assert COL_DATE in actual_news_df_with_date


def test_export_df(working_dir, expected_news_df):
    export_df(expected_news_df, working_dir, "data")
    file_path = os.path.join(working_dir, "data")
    actual_news_df = pd.read_parquet(file_path, engine="pyarrow")
    assert actual_news_df.equals(expected_news_df)
