import os
from typing import List

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime

from utils.constants import COL_CATEGORY, COL_ARTICLE, COL_HEADLINE, COL_DATE, DATA_PATH

seed_urls = [
    "https://inshorts.com/en/read/politics",
    "https://inshorts.com/en/read/business",
    "https://inshorts.com/en/read/world",
]


def build_dataset(seed_urls: List) -> List:
    news_data = []
    for url in seed_urls:
        news_category = url.split("/")[-1]
        data = requests.get(url)
        soup = BeautifulSoup(data.content, "html.parser")

        news_articles = [
            {
                "news_headline": headline.find(
                    "span", attrs={"itemprop": "headline"}
                ).string,
                "news_article": article.find(
                    "div", attrs={"itemprop": "articleBody"}
                ).string,
                "news_category": news_category,
            }
            for headline, article in zip(
                soup.find_all("div", class_=["news-card-title news-right-box"]),
                soup.find_all("div", class_=["news-card-content news-right-box"]),
            )
        ]
        news_data.extend(news_articles)

    return news_data


def create_df_from_list(news_list: List) -> pd.DataFrame:
    df = pd.DataFrame(news_list, columns=[COL_HEADLINE, COL_ARTICLE, COL_CATEGORY])
    return df


def add_date_col(df: pd.DataFrame) -> pd.DataFrame:
    kwargs = {COL_DATE: np.datetime64(datetime.now()).astype("datetime64[ms]")}
    new_df = df.assign(**kwargs)
    return new_df


def export_df(df: pd.DataFrame, path: str, file_name: str):
    file_path = os.path.join(path, file_name)
    df.to_parquet(file_path)


if __name__ == "__main__":
    news_list = build_dataset(seed_urls)
    news_df = create_df_from_list(news_list)
    news_df = add_date_col(news_df)
    export_df(
        news_df,
        path=DATA_PATH,
        file_name=f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_data_v1.parquet",
    )
