import os

import pandas as pd
from data_modeling.data_preprocessing.textblob_sentiment import (
    add_col_polarity,
    add_col_subjectivity,
)
from utils.constants import (
    COL_TITLE,
    COL_ARTICLE,
    COL_DATE,
    COL_TITLE_POLARITY,
    COL_TITLE_SUBJECTIVITY,
    COL_ARTICLE_POLARITY,
    COL_ARTICLE_SUBJECTIVITY,
    DATA_PREPROCESSING_PATH,
    PARQUET_PARTITION_V2_PATH,
    LOG_CONFIG_PATH,
    NEWS_FEATURE_PATH,
)
import logging.config

from utils.pipeline_abc import Pipeline

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def calculate_average_value_per_day(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating average value for each day")
    new_df = df.groupby(COL_DATE).agg(
        {
            COL_TITLE_POLARITY: "mean",
            COL_TITLE_SUBJECTIVITY: "mean",
            COL_ARTICLE_POLARITY: "mean",
            COL_ARTICLE_SUBJECTIVITY: "mean",
        }
    )
    new_df.columns = [
        COL_TITLE_POLARITY,
        COL_TITLE_SUBJECTIVITY,
        COL_ARTICLE_POLARITY,
        COL_ARTICLE_SUBJECTIVITY,
    ]
    return new_df


class PreprocessingPipelineNews(Pipeline):

    _df: pd.DataFrame
    _processed_df: pd.DataFrame

    @classmethod
    def extract(cls):
        cls._read_path = PARQUET_PARTITION_V2_PATH
        super().extract()
        cls._df = pd.read_parquet(cls._read_path)

    @classmethod
    def transform(cls):
        super().transform()
        processed_df = (
            cls._df.pipe(add_col_polarity, COL_TITLE)
            .pipe(add_col_subjectivity, COL_TITLE)
            .pipe(add_col_polarity, COL_ARTICLE)
            .pipe(add_col_subjectivity, COL_ARTICLE)
            .pipe(calculate_average_value_per_day)
        )
        cls._processed_df = processed_df

    @classmethod
    def load(cls):
        cls._write_path = NEWS_FEATURE_PATH
        super().load()
        cls._processed_df.to_parquet(cls._write_path)

    @classmethod
    def main(cls):
        super().main()


if __name__ == "__main__":
    PreprocessingPipelineNews.main()
