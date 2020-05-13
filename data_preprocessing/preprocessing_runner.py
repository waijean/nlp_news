import os

import pandas as pd
from data_preprocessing.sentiment import add_col_polarity, add_col_subjectivity
from utils.constants import (
    COL_TITLE,
    COL_ARTICLE,
    COL_DATE,
    COL_TITLE_POLARITY,
    COL_TITLE_SUBJECTIVITY,
    COL_ARTICLE_POLARITY,
    COL_ARTICLE_SUBJECTIVITY,
    DATA_PREPROCESSING_PATH,
    PARQUET_PARTITION_PATH,
    LOG_CONFIG_PATH,
)
import logging.config

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


if __name__ == "__main__":
    # TODO convert reading and logging into utils function
    logger.info(f"Reading {PARQUET_PARTITION_PATH}")
    df = pd.read_parquet(PARQUET_PARTITION_PATH)
    processed_df = (
        df.pipe(add_col_polarity, COL_TITLE)
        .pipe(add_col_subjectivity, COL_TITLE)
        .pipe(add_col_polarity, COL_ARTICLE)
        .pipe(add_col_subjectivity, COL_ARTICLE)
        .pipe(calculate_average_value_per_day)
    )
    logger.info(f"Writing to {DATA_PREPROCESSING_PATH}")
    processed_df.to_parquet(
        os.path.join(DATA_PREPROCESSING_PATH, "news_feature.parquet")
    )
