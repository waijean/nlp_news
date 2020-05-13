import gc
import glob
import os
import logging.config
import pandas as pd

from utils.constants import (
    CSV_PARTITION_PATH,
    PARQUET_PARTITION_PATH,
    LOG_CONFIG_PATH,
    COL_SECTION,
    COL_DATE,
    COL_TITLE,
    COL_ARTICLE,
)

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def drop_rows_with_null_section(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        f"Number of null values in section column: {df[COL_SECTION].isna().sum()}"
    )
    new_df = df.dropna(subset=[COL_SECTION])
    return new_df


def set_col_section_as_categortical(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.astype({COL_SECTION: "category"})
    logger.info(f"Column section has been converted to: {new_df[COL_SECTION].dtype}")
    return new_df


def select_relevant_section(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df[
        df[COL_SECTION].isin(
            ["Market News", "World News", "Business News", "Wires", "Financials"]
        )
    ]
    logger.info(f"Number of rows selected: {len(new_df)}")
    return new_df


if __name__ == "__main__":
    for i, file in enumerate(glob.glob((f"{CSV_PARTITION_PATH}/news_csv*"))):
        logger.info(f"Reading {file}")
        df = pd.read_csv(
            file,
            usecols=[COL_DATE, COL_TITLE, COL_ARTICLE, COL_SECTION],
            dtype={COL_DATE: str, COL_TITLE: str, COL_ARTICLE: str, COL_SECTION: str,},
            engine="c",
            encoding="utf8",
            quoting=0,
            quotechar='"',
            doublequote=True,
            parse_dates=[COL_DATE],
        )
        processed_df = (
            df.pipe(drop_rows_with_null_section)
            .pipe(set_col_section_as_categortical)
            .pipe(select_relevant_section)
        )
        logger.info(f"Writing {file} to {PARQUET_PARTITION_PATH}")
        processed_df.to_parquet(
            os.path.join(PARQUET_PARTITION_PATH, f"news_parquet_0{i}")
        )
        del [[df, processed_df]]
        gc.collect()
