import logging.config
import os

from utils.constants import (
    DATA_PIPELINE_PATH,
    LOG_CONFIG_PATH,
    COL_DATE,
    COL_TITLE,
    COL_ARTICLE,
    COL_SECTION,
)

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def set_up_client():
    logger.info("Setting up Dask Client")
    cluster = LocalCluster(dashboard_address="localhost:8788")
    Client(cluster)
    logger.info("Diagnostic Dashboard link: http://localhost:8788/status")


def read_csv_dask(path):
    """
    Best practices:
    - Select only the columns that you plan to use
    - Specify dtypes directly using the dtype keyword. For datetime dtype, use parse_dates.

    Notes:
    - For engine, 'c' is faster, while 'python' is currently more feature-complete
    - Set quoting as QUOTE_MINIMAL (0) as the csv file only quote fields with special characters
    - Set quotechar as "
    - Set doublequote as True to interpret two consecutive quotechar elements INSIDE a field as a single quotechar element

    Improvement
    - Set error_bad_lines as False to skip error lines. For example, field larger than field limit (131072) and unexpected end of data
    - Cannot parse_dates at reading because error lines cause confusion
    """
    df = dd.read_csv(
        path,
        usecols=[COL_DATE, COL_TITLE, COL_ARTICLE, COL_SECTION],
        dtype={COL_DATE: str, COL_TITLE: str, COL_ARTICLE: str, COL_SECTION: str},
        # parse_dates=[COL_DATE],
        engine="python",
        encoding="utf8",
        quoting=0,
        quotechar='"',
        doublequote=True,
        error_bad_lines=False,
    )
    logger.info(f"Number of partitions: {df.npartitions}")
    logger.info(f"Divisions of partitions: {df.divisions}")
    return df


def drop_rows_with_null_section_dask():
    logger.info(
        f"Number of null values in section column: {df[COL_SECTION].isna().sum().compute()}"
    )
    new_df = df.dropna(subset=[COL_SECTION])
    return new_df


def process_date(df):
    """
    Filter out rows where the date column don't start with a number
    Convert date column dtype to datetime
    """
    logger.info("Processing date")
    normal_date_df = df[df[COL_DATE].str.contains("^[0-9]+", na=False)]
    normal_date_df[COL_DATE] = dd.to_datetime(normal_date_df[COL_DATE])
    logger.info("Date column converted to datetime format")
    return normal_date_df


if __name__ == "__main__":
    set_up_client()
    df = read_csv_dask(
        os.path.join(DATA_PIPELINE_PATH, "processed-all-the-news-2-1.csv")
    )
    processed_df = df.pipe(drop_rows_with_null_section_dask).pipe(process_date)
    processed_df.to_parquet(
        os.path.join(DATA_PIPELINE_PATH, "news_v1.parquet"),
        engine="pyarrow",
        write_index=False,
    )
