import gc
import os
import logging.config
import pandas as pd

from utils.constants import CSV_PARTITION_PATH, PARQUET_PARTITION_PATH, LOG_CONFIG_PATH

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def drop_rows_with_null_section(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        f"Number of null values in section column: {df['section'].isna().sum()}"
    )
    new_df = df.dropna(subset=["section"])
    return new_df


if __name__ == "__main__":
    for i in range(1, 9):
        logger.info(f"Reading news_csv_0{i} from {CSV_PARTITION_PATH}")
        df = pd.read_csv(
            os.path.join(CSV_PARTITION_PATH, f"news_csv_0{i}"),
            usecols=["date", "title", "article", "section"],
            dtype={"date": str, "title": str, "article": str, "section": str,},
            engine="c",
            encoding="utf8",
            quoting=0,
            quotechar='"',
            doublequote=True,
            parse_dates=["date"],
        )
        processed_df = df.pipe(drop_rows_with_null_section)
        logger.info(f"Writing news_csv_0{i} to {PARQUET_PARTITION_PATH}")
        processed_df.to_parquet(
            os.path.join(PARQUET_PARTITION_PATH, f"news_parquet_0{i}")
        )
        del [[df, processed_df]]
        gc.collect()
