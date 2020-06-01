import argparse
import gc
import glob
import os
import logging.config

import pandas as pd

from utils.pipeline_abc import ETLPipeline
from utils.constants import (
    NEWS_PARTITION_CSV_PATH,
    PARQUET_PARTITION_V2_PATH,
    LOG_CONFIG_PATH,
    COL_SECTION,
    COL_DATE,
    COL_TITLE,
    COL_ARTICLE,
    PARQUET_PARTITION_V1_PATH,
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


class DataPipelineNews(ETLPipeline):

    _df: pd.DataFrame
    _processed_df: pd.DataFrame
    _version: int
    _i: int

    @classmethod
    def extract(cls):
        super().extract()
        cls._df = pd.read_csv(
            cls._read_path,
            usecols=[COL_DATE, COL_TITLE, COL_ARTICLE, COL_SECTION],
            dtype={COL_DATE: str, COL_TITLE: str, COL_ARTICLE: str, COL_SECTION: str,},
            engine="c",
            encoding="utf8",
            quoting=0,
            quotechar='"',
            doublequote=True,
            parse_dates=[COL_DATE],
        )

    @classmethod
    def transform(cls):
        super().transform()

    @classmethod
    def load(cls):
        def get_path_version(version: int) -> str:
            switcher = {1: PARQUET_PARTITION_V1_PATH, 2: PARQUET_PARTITION_V2_PATH}
            path = switcher.get(version)
            if path is None:
                raise KeyError(f"Path version {version} is not available")
            return path

        cls._write_path = get_path_version(cls._version)
        super().load()
        cls._processed_df.to_parquet(
            os.path.join(cls._write_path, f"news_parquet_0{cls._i}")
        )
        del [[cls._df, cls._processed_df]]
        gc.collect()

    @classmethod
    def main(cls):
        for i, file in enumerate(glob.glob(f"{NEWS_PARTITION_CSV_PATH}/news_csv*")):
            cls._i = i
            cls._read_path = file
            super().main()


class DataPipelineNewsV1(DataPipelineNews):
    _version = 1

    @classmethod
    def transform(cls):
        super().transform()
        processed_df = cls._df.pipe(drop_rows_with_null_section).pipe(
            set_col_section_as_categortical
        )
        cls._processed_df = processed_df


class DataPipelineNewsV2(DataPipelineNews):
    """
    Filter relevant sections in addition to v1
    """

    _version = 2

    @classmethod
    def transform(cls):
        super().transform()
        processed_df = (
            cls._df.pipe(drop_rows_with_null_section)
            .pipe(set_col_section_as_categortical)
            .pipe(select_relevant_section)
        )
        cls._processed_df = processed_df


def get_pipeline_version(version: int):
    logger.info(f"Getting pipeline version {version}")
    switcher = {1: DataPipelineNewsV1, 2: DataPipelineNewsV2}
    pipeline = switcher.get(version)
    if pipeline is None:
        raise KeyError(f"Pipeline version {version} is not available")
    return pipeline


parser = argparse.ArgumentParser()

parser.add_argument("--version", required=True)

args = parser.parse_args()

if __name__ == "__main__":
    pipeline = get_pipeline_version(int(args.version))
    pipeline.main()
