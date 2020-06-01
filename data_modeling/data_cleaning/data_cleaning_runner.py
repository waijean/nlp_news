from data_modeling.data_cleaning.spacy_tokenizer import spacy_tokenizer
from utils.constants import (
    PARQUET_PARTITION_V2_PATH,
    COL_TITLE,
    COL_DATE,
    PROCESSED_PRICE_PATH,
    COL_SIGN,
    CLEANED_NEWS_TITLE_PATH,
)
from utils.pipeline_abc import ETLPipeline
import pandas as pd


class CleaningPipelineNews(ETLPipeline):

    _news_df: pd.DataFrame
    _price_df: pd.DataFrame
    _merged_df: pd.DataFrame

    @classmethod
    def extract(cls):
        # extract news df
        cls._read_path = PARQUET_PARTITION_V2_PATH
        super().extract()
        cls._news_df = pd.read_parquet(
            cls._read_path, columns=[COL_DATE, COL_TITLE]
        ).set_index(COL_DATE)

        # extract price df
        cls._read_path = PROCESSED_PRICE_PATH
        super().extract()
        cls._price_df = pd.read_parquet(cls._read_path, columns=[COL_SIGN])

    @classmethod
    def transform(cls):
        super().transform()
        cls._merged_df = cls._news_df.merge(
            cls._price_df, how="inner", left_index=True, right_index=True
        )
        cls._merged_df[COL_TITLE] = spacy_tokenizer(cls._merged_df[COL_TITLE])

    @classmethod
    def load(cls):
        cls._write_path = CLEANED_NEWS_TITLE_PATH
        super().load()
        cls._merged_df.to_parquet(cls._write_path)

    @classmethod
    def main(cls):
        super().main()


if __name__ == "__main__":
    CleaningPipelineNews.main()
