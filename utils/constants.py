import os

import git

# price_df
COL_PRICE_DATE = "Date"
COL_OPEN = "Open"
COL_CLOSE = "Close"
COL_VOLUME = "Volume"
COL_PREVIOUS_CLOSE = "previous_close"
COL_ABSOLUTE_RETURN = "absolute_return"
COL_SIGN = "sign"

# news df(web scrape)
COL_HEADLINE = "headline"
COL_ARTICLE = "article"
COL_CATEGORY = "category"
COL_DATE = "date"

# news df columns
COL_SECTION = "section"
COL_TITLE = "title"

# training df columns
COL_TITLE_POLARITY = "title_polarity"
COL_TITLE_SUBJECTIVITY = "title_subjectivity"
COL_ARTICLE_POLARITY = "article_polarity"
COL_ARTICLE_SUBJECTIVITY = "article_subjectivity"

repo = git.Repo(".", search_parent_directories=True)
ROOT_DIR_PATH = repo.working_tree_dir

LOG_CONFIG_PATH = os.path.join(ROOT_DIR_PATH, "utils/logging.conf")
WEB_SCRAPING_DATA_PATH = os.path.join(
    ROOT_DIR_PATH, "data_engineering/web_scraping/data"
)


# news df path
DATA_PIPELINE_NEWS_PATH = os.path.join(
    ROOT_DIR_PATH, "data_engineering/data_pipeline/news/data"
)
RAW_NEWS_PATH = os.path.join(DATA_PIPELINE_NEWS_PATH, "all-the-news-2-1.csv")
PROCESSED_NEWS_PATH = os.path.join(
    DATA_PIPELINE_NEWS_PATH, "processed-all-the-news-2-1.csv"
)
NEWS_PARTITION_CSV_PATH = os.path.join(DATA_PIPELINE_NEWS_PATH, "partition_csv")
PARQUET_PARTITION_V1_PATH = os.path.join(
    DATA_PIPELINE_NEWS_PATH, "partition_v1.parquet"
)
PARQUET_PARTITION_V2_PATH = os.path.join(
    DATA_PIPELINE_NEWS_PATH, "partition_v2.parquet"
)

# archive df path
ARCHIVE_PATH = os.path.join(ROOT_DIR_PATH, "data_pipeline/data")


# price df path
DATA_PIPELINE_PRICE_PATH = os.path.join(
    ROOT_DIR_PATH, "data_engineering/data_pipeline/price/data"
)
RAW_PRICE_PATH = os.path.join(
    DATA_PIPELINE_PRICE_PATH, "VEVE_HistoricPrices_20141001-20200505.csv"
)
PROCESSED_PRICE_PATH = os.path.join(DATA_PIPELINE_PRICE_PATH, "processed_price.parquet")

# data cleaning df
DATA_CLEANING_PATH = os.path.join(ROOT_DIR_PATH, "data_cleaning/data")


# data preprocessing df
DATA_PREPROCESSING_PATH = os.path.join(ROOT_DIR_PATH, "data_preprocessing/data")
NEWS_FEATURE_PATH = os.path.join(DATA_PREPROCESSING_PATH, "news_feature.parquet")

# data cleaning df
SQL_DATABASE_PATH = "D:/sqlite/db"
TEST_SQL_DATABASE_PATH = os.path.join(SQL_DATABASE_PATH, "test.db")
ARTIFACT_PATH = os.path.join(ROOT_DIR_PATH, "mlruns")
