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

# news df
COL_SECTION = "section"
COL_TITLE = "title"

# training df
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
DATA_PIPELINE_PATH = os.path.join(ROOT_DIR_PATH, "data_pipeline/data")
CSV_PARTITION_PATH = os.path.join(ROOT_DIR_PATH, "data_pipeline/data/partition_csv")
PARQUET_PARTITION_V1_PATH = os.path.join(
    ROOT_DIR_PATH, "data_pipeline/data/partition_v1.parquet"
)
PARQUET_PARTITION_V2_PATH = os.path.join(
    ROOT_DIR_PATH, "data_pipeline/data/partition_v2.parquet"
)
DATA_PREPROCESSING_PATH = os.path.join(ROOT_DIR_PATH, "data_preprocessing/data")
SQL_DATABASE_PATH = "D:/sqlite/db"
TEST_SQL_DATABASE_PATH = os.path.join(SQL_DATABASE_PATH, "test.db")
ARTIFACT_PATH = "D:/mlflow"
