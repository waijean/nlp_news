import os
from typing import Any, Dict

import git

# price_df
from sklearn.metrics import recall_score, f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold

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
DATA_CLEANING_PATH = os.path.join(ROOT_DIR_PATH, "data_modeling/data_cleaning/data")
CLEANED_NEWS_TITLE_PATH = os.path.join(DATA_CLEANING_PATH, "cleaned_news_title.parquet")

# data preprocessing df
DATA_PREPROCESSING_PATH = os.path.join(
    ROOT_DIR_PATH, "data_modeling/data_preprocessing/data"
)
NEWS_FEATURE_PATH = os.path.join(DATA_PREPROCESSING_PATH, "news_feature.parquet")

# mlrun path
SQL_DATABASE_PATH = "/D:/sqlite/db"
TEST_SQL_DATABASE_PATH = "sqlite://" + os.path.join(SQL_DATABASE_PATH, "test.db")
MLRUN_SQL_DATABASE_PATH = "sqlite://" + os.path.join(SQL_DATABASE_PATH, "mlrun.db")
TEST_ARTIFACT_PATH = "file:" + os.path.join(ROOT_DIR_PATH, "mlruns/test")
ARTIFACT_PATH = "file:" + os.path.join(ROOT_DIR_PATH, "mlruns")
TEST_EXPERIMENT_NAME = "Pytest"
TEST_RUN_NAME = "DecisionTree"

# mlrun tags
X_COL = "X_col"
Y_COL = "y_col"
iris_X_COL = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
iris_y_COL = "target"

# mlrun artifacts
PIPELINE_HTML = "pipeline.html"
SCORES_CSV = "scores.csv"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT = "feature_importance_plot"

# mlrun metrics
ACCURACY = "accuracy"
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
CLASSIFIER_SCORING = {
    ACCURACY: ACCURACY,
    PRECISION: make_scorer(precision_score, average="binary"),
    RECALL: make_scorer(recall_score, average="binary"),
    F1: make_scorer(f1_score, average="binary"),
}
MICRO_CLASSIFIER_SCORING = {
    ACCURACY: ACCURACY,
    PRECISION: make_scorer(precision_score, average="micro"),
    RECALL: make_scorer(recall_score, average="micro"),
    F1: make_scorer(f1_score, average="micro"),
}

# cross validate constants
VECTORIZER = "vectorizer"
CLASSIFIER = "classifier"
DEFAULT_CV = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
