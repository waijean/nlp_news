from functools import lru_cache

import mlflow
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, make_scorer, recall_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_modeling.cross_validate_runner import CrossValidatePipeline
from utils.constants import (
    COL_HEADLINE,
    COL_ARTICLE,
    COL_CATEGORY,
    COL_DATE,
    COL_CLOSE,
    COL_VOLUME,
    COL_OPEN,
    COL_PRICE_DATE,
    CLASSIFIER,
    iris_X_COL,
    iris_y_COL,
    FULL_PARAM,
    SCALER_PARAM,
    CLF_PARAM,
    CLASSIFIER_SCORING,
    MICRO_CLASSIFIER_SCORING,
    TEST_EXPERIMENT_NAME,
)


@pytest.fixture
def expected_news_df():
    return pd.DataFrame(
        {
            COL_HEADLINE: ["headline_1", "headline_2"],
            COL_ARTICLE: ["article_1", "article_2"],
            COL_CATEGORY: ["category_1", "category_2"],
        }
    )


@pytest.fixture(scope="session")
def working_dir(tmp_path_factory):
    # Default base temporary directory is C:\Users\kwj_9\AppData\Local\Temp\pytest-of-kwj_9
    tmp_path = tmp_path_factory.mktemp("working_dir")
    return tmp_path


@pytest.fixture(scope="session")
def iris_df():
    iris = load_iris()
    return pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )


@pytest.fixture(scope="session")
def expected_X_train(iris_df):
    # TODO return tuple for fixture
    X = iris_df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ]
    y = iris_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train


@pytest.fixture(scope="session")
def expected_y_train(iris_df):
    X = iris_df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ]
    y = iris_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return y_train


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, iris_df):
    # Default base temporary directory is C:\Users\kwj_9\AppData\Local\Temp\pytest-of-kwj_9
    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "iris.parquet"
    iris_df.to_parquet(p)
    return tmp_path


@pytest.fixture(scope="session")
def mlrun_dir(tmp_path_factory):
    # Default base temporary directory is C:\Users\kwj_9\AppData\Local\Temp\pytest-of-kwj_9
    tmp_path = tmp_path_factory.mktemp("mlruns")
    return tmp_path


@pytest.fixture(scope="session")
def expected_price_df():
    return pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Close": [50.0, 50.1, 50.2],
        }
    )


@pytest.fixture(scope="session")
def input_price_df():
    return pd.DataFrame(
        {
            COL_PRICE_DATE: ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            COL_OPEN: [50.0, 50.3, 0.0, 50.5],
            COL_CLOSE: [50.1, 50.2, 50.4, 50.5],
            COL_VOLUME: [1000, 1000, 1000, 0],
        }
    )


@pytest.fixture(scope="session")
def input_text_df(input_text):
    return pd.DataFrame({COL_ARTICLE: [input_text]})


@pytest.fixture(scope="session")
def input_text():
    return """
    \tMr Shapps said that any announcement of a change in strategy — including the imposition of quarantine arrangements 
    for people arriving in Britain — would be made by Boris Johnson in a televised address on Sunday night.\n
    \tBut he said: “We can't have a situation where everyone else is being asked to stay at home but others can come 
    into the country.” He also noted that the number of people coming into the UK was currently “very, very small".
    """


@pytest.fixture(scope="session")
def input_text_no_punc():
    return """
    \tMr Shapps said that any announcement of a change in strategy  including the imposition of quarantine arrangements 
    for people arriving in Britain  would be made by Boris Johnson in a televised address on Sunday night\n
    \tBut he said We cant have a situation where everyone else is being asked to stay at home but others can come 
    into the country He also noted that the number of people coming into the UK was currently very very small
    """


@pytest.fixture(scope="session")
def tokenized_list():
    return [
        "mr",
        "shapps",
        "said",
        "that",
        "any",
        "announcement",
        "of",
        "a",
        "change",
        "in",
        "strategy",
        "—",
        "including",
        "the",
        "imposition",
        "of",
        "quarantine",
        "arrangements",
        "for",
        "people",
        "arriving",
        "in",
        "britain",
        "—",
        "would",
        "be",
        "made",
        "by",
        "boris",
        "johnson",
        "in",
        "a",
        "televised",
        "address",
        "on",
        "sunday",
        "night",
        ".",
        "but",
        "he",
        "said",
        ":",
        "“",
        "we",
        "can",
        "'",
        "t",
        "have",
        "a",
        "situation",
        "where",
        "everyone",
        "else",
        "is",
        "being",
        "asked",
        "to",
        "stay",
        "at",
        "home",
        "but",
        "others",
        "can",
        "come",
        "into",
        "the",
        "country",
        ".”",
        "he",
        "also",
        "noted",
        "that",
        "the",
        "number",
        "of",
        "people",
        "coming",
        "into",
        "the",
        "uk",
        "was",
        "currently",
        "“",
        "very",
        ",",
        "very",
        "small",
        '".',
    ]


@pytest.fixture(scope="session")
def tokenized_list_no_punc():
    return [
        "mr",
        "shapps",
        "said",
        "that",
        "any",
        "announcement",
        "of",
        "a",
        "change",
        "in",
        "strategy",
        "including",
        "the",
        "imposition",
        "of",
        "quarantine",
        "arrangements",
        "for",
        "people",
        "arriving",
        "in",
        "britain",
        "would",
        "be",
        "made",
        "by",
        "boris",
        "johnson",
        "in",
        "a",
        "televised",
        "address",
        "on",
        "sunday",
        "night",
        "but",
        "he",
        "said",
        "we",
        "can",
        "t",
        "have",
        "a",
        "situation",
        "where",
        "everyone",
        "else",
        "is",
        "being",
        "asked",
        "to",
        "stay",
        "at",
        "home",
        "but",
        "others",
        "can",
        "come",
        "into",
        "the",
        "country",
        "he",
        "also",
        "noted",
        "that",
        "the",
        "number",
        "of",
        "people",
        "coming",
        "into",
        "the",
        "uk",
        "was",
        "currently",
        "very",
        "very",
        "small",
    ]


@pytest.fixture(scope="session")
def tokenized_list_no_stopwords():
    return [
        "mr",
        "shapps",
        "said",
        "announcement",
        "change",
        "strategy",
        "including",
        "imposition",
        "quarantine",
        "arrangements",
        "people",
        "arriving",
        "britain",
        "would",
        "made",
        "boris",
        "johnson",
        "televised",
        "address",
        "sunday",
        "night",
        "said",
        "situation",
        "everyone",
        "else",
        "asked",
        "stay",
        "home",
        "others",
        "come",
        "country",
        "also",
        "noted",
        "number",
        "people",
        "coming",
        "uk",
        "currently",
        "small",
    ]


@pytest.fixture(scope="session")
def tokenized_list_stemming():
    return [
        "mr",
        "shapp",
        "said",
        "announc",
        "chang",
        "strategi",
        "includ",
        "imposit",
        "quarantin",
        "arrang",
        "peopl",
        "arriv",
        "britain",
        "would",
        "made",
        "bori",
        "johnson",
        "televis",
        "address",
        "sunday",
        "night",
        "said",
        "situat",
        "everyon",
        "els",
        "ask",
        "stay",
        "home",
        "other",
        "come",
        "countri",
        "also",
        "note",
        "number",
        "peopl",
        "come",
        "uk",
        "current",
        "small",
    ]


@pytest.fixture(scope="session")
def tokenized_list_lemmatize():
    return [
        "mr",
        "shapps",
        "said",
        "announcement",
        "change",
        "strategy",
        "including",
        "imposition",
        "quarantine",
        "arrangement",
        "people",
        "arriving",
        "britain",
        "would",
        "made",
        "boris",
        "johnson",
        "televised",
        "address",
        "sunday",
        "night",
        "said",
        "situation",
        "everyone",
        "else",
        "asked",
        "stay",
        "home",
        "others",
        "come",
        "country",
        "also",
        "noted",
        "number",
        "people",
        "coming",
        "uk",
        "currently",
        "small",
    ]


@pytest.fixture(scope="session")
def expected_spacy_tokenized_col():
    return [
        "mr shapps say announcement change strategy include imposition quarantine arrangement people arrive britain boris johnson televise address sunday night say situation ask stay home come country note numb people come uk currently small"
    ]


@pytest.fixture(scope="session")
def article_col():
    return pd.Series(
        [
            "this is the first document",
            "this document is the second document",
            "and this is the third one",
            "is this the first document",
        ]
    )


@pytest.fixture(scope="session")
def X_count_df():
    return pd.DataFrame(
        np.array(
            [[1, 1, 0, 0, 0], [2, 0, 0, 1, 0], [0, 0, 1, 0, 1], [1, 1, 0, 0, 0]],
            dtype=np.int64,
        ),
        columns=["document", "first", "one", "second", "third"],
    )


@pytest.fixture(scope="session")
def X_tfidf_df():
    return pd.DataFrame(
        np.array(
            [
                [0.6292275146695526, 0.7772211620785797, 0, 0, 0],
                [0.78722297610404, 0.0, 0, 0.6166684570284895, 0.0],
                [0.0, 0.0, 0.7071067811865476, 0.0, 0.7071067811865476],
                [0.6292275146695526, 0.7772211620785797, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )


@pytest.fixture(scope="session")
def pipeline():
    scaler = StandardScaler(**SCALER_PARAM)
    clf = DecisionTreeClassifier(**CLF_PARAM)
    return Pipeline([("scaler", scaler), (CLASSIFIER, clf)])


@pytest.fixture(scope="session")
def cross_validate_pipeline(data_dir, mlrun_dir, pipeline):
    test_mlrun_dir = "file:" + str(mlrun_dir)
    read_path = data_dir / "iris.parquet"

    return CrossValidatePipeline(
        experiment_name=TEST_EXPERIMENT_NAME,
        read_path=read_path,
        X_col=iris_X_COL,
        y_col=iris_y_COL,
        params=FULL_PARAM,
        pipeline=pipeline,
        scoring=MICRO_CLASSIFIER_SCORING,
        tracking_uri=test_mlrun_dir,
        artifact_location=test_mlrun_dir,
    )


@pytest.fixture(scope="session")
def cv_result(pipeline, expected_X_train, expected_y_train):
    cross_validation = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    return cross_validate(
        pipeline,
        expected_X_train,
        expected_y_train,
        scoring=MICRO_CLASSIFIER_SCORING,
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True,
    )


@pytest.fixture(scope="session")
def setup_mlflow_experiment_id(cross_validate_pipeline):
    return cross_validate_pipeline.setup_mlflow()


@pytest.fixture(scope="session")
def setup_mlflow_run(setup_mlflow_experiment_id):
    with mlflow.start_run(experiment_id=setup_mlflow_experiment_id) as active_run:
        yield active_run
