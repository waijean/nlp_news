import pytest
import pandas as pd
import numpy as np

from utils.constants import (
    COL_HEADLINE,
    COL_ARTICLE,
    COL_CATEGORY,
    COL_DATE,
    COL_CLOSE,
    COL_VOLUME,
    COL_OPEN,
    COL_PRICE_DATE,
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
def working_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("working_dir")
    return fn


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
