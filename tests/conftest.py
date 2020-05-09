import pytest
import pandas as pd

from utils.constants import COL_HEADLINE, COL_ARTICLE, COL_CATEGORY


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
