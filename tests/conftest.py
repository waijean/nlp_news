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
