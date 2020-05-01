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
