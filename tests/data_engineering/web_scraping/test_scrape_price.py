from data_engineering.web_scraping.scrape_price import (
    get_df_from_url_manual,
    get_df_from_url,
)
from data_engineering.web_scraping.web_url import price_url


def test_get_df_from_url_manual():
    df = get_df_from_url_manual(price_url)
    assert not df.empty


def test_get_df_from_url():
    df = get_df_from_url(price_url)
    assert not df.empty
