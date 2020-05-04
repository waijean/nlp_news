import pandas as pd
import requests
from _datetime import datetime
import logging.config

from utils.constants import LOG_CONFIG_PATH
from utils.helper import get_df_info_to_logger
from web_scraping.scrape_news import export_df
from web_scraping.validator import drop_null_rows

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def get_df_from_url_manual(url: str) -> pd.DataFrame:
    r = requests.get(url)

    file = r.content.decode(encoding="utf-8")
    lines = [line.split(",") for line in file.split("\n")]

    header = lines[0]
    result = []

    for line in lines[1:]:
        d = {}
        for num, col in enumerate(line):
            d[header[num]] = col
        result.append(d)

    df = pd.DataFrame(result)
    logger.info(df.info())
    return df


def get_df_from_url(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    get_df_info_to_logger(df, logger)
    return df


if __name__ == "__main__":
    url = "https://query1.finance.yahoo.com/v7/finance/download/VEVE.L?period1=1412035200&period2=1588291200&interval=1d&events=history"
    price_df = get_df_from_url(url)
    processed_price_df = drop_null_rows(price_df)
    export_df(
        processed_price_df,
        dir="data",
        file_name=f"{datetime.now().strftime('%Y%m%d')}_price_v1.parquet",
    )
