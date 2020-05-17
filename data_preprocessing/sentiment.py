from textblob import TextBlob
import pandas as pd
import logging.config

from utils.constants import LOG_CONFIG_PATH

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# TODO combine polarity and subjectivity so don't have to convert to text blob twivr
def calculate_polarity(text: str) -> float:
    if text is None:
        return 0
    else:
        return TextBlob(text).sentiment.polarity


def calculate_subjectivity(text: str) -> float:
    if text is None:
        return 0
    else:
        return TextBlob(text).sentiment.subjectivity


def add_col_polarity(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Pass in a dataframe which contains the column that we wish to calculate polarity from
    Return a dataframe with the added polarity column
    """
    logger.info(f"Calculating polarity for {col}")
    kwargs = {f"{col}_polarity": df[col].apply(calculate_polarity)}
    new_df = df.assign(**kwargs)
    return new_df


def add_col_subjectivity(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Pass in a dataframe which contains the column that we wish to calculate subjectivity from
    Return a dataframe with the added subjectivity column
    """
    logger.info(f"Calculating subjectivity for {col}")
    kwargs = {f"{col}_subjectivity": df[col].apply(calculate_subjectivity)}
    new_df = df.assign(**kwargs)
    return new_df
