import logging.config
import os
from datetime import datetime
import pandas as pd

from utils.constants import (
    LOG_CONFIG_PATH,
    DATA_PIPELINE_PATH,
    COL_OPEN,
    COL_PRICE_DATE,
    COL_PREVIOUS_CLOSE,
    COL_CLOSE,
    COL_ABSOLUTE_RETURN,
    COL_VOLUME,
    COL_SIGN,
)

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def add_col_previous_close(df: pd.DataFrame) -> pd.DataFrame:
    kwargs = {
        COL_PREVIOUS_CLOSE: df.sort_values(by=[COL_PRICE_DATE])[COL_CLOSE].shift(
            periods=1, fill_value=0
        )
    }
    new_df = df.assign(**kwargs)
    return new_df


def impute_missing_open_with_previous_close(df: pd.DataFrame) -> pd.DataFrame:
    tmp_df = add_col_previous_close(df)
    mask = df[COL_OPEN] == 0
    logger.info(f"Number of rows with missing open: {len(df[mask])}")
    kwargs = {COL_OPEN: df[COL_OPEN].where(~mask, tmp_df[COL_PREVIOUS_CLOSE])}
    new_df = df.assign(**kwargs)
    return new_df


def remove_zero_volume_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[COL_VOLUME] == 0
    logger.info(f"Number of rows with zero volume: {len(df[mask])}")
    new_df = df[~mask]
    return new_df


# TODO create decorator for adding column
def add_col_absolute_return(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Adding column {COL_ABSOLUTE_RETURN}")
    kwargs = {COL_ABSOLUTE_RETURN: df[COL_CLOSE] - df[COL_OPEN]}
    new_df = df.assign(**kwargs)
    return new_df


def remove_zero_return_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[COL_ABSOLUTE_RETURN] == 0
    logger.info(f"Number of rows with zero return: {len(df[mask])}")
    new_df = df[~mask]
    return new_df


def add_col_sign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Different methods to create new column based on existing column values

    1. Dataframe loc
    2. Series map
    3. Series apply
    4. np.where
    """
    logger.info(f"Adding column {COL_SIGN}")
    new_df = df.copy()
    new_df.loc[new_df[COL_ABSOLUTE_RETURN] > 0, COL_SIGN] = 1
    new_df.loc[new_df[COL_ABSOLUTE_RETURN] < 0, COL_SIGN] = -1
    logger.info(f"Distribution of sign:\n{new_df[COL_SIGN].value_counts(dropna=False)}")
    return new_df


def select_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select columns to output and set date as index
    """
    cols_to_select = [COL_PRICE_DATE, COL_ABSOLUTE_RETURN, COL_SIGN]
    logger.info(f"Selecting columns {cols_to_select}")
    new_df = df[cols_to_select].set_index(COL_PRICE_DATE)
    return new_df


if __name__ == "__main__":
    logger.info(f"Reading from {DATA_PIPELINE_PATH}")
    df = pd.read_csv(
        os.path.join(DATA_PIPELINE_PATH, "VEVE_HistoricPrices_20141001-20200505.csv"),
        usecols=[COL_PRICE_DATE, COL_OPEN, COL_CLOSE, COL_VOLUME],
        thousands=",",
        parse_dates=[COL_PRICE_DATE],
        date_parser=lambda x: datetime.strptime(x, "%d/%m/%Y"),
    )
    processed_df = (
        df.pipe(impute_missing_open_with_previous_close)
        .pipe(remove_zero_volume_rows)
        .pipe(add_col_absolute_return)
        .pipe(remove_zero_return_rows)
        .pipe(add_col_sign)
        .pipe(select_cols)
    )
    logger.info(f"Writing to {DATA_PIPELINE_PATH}")
    processed_df.to_parquet(os.path.join(DATA_PIPELINE_PATH, "processed_price.parquet"))
