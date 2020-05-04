from utils.constants import LOG_CONFIG_PATH
from utils.decorator import logging_time

import logging.config
import pandas as pd

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@logging_time(level=logging.DEBUG)
def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Number of rows before dropping null: {df.shape}")
    new_df = df.dropna(axis=0)
    logger.info(f"Number of rows after dropping null: {new_df.shape}")
    return new_df
