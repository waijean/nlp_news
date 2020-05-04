import pandas as pd
import io


def get_df_info_to_logger(df: pd.DataFrame, logger):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    logger.info(f"Dataframe info: \n {s}")
