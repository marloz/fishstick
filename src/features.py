from typing import List

import pandas as pd
from loguru import logger

from src.utils import log_io_length


def moving_avg(df: pd.DataFrame, index: str | List[str], value_col: str, window: int) -> pd.Series:
    """Divide current value by its moving average to standardize values by
    considering relative features"""
    return df[value_col].div(
        df.groupby(index)[value_col]  # type: ignore
        .rolling(window=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


@log_io_length
def calculate_features(df: pd.DataFrame, window_lengths: List[int]) -> pd.DataFrame:
    logger.info("Calculating features")
    input_rows = len(df)
    moving_averages = {
        f"sma_{str(window)}": lambda x: moving_avg(x, "Symbol", "Close", window)
        for window in window_lengths
    }
    df = df.sort_values(by=["Symbol", "Date"]).assign(**moving_averages)
    assert input_rows == len(df), "Number of rows changed!"
    return df
