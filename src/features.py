from typing import List

import hydra
import pandas as pd
from loguru import logger

from src.config import FeatureConfig


def moving_avg(
    df: pd.DataFrame, index: str | List[str], value_col: str, window: int
) -> pd.Series:
    return (
        df.groupby(index)[value_col]
        .rolling(window=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def calculate_features(df: pd.DataFrame, window_lengths: List[int]) -> pd.DataFrame:
    moving_averages = {
        f"sma_{str(window)}": lambda x: moving_avg(x, "Symbol", "Close", window)
        for window in window_lengths
    }
    return df.sort_values(by=["Symbol", "Date"]).assign(**moving_averages)


@hydra.main(config_path="../config", config_name="features", version_base=None)
def main(config: FeatureConfig) -> None:
    logger.info(f"Starting feature creation step, using config: \n{config}")

    logger.info("Reading data")
    df = pd.read_parquet(config.input_path, columns=config.columns)

    logger.info("Calculating features")
    df = calculate_features(df, window_lengths=config.window_lengths)
    logger.info(f"Output shape: {df.shape}")

    logger.info("Writing result")
    df.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
