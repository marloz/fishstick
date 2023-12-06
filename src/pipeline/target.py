from dataclasses import dataclass
from typing import List

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.utils import log_io_length, parse_dict_config


@dataclass
class TargetConfig:
    look_ahead_days: int
    columns: List[str]
    input_path: str
    output_path: str


@log_io_length
def calculate_target(df: pd.DataFrame, look_ahead_days: int) -> pd.DataFrame:
    """Calcuate target - if close price for given ticker is higher or lower after specified number
    of days in the future. Using np.sign instead of bool comparison, to avoid casting of
    NaN values into False."""
    logger.info("Creating target")
    input_rows = len(df)

    def _target(x):
        close_shift = x.groupby("Symbol")["Close"].shift(-look_ahead_days)
        change_sign = np.sign(close_shift - x["Close"])
        return (change_sign + 1) / 2

    df = df.sort_values(["Symbol", "Date"]).assign(target=_target)

    assert input_rows == len(df), "Number of rows changed!"
    return df


@hydra.main(config_path="../../config", config_name="target", version_base=None)
def main(config_: DictConfig) -> None:
    config: TargetConfig = parse_dict_config(TargetConfig, config_)
    logger.info(f"Starting target creation step, using config: \n{config}")

    logger.info("Reading data")
    df = pd.read_parquet(config.input_path, columns=config.columns)

    df = calculate_target(df, config.look_ahead_days)

    logger.info("Writing output")
    df.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
