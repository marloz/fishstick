import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.config import TargetConfig
from src.utils import parse_dict_config


def calculate_target(df: pd.DataFrame, look_ahead_days: int) -> pd.DataFrame:
    """Calcuate target - if close price for given ticker is higher or lower after specified number
    of days in the future. Using np.sign instead of bool comparison, to avoid casting of
    NaN values into False."""
    return df.sort_values(["Symbol", "Date"]).assign(
        target=lambda x: (
            np.sign(x.groupby("Symbol")["Close"].shift(-look_ahead_days) - x["Close"])
            + 1
        )
        / 2
    )


@hydra.main(config_path="../config", config_name="target", version_base=None)
def main(config_: DictConfig) -> None:
    config = parse_dict_config(TargetConfig, config_)
    logger.info(f"Starting target creation step, using config: \n{config}")

    logger.info("Reading data")
    df = pd.read_parquet(config.input_path, columns=config.columns)
    logger.info(f"Input shape: {df.shape}")
    input_rows = len(df)

    logger.info("Creating target")
    df = calculate_target(df, config.look_ahead_days)
    logger.info(f"Output shape: {df.shape}")
    assert input_rows == len(df), "Number of rows changed!"

    logger.info("Writing output")
    df.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
