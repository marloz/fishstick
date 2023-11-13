import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.config import FeatureConfig
from src.features import calculate_features
from src.utils import parse_dict_config


@hydra.main(config_path="../../config", config_name="features", version_base=None)
def main(config_: DictConfig) -> None:
    config: FeatureConfig = parse_dict_config(FeatureConfig, config_)
    logger.info(f"Starting feature creation step, using config: \n{config}")

    logger.info("Reading data")
    df = pd.read_parquet(config.input_path, columns=config.columns)

    df = calculate_features(df, window_lengths=config.window_lengths)

    logger.info("Writing result")
    df.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
