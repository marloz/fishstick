import hydra
import numpy as np
import pandas as pd
from loguru import logger

from src.config import DatasetConfig


def create_dataset(
    df_features: pd.DataFrame,
    df_target: pd.DataFrame,
    longest_window_feature: str,
    train_cutoff: str,
) -> pd.DataFrame:
    """Join target with features, filter out rows where either target or longest
    window feature is not available and indicate train/test split based on point
    in time"""
    join_columns = ["Date", "Symbol"]
    return (
        df_features.merge(
            df_target[join_columns + ["target"]], on=join_columns, how="inner"
        )
        .loc[lambda x: ~(x["target"].isnull() | x[longest_window_feature].isnull())]
        .assign(dataset=lambda x: np.where(x["Date"] < train_cutoff, "train", "test"))
    )


@hydra.main(config_path="../config", config_name="dataset", version_base=None)
def main(config: DatasetConfig) -> None:
    logger.info(f"Starting dataset creation step, using config: \n{config}")

    logger.info("Reading inputs")
    df_features = pd.read_parquet(config.features_path)
    df_target = pd.read_parquet(config.target_path)
    msg = "Raw target and features should contain same number of rows!"
    assert len(df_features) == len(df_target), msg

    logger.info("Creating dataset")
    df_res = create_dataset(
        df_features, df_target, config.longest_window_feature, config.train_cutoff
    )
    logger.info(f"Output shape: {df_res.shape}")

    logger.info("Writing result")
    df_res.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
