from dataclasses import dataclass
from typing import Any

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.model import Model, ModelTrainer, make_pipeline, save_model
from src.utils import parse_dict_config


@dataclass
class TrainConfig:
    input_path: str
    model_path: str
    metrics_path: str
    features: list[str]
    steps: dict[str, Any]


@hydra.main(config_path="../../config", config_name="train", version_base=None)
def main(config_: DictConfig) -> None:
    config: TrainConfig = parse_dict_config(TrainConfig, config_)
    logger.info(f"Starting training step, using config: \n{config}")

    logger.info("Loading model")
    model: Model = make_pipeline(config.steps)

    logger.info("Loading data")
    df = pd.read_parquet(config.input_path)

    trainer = ModelTrainer(config.features)
    trainer.run(model, df)

    logger.info("Saving model and metrics")
    trainer.metrics.save(config.metrics_path)
    save_model(trainer.model, config.model_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
