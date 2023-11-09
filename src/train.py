import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import dill
import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import auc, roc_curve

from src.config import TrainConfig
from src.utils import parse_dict_config


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class Metrics:
    train_score: float
    test_score: float

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f)


class Dataset(StrEnum):
    TEST = "test"
    TRAIN = "train"


def get_dataset_xy(
    df: pd.DataFrame, dataset: Dataset, features: list[str], target_col: str = "target"
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.loc[lambda x: x["dataset"] == dataset]
    return df[features], df[target_col]


def score_model(X: pd.DataFrame, y: pd.Series, model: Model) -> float:
    preds = model.predict(X)
    fpr, tpr, _ = roc_curve(y, preds)
    return float(auc(fpr, tpr))


def save_model(model: Model, path: str) -> None:
    with open(path, "wb") as f:
        dill.dump(model, f)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(config_: DictConfig) -> None:
    config: TrainConfig = parse_dict_config(TrainConfig, config_)
    logger.info(f"Starting training step, using config: \n{config}")

    logger.info("Loading model")
    model: Model = instantiate(config.model)

    logger.info("Loading data")
    df = pd.read_parquet(config.input_path)

    logger.info("Splitting into train/test")
    x_train, y_train = get_dataset_xy(df, Dataset.TRAIN, features=config.features)
    x_test, y_test = get_dataset_xy(df, Dataset.TEST, features=config.features)

    logger.info("Training model")
    model.fit(x_train, y_train)

    metrics = Metrics(
        train_score=score_model(x_train, y_train, model),
        test_score=score_model(x_test, y_test, model),
    )
    logger.info(f"Model metrics: {metrics}")

    logger.info("Saving model and metrics")
    metrics.save(config.metrics_path)
    save_model(model, config.model_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
