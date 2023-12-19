import json
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Optional, Protocol

import dill
import hydra
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import Pipeline


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
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


def save_model(model: Model, path: str) -> None:
    with open(path, "wb") as f:
        dill.dump(model, f)


def load_model(path: str) -> Model:
    with open(path, "rb") as f:
        return dill.load(f)


class ModelTrainer:
    model: Model
    metrics: Metrics

    def __init__(self, features: List[str], target_col: str = "target") -> None:
        self.features = features
        self.target_col = target_col

    def run(self, model: Model, df: pd.DataFrame) -> None:
        logger.info("Training model")
        x_train, y_train = self.get_dataset_xy(df, Dataset.TRAIN)
        x_test, y_test = self.get_dataset_xy(df, Dataset.TEST)

        logger.info("Training model")
        self.model = model.fit(x_train, y_train)

        self.metrics = Metrics(
            train_score=self.score_model(x_train, y_train),
            test_score=self.score_model(x_test, y_test),
        )
        logger.info(f"Model metrics: {self.metrics}")

    def get_dataset_xy(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
    ) -> tuple[pd.DataFrame, pd.Series]:
        logger.info("Splitting into train/test")
        df = df.loc[lambda x: x["dataset"] == dataset]  # type: ignore
        return df[self.features], df[self.target_col]

    def score_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        preds = self.model.predict(X)
        fpr, tpr, _ = roc_curve(y, preds)
        return float(auc(fpr, tpr))


def make_pipeline(steps_config: dict) -> Model:
    """Creates a pipeline with all the preprocessing steps specified in `steps_config`, ordered in a sequential manner
    https://medium.com/beyondminds/creating-configurable-data-pre-processing-pipelines-by-combining-hydra-and-sklearn-812065c9ab64
    """
    return Pipeline(
        [
            (step_name, hydra.utils.instantiate(step_params, _convert_="partial"))
            for step_name, step_params in steps_config.items()
        ]
    )  # type: ignore


class ColumnSelector(BaseEstimator, TransformerMixin):
    "Using this instead of ColumnTransformer, to simplify tuple handling in config"

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnSelector":
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return X[self.columns]
