from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TickerScrapeConfig:
    url: str
    ticker_limit: Optional[int] = None


@dataclass
class YahooFinanceConfig:
    start_date: str
    end_date: str
    interval: str


@dataclass
class GetDataConfig:
    ticker_config: TickerScrapeConfig
    yahoo_config: YahooFinanceConfig
    output_path: str


@dataclass
class FeatureConfig:
    input_path: str
    output_path: str
    columns: List[str]
    window_lengths: List[int]


@dataclass
class TargetConfig:
    look_ahead_days: int
    columns: List[str]
    input_path: str
    output_path: str


@dataclass
class DatasetConfig:
    features_path: str
    target_path: str
    longest_window_feature: str
    train_cutoff: str
    output_path: str


@dataclass
class TrainConfig:
    input_path: str
    model_path: str
    metrics_path: str
    features: List[str]
    model: Dict[str, Any]
