from dataclasses import dataclass
from typing import Optional


@dataclass
class TickerScrapeConfig:
    url: str
    include_historical: bool
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
