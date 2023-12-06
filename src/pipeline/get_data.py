from dataclasses import dataclass
from typing import Optional

import hydra
from loguru import logger
from omegaconf import DictConfig

from src.data import scrape_tickers, ticker_pipe
from src.utils import parse_dict_config


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


@hydra.main(config_path="../../config", config_name="get_data", version_base=None)
def main(config_: DictConfig) -> None:
    """Scrapes current stock tickers from wiki,
    then gets their price data from yahoo finance and stores in a
    parquet file."""
    config: GetDataConfig = parse_dict_config(GetDataConfig, config_)
    logger.info(f"Starting get data step, using config: \n{config}")
    tickers = scrape_tickers(
        config.ticker_config.url,
        config.ticker_config.ticker_limit,
    )
    df = ticker_pipe(
        tickers,
        config.yahoo_config.start_date,
        config.yahoo_config.end_date,
        config.yahoo_config.interval,
    )
    logger.info("Writing results")
    df.to_parquet(config.output_path, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
