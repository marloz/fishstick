from typing import Optional

import bs4 as bs
import hydra
import pandas as pd
import requests
import yfinance as yf
from hydra.core.config_store import ConfigStore
from loguru import logger

from src.config import GetDataConfig

cs = ConfigStore.instance()
cs.store(name="get_data_confg", node=GetDataConfig)


def scrape_tickers(
    url: str, include_historical: bool = False, limit: Optional[int] = None
) -> list[str]:
    """Scrape stock tickers from Wikipedia that are either currently on
    S&P or historically were included"""
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, "lxml")

    current_table = soup.find(
        "table", {"class": "wikitable sortable", "id": "constituents"}
    )

    tickers = [
        row.findAll("td")[0].text for row in current_table.findAll("tr")[1:]  # type: ignore
    ]

    if include_historical:
        historical_table = soup.find(
            "table", {"class": "wikitable sortable", "id": "changes"}
        )
        historical_tickers = []
        for row in historical_table.findAll("tr")[2:]:  # type: ignore
            data = row.findAll("td")
            added_ticker, removed_ticker = data[1].text, data[3].text
            historical_tickers.append(added_ticker)
            historical_tickers.append(removed_ticker)
        tickers += historical_tickers

    return [*set(ticker.strip() for ticker in tickers)][:limit]


def get_daily_ticker_data(
    tickers: list[str], start_date: str, end_date: str, interval: str = "1d", **kwargs
) -> pd.DataFrame:
    """Download collected ticker price data for selected dates and frequency
    from yahoo finance https://aroussi.com/post/python-yahoo-finance"""
    return (
        yf.download(
            tickers, start=start_date, end=end_date, interval=interval, **kwargs
        )
        .stack()
        .reset_index()
        .rename(index=str, columns={"level_1": "Symbol"})
        .sort_values(["Symbol", "Date"])
    )


@hydra.main(config_path="../config", config_name="get_data", version_base=None)
def main(config: GetDataConfig) -> None:
    """Scrapes current and/or historical stock tickers from wiki,
    then gets their price data from yahoo finance and stores in a
    parquet file."""
    logger.info(f"Starting get data step, using config: \n{config}")

    logger.info("Scrapping S&P500 tickers")
    tickers = scrape_tickers(
        config.ticker_config.url,
        config.ticker_config.include_historical,
        config.ticker_config.ticker_limit,
    )
    logger.info(f"Collected {len(tickers)} tickers")

    logger.info("Collecting ticker data from yahoo finance")
    df = get_daily_ticker_data(
        tickers,
        config.yahoo_config.start_date,
        config.yahoo_config.end_date,
        config.yahoo_config.interval,
    )
    logger.info(f"Processed data shape: {df.shape}")

    logger.info("Writing results")
    df.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
