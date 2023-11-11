from datetime import datetime, timedelta
from typing import Optional

import bs4 as bs
import hydra
import pandas as pd
import pandera as pa
import requests
import yfinance as yf
from loguru import logger
from omegaconf import DictConfig

from src.config import GetDataConfig
from src.utils import parse_dict_config


def scrape_tickers(url: str, limit: Optional[int] = None) -> list[str]:
    """Scrape stock tickers from Wikipedia that are either currently listed on S&P500"""
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable", "id": "constituents"})
    tickers = [row.findAll("td")[0].text for row in table.findAll("tr")[1:]]  # type: ignore
    return sorted([*set(ticker.strip() for ticker in tickers)])[:limit]


def get_daily_ticker_data(
    tickers: list[str], start_date: str, end_date: str, interval: str = "1d", **kwargs
) -> pd.DataFrame:
    """Download collected ticker price data for selected dates and frequency
    from yahoo finance https://aroussi.com/post/python-yahoo-finance"""
    return (
        yf.download(tickers, start=start_date, end=end_date, interval=interval, **kwargs)
        .stack()
        .reset_index()
        .rename(columns={"level_1": "Symbol"})
        .sort_values(["Symbol", "Date"])
    )


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert floats from 64 to 32 bytes and change Symbol from string to category,
    to save memory usage."""
    numeric_df = df.select_dtypes(include="float")
    df[numeric_df.columns] = numeric_df.astype("float32")
    return df.assign(Symbol=lambda x: x["Symbol"].astype("category"))


def validate(
    df: pd.DataFrame, min_date: str, max_date: str, dt_fmt: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """Use pandera to perform data quality checks:
    - if required columns exist and are not null
    - are dates within expected range, yfinance uses end_date exlcusively,
        need to subtract one day
    - symbol is capitalized 1-5 char length string
    """
    max_date = (datetime.strptime(max_date, dt_fmt) - timedelta(days=1)).strftime(dt_fmt)
    min_dt_check = pa.Check.greater_than_or_equal_to(pd.Timestamp(min_date))
    max_dt_check = pa.Check.less_than_or_equal_to(pd.Timestamp(max_date))
    capitalized_one_to_five_chars = pa.Check.str_matches(r"^[A-Z]{1,5}$")
    schema = pa.DataFrameSchema(
        {
            "Date": pa.Column(pa.Timestamp, nullable=False, checks=[min_dt_check, max_dt_check]),
            "Symbol": pa.Column(pa.String, nullable=False, checks=[capitalized_one_to_five_chars]),
            "Close": pa.Column(pa.Float32, nullable=False, checks=[pa.Check.greater_than(0)]),
        }
    )
    return schema.validate(df)


@hydra.main(config_path="../config", config_name="get_data", version_base=None)
def main(config_: DictConfig) -> None:
    """Scrapes current stock tickers from wiki,
    then gets their price data from yahoo finance and stores in a
    parquet file."""
    config: GetDataConfig = parse_dict_config(GetDataConfig, config_)
    logger.info(f"Starting get data step, using config: \n{config}")

    logger.info("Scrapping S&P500 tickers")
    tickers = scrape_tickers(
        config.ticker_config.url,
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

    logger.info("Optimizing data types")
    df = downcast_dtypes(df)

    logger.info("Validating data")
    df = validate(df, config.yahoo_config.start_date, config.yahoo_config.end_date)

    logger.info("Writing results")
    df.to_parquet(config.output_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()  # pylint: disable=E1120:no-value-for-parameter
