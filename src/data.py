from datetime import datetime, timedelta
from typing import Optional

import bs4 as bs
import pandas as pd
import pandera as pa
import requests
import yfinance as yf
from loguru import logger

from src.utils import log_io_length

DT_FMT = "%Y-%m-%d"


def scrape_tickers(url: str, limit: Optional[int] = None) -> list[str]:
    """Scrape stock tickers from Wikipedia that are either currently listed on S&P500"""
    logger.info("Scrapping S&P500 tickers")
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable", "id": "constituents"})
    tickers = [row.findAll("td")[0].text for row in table.findAll("tr")[1:]]  # type: ignore
    tickers = sorted([*set(ticker.strip() for ticker in tickers)])[:limit]
    logger.info(f"Collected {len(tickers)} tickers")
    return tickers


def get_daily_ticker_data(
    tickers: str | list[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
    **kwargs,
) -> pd.DataFrame:
    """Download collected ticker price data for selected dates and frequency
    from yahoo finance https://aroussi.com/post/python-yahoo-finance"""
    logger.info("Collecting ticker data from yahoo finance")
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
    logger.info("Downcasting data types to optimize memory usage")
    numeric_df = df.select_dtypes(include="float")
    df[numeric_df.columns] = numeric_df.astype("float32")  # type: ignore
    return df.assign(Symbol=lambda x: x["Symbol"].astype("category"))


@log_io_length
def validate(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Use pandera to perform data quality checks:
    - if required columns exist and are not null
    - are dates within expected range, yfinance uses end_date exlcusively,
        need to subtract one day
    - symbol is capitalized 1-5 char length string
    """
    logger.info("Validating data")
    end_date = (datetime.strptime(end_date, DT_FMT) - timedelta(days=1)).strftime(DT_FMT)
    min_dt_check = pa.Check.greater_than_or_equal_to(pd.Timestamp(start_date))
    max_dt_check = pa.Check.less_than_or_equal_to(pd.Timestamp(end_date))
    capitalized_one_to_five_chars = pa.Check.str_matches(r"^[A-Z]{1,5}$")
    schema = pa.DataFrameSchema(
        {
            "Date": pa.Column(pa.Timestamp, nullable=False, checks=[min_dt_check, max_dt_check]),
            "Symbol": pa.Column(pa.String, nullable=False, checks=[capitalized_one_to_five_chars]),
            "Close": pa.Column(pa.Float64, nullable=False, checks=[pa.Check.greater_than(0)]),
        }
    )
    return schema.validate(df)


def ticker_pipe(
    tickers: list[str], start_date: str, end_date: Optional[str] = None, interval: str = "1d"
) -> pd.DataFrame:
    """Apply get ticker data, downcast and validation in sequence for use in both feature
    and inference pipelines. In case of inference end_date will not be specified.

    If tickers list contains a single ticker, add empty string, otherwise yfinance returned df
    changes its shape."""
    if end_date is None:
        end_date = (datetime.strptime(start_date, DT_FMT) + timedelta(days=1)).strftime(DT_FMT)

    tickers = tickers + [""] if len(tickers) == 1 else tickers
    return (
        get_daily_ticker_data(tickers, start_date, end_date, interval)
        .pipe(validate, start_date=start_date, end_date=end_date)
        .pipe(downcast_dtypes)
    )
