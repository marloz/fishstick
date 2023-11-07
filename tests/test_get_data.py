import pandas as pd
import pytest
from mock import Mock, patch

from src.get_data import get_daily_ticker_data, scrape_tickers


@pytest.fixture
def scrape_result(test_data_dir) -> str:
    path = test_data_dir + "/get_data/scrape_tickers_input.txt"
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def mock_response(scrape_result) -> Mock:
    mock_resp = Mock()
    mock_resp.text = scrape_result
    return mock_resp


@pytest.mark.parametrize(
    ["limit", "expected"],
    [
        # fetch current tickers
        (None, ["AOS", "MMM"]),
        # limit returned list length
        (1, ["AOS"]),
    ],
)
def test_scrape_tickers(limit, expected, mock_response) -> None:
    with patch("src.get_data.requests") as mock_requests:
        mock_requests.get = Mock(return_value=mock_response)
        assert scrape_tickers("", limit) == expected


@pytest.fixture
def yahoo_df() -> pd.DataFrame:
    columns = pd.MultiIndex(
        levels=[
            ["Adj Close", "Close", "High", "Low", "Open", "Volume"],
            ["LULU", "MMM"],
        ],
        codes=[
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
    )
    index = pd.DatetimeIndex(
        data=["2020-01-02", "2020-01-03"], dtype="datetime64[ns]", name="Date"
    )
    return pd.DataFrame(
        [
            [233, 154, 233, 180, 233, 180, 231, 177, 232, 177, 1449300, 3601700],
            [232, 153, 232, 178, 234, 178, 230, 175, 231, 177, 1315400, 2466900],
        ],
        columns=columns,
        index=index,
    )


@pytest.fixture
def get_daily_ticker_data_expected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                pd.Timestamp("2020-01-02 00:00:00"),
                "LULU",
                233,
                233,
                233,
                231,
                232,
                1449300,
            ],
            [
                pd.Timestamp("2020-01-03 00:00:00"),
                "LULU",
                232,
                232,
                234,
                230,
                231,
                1315400,
            ],
            [
                pd.Timestamp("2020-01-02 00:00:00"),
                "MMM",
                154,
                180,
                180,
                177,
                177,
                3601700,
            ],
            [
                pd.Timestamp("2020-01-03 00:00:00"),
                "MMM",
                153,
                178,
                178,
                175,
                177,
                2466900,
            ],
        ],
        columns=[
            "Date",
            "Symbol",
            "Adj Close",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ],
        index=[0, 2, 1, 3],
    )


def test_get_daily_ticker_data(yahoo_df, get_daily_ticker_data_expected) -> None:
    with patch("src.get_data.yf") as mock_yf:
        mock_yf.download = Mock(return_value=yahoo_df)
        tickers = ["LULU", "MMM"]
        start = "2020-01-01"
        end = "2020-01-04"
        res = get_daily_ticker_data(tickers, start, end)
        pd.testing.assert_frame_equal(res, get_daily_ticker_data_expected)
