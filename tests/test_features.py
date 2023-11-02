import numpy as np
import pandas as pd

from src.features import calculate_features


def test_calculate_features():
    """Expected:
    - df is sorted by symbol and date
    - moving average column added with specified window length"""
    df = pd.DataFrame(
        [
            ["B", "2020-01-02", 1],
            ["B", "2020-01-01", 1],
            ["A", "2020-01-01", 100],
            ["A", "2020-01-02", 200],
        ],
        columns=["Symbol", "Date", "Close"],
    )
    expected = pd.DataFrame(
        [
            ["A", "2020-01-01", 100, np.NaN],
            ["A", "2020-01-02", 200, 150],
            ["B", "2020-01-01", 1, np.NaN],
            ["B", "2020-01-02", 1, 1],
        ],
        columns=["Symbol", "Date", "Close", "sma_2"],
    )
    res = calculate_features(df, window_lengths=[2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(res, expected)
