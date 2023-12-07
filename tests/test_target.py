import numpy as np
import pandas as pd

from src.pipeline.target import calculate_target


def test_calculate_target() -> None:
    """Expected:
    - data sorted by symbol, data
    - target column created with 0/1 indicating if tomorrow's price is higher
    - target value for last day in range is NaN"""
    df = pd.DataFrame(
        [
            ["A", "2020-01-01", 100],
            ["A", "2020-01-02", 200],
            ["B", "2020-01-02", 1],
            ["B", "2020-01-01", 2],
        ],
        columns=["Symbol", "Date", "Close"],
    )
    expected = pd.DataFrame(
        [
            ["A", "2020-01-01", 100, 1.0],
            ["A", "2020-01-02", 200, np.NaN],
            ["B", "2020-01-01", 2, 0.0],
            ["B", "2020-01-02", 1, np.NaN],
        ],
        columns=["Symbol", "Date", "Close", "target"],
    )
    res = calculate_target(df, 1).reset_index(drop=True)
    pd.testing.assert_frame_equal(res, expected)
