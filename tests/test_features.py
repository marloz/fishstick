import pandas as pd

from src.features import calculate_features


def test_calculate_features(test_data_dir):
    """Expected:
    - Dataframe sorted by symbol, date
    - SMA features added for specified lengths"""
    df = pd.read_csv(test_data_dir + "/features/input.csv")
    expected = pd.read_csv(test_data_dir + "/features/expected.csv")
    res = calculate_features(df, window_lengths=[10, 20]).reset_index(drop=True)
    pd.testing.assert_frame_equal(res, expected)
