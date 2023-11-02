import pandas as pd

from src.features import calculate_features


def test_calculate_features(test_data_dir):
    df = pd.read_parquet(test_data_dir + "/features/input.parquet")
    expected = pd.read_parquet(test_data_dir + "/features/expected.parquet")
    res = calculate_features(df, window_lengths=[10, 20])
    pd.testing.assert_frame_equal(res, expected)
