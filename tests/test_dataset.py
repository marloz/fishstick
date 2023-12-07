import numpy as np
import pandas as pd

from src.pipeline.dataset import create_dataset


def test_dataset() -> None:
    df_feat = pd.DataFrame(
        [
            # Date outside join range
            ["2019-01-01", "A", 100],
            # Train row
            ["2020-01-01", "A", 100],
            # Test row
            ["2020-02-01", "A", 100],
            # Feature is NaN
            ["2020-03-01", "A", 100],
        ],
        columns=["Date", "Symbol", "some_feat"],
    )

    df_target = pd.DataFrame(
        [
            # Train row
            ["2020-01-01", "A", 1.0],
            # Test row
            ["2020-02-01", "A", 0.0],
            # Target is NaN
            ["2020-03-01", "A", np.nan],
            # Date outside join range
            ["2021-01-01", "A", 0.0],
        ],
        columns=["Date", "Symbol", "target"],
    )

    expected = pd.DataFrame(
        [["2020-01-01", "A", 100, 1.0, "train"], ["2020-02-01", "A", 100, 0.0, "test"]],
        columns=["Date", "Symbol", "some_feat", "target", "dataset"],
    ).assign(target=lambda x: x["target"].astype("int32"))

    res = create_dataset(
        df_feat,
        df_target,
        longest_window_feature="some_feat",
        train_cutoff="2020-02-01",
    )

    pd.testing.assert_frame_equal(res, expected)
