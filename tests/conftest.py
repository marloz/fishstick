import os

import pytest


@pytest.fixture
def test_data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "data")
