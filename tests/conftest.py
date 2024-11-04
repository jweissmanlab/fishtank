from pathlib import Path

import pandas as pd
import pytest

import fishtank as ft


@pytest.fixture(scope="session")
def img_path() -> Path:
    return Path("tests/data/merfish")


@pytest.fixture(scope="session")
def channels(img_path) -> pd.DataFrame:
    return ft.io.read_color_usage(img_path / "color_usage.csv")
