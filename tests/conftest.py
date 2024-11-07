from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import fishtank as ft


@pytest.fixture(scope="session")
def img_path() -> Path:
    yield Path("tests/data/merfish")


@pytest.fixture(scope="session")
def channels(img_path) -> pd.DataFrame:
    yield ft.io.read_color_usage(img_path / "color_usage.csv").query("color.isin([748,477,405])")


@pytest.fixture(scope="session")
def img_attrs(img_path, channels) -> np.ndarray:
    yield ft.io.read_fov(img_path, channels=channels, fov=1)


@pytest.fixture(scope="session")
def img(img_attrs) -> np.array:
    yield img_attrs[0]


@pytest.fixture(scope="session")
def attrs(img_attrs) -> dict:
    yield img_attrs[1]


@pytest.fixture(scope="session")
def corr_path() -> Path:
    yield Path("tests/data/corrections")


@pytest.fixture(scope="session")
def polygons() -> gpd.GeoDataFrame:
    yield gpd.read_file("tests/data/polygons/polygons_0.geojson")


@pytest.fixture(scope="session")
def masks() -> np.ndarray:
    yield np.load("tests/data/masks/masks_0.npy")
