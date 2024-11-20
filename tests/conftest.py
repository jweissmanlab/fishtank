from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import fishtank as ft


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


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
def dapi_img(img) -> np.array:
    yield img[2]


@pytest.fixture(scope="session")
def attrs(img_attrs) -> dict:
    yield img_attrs[1]


@pytest.fixture(scope="session")
def corr_path() -> Path:
    yield Path("tests/data/corrections")


@pytest.fixture(scope="session")
def polygons_path() -> Path:
    yield Path("tests/data/polygons/")


@pytest.fixture(scope="session")
def polygons(polygons_path) -> gpd.GeoDataFrame:
    polygons_0 = gpd.read_file(polygons_path / "polygons_0.json")
    polygons_0.set_crs(None, allow_override=True, inplace=True)
    yield polygons_0


@pytest.fixture(scope="session")
def overlapping_polygons(polygons_path) -> gpd.GeoDataFrame:
    polygons_0 = gpd.read_file(polygons_path / "polygons_0.json")
    polygons_0.set_crs(None, allow_override=True, inplace=True)
    polygons_1 = gpd.read_file(polygons_path / "polygons_1.json")
    polygons_1.set_crs(None, allow_override=True, inplace=True)
    polygons_1.geometry = polygons_1.geometry.translate(xoff=200)
    yield gpd.GeoDataFrame(pd.concat([polygons_0, polygons_1]))


@pytest.fixture(scope="session")
def masks() -> np.ndarray:
    yield np.load("tests/data/masks/masks_0.npy")


@pytest.fixture(scope="session")
def spots_path() -> Path:
    yield Path("tests/data/spots")


@pytest.fixture(scope="session")
def spots(spots_path) -> pd.DataFrame:
    yield pd.read_csv(spots_path / "spots_0.csv")


@pytest.fixture(scope="session")
def spot_channels(spots_path) -> pd.DataFrame:
    yield ft.io.read_color_usage(spots_path / "color_usage.csv")


@pytest.fixture(scope="session")
def codebook() -> pd.DataFrame:
    yield pd.read_csv("tests/data/decoding/codebook.csv", index_col=0)


@pytest.fixture(scope="session")
def weights() -> pd.DataFrame:
    yield pd.read_csv("tests/data/decoding/weights.csv", index_col=0)
