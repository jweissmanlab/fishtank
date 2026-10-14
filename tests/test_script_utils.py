import argparse

import pytest

import fishtank as ft
from fishtank.scripts._utils import parse_bool


@pytest.mark.parametrize("arg", ["True", "true", "1", "yes", "y", "t"])
def test_parse_bool_true(arg):
    assert parse_bool(arg) is True


@pytest.mark.parametrize("arg", ["False", "false", "0", "no", "n", "f"])
def test_parse_bool_false(arg):
    assert parse_bool(arg) is False


def test_parse_bool_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("Flase")


@pytest.mark.parametrize("value", [True, False])
def test_cellpose_bool_flags(value):
    parser = ft.scripts.cellpose_script.get_parser()
    args = ["--input", "in", "--fov", "1", "--channels", "DAPI"]
    args = parser.parse_args(args + ["--gpu", str(value), "--do_3D", str(value), "--clear_border", str(value)])
    assert (args.gpu, args.do_3D, args.clear_border) == (value, value, value)


def test_bool_flags_default_false():
    args = ft.scripts.cellpose_script.get_parser().parse_args(["--input", "in", "--fov", "1", "--channels", "DAPI"])
    assert (args.gpu, args.do_3D, args.clear_border) == (False, False, False)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
