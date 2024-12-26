import ast
import math
from pathlib import Path

import numpy as np


def parse_dict(arg: str) -> dict:
    """Parse dictionary input in the form of key1=val1,key2=val2"""
    if arg is None:
        return {}
    return {key: ast.literal_eval(value) for key, value in (item.split("=") for item in arg.split(","))}


def parse_list(arg: str) -> list:
    """Parse list input separated by commas"""
    if arg is None:
        return None
    return arg.split(",")


def parse_index(arg: str) -> list:
    """Parse index input in the form of start:end:step"""
    if arg is None:
        return None
    if "," in arg:
        return parse_list(arg)
    if ":" in arg:
        return list(range(*map(int, arg.split(":"))))
    return [int(arg)]


def parse_path(arg: str) -> Path:
    """Parse path input"""
    if arg is None:
        return None
    return Path(arg)


def parse_bool(arg: str) -> bool:
    """Parse boolean input"""
    if arg is None:
        return False
    return arg.lower() in ["true", "1", "t", "y", "yes"]


def parse_rotation(arg: str) -> float:
    """Parse rotation input"""
    if arg is None:
        return None
    elif arg.endswith(".npy"):
        matrix = np.load(Path(arg))
        radians = math.atan2(matrix[1, 0], matrix[0, 0])
        return math.degrees(radians)
    return float(arg)
