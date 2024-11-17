import ast
from pathlib import Path


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
