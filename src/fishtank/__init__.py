from importlib.metadata import version

from . import cli, correct, decode, filters, io, pl, seg

__all__ = ["io", "pl", "correct", "decode", "seg", "filters", "cli"]

__version__ = version("fishtank")
