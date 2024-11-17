from importlib.metadata import version

from . import cli, correct, decode, filters, io, seg

__all__ = ["io", "correct", "decode", "seg", "filters", "cli"]

__version__ = version("fishtank")
