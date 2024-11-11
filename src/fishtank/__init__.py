from importlib.metadata import version

from . import cli, correct, filters, io, seg

__all__ = ["io", "correct", "seg", "filters", "cli"]

__version__ = version("fishtank")
