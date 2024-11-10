from importlib.metadata import version

from . import correct, filters, io, seg

__all__ = ["io", "correct", "seg", "filters"]

__version__ = version("fishtank")
