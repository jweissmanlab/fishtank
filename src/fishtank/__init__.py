from importlib.metadata import version

from . import correct, io, seg

__all__ = ["io", "correct", "seg"]

__version__ = version("fishtank")
