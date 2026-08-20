"""AlphaRank core library with an intentional package-level API."""

from importlib import import_module
from types import ModuleType

__all__ = ["replay"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f"alpharank.{name}")
    raise AttributeError(f"module 'alpharank' has no attribute {name!r}")
