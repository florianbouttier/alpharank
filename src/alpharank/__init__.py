"""AlphaRank core library.

Public surface:
- ``alpharank.legacy`` for the legacy production workflow
- ``alpharank.boosting`` for the experimental boosting workflow
"""

from importlib import import_module

__all__ = ["legacy", "boosting"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"alpharank.{name}")
    raise AttributeError(f"module 'alpharank' has no attribute {name!r}")
