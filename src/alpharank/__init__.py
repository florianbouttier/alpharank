"""AlphaRank core library.

Public surface:
- ``alpharank.legacy`` for the legacy production workflow
- ``alpharank.boosting`` for the experimental boosting workflow
"""

from alpharank import boosting, legacy

__all__ = ["legacy", "boosting"]
