"""Shared portfolio construction, simulation, and performance primitives.

Signal generation intentionally remains outside this package.  Legacy EMA and
boosting models adapt their decisions to the same holdings contract before
returns and statistics are computed.
"""

from alpharank.portfolio.allocation import (
    capped_inverse_risk_weights,
    constrained_inverse_risk_weights,
    equal_weights,
    portfolio_turnover,
    select_ranked_candidates,
)
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import align_return_series, reference_monthly_series
from alpharank.portfolio.contracts import (
    HOLDINGS_REQUIRED_COLUMNS,
    MONTHLY_REQUIRED_COLUMNS,
    validate_holdings,
    validate_monthly_returns,
)
from alpharank.portfolio.performance import (
    annual_returns,
    legacy_report_statistics,
    performance_statistics,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio

__all__ = [
    "HOLDINGS_REQUIRED_COLUMNS",
    "MONTHLY_REQUIRED_COLUMNS",
    "align_return_series",
    "annual_returns",
    "capped_inverse_risk_weights",
    "constrained_inverse_risk_weights",
    "equal_weights",
    "legacy_report_statistics",
    "performance_statistics",
    "portfolio_turnover",
    "reference_monthly_series",
    "select_ranked_candidates",
    "simulate_weighted_portfolio",
    "validate_holdings",
    "validate_monthly_returns",
    "write_common_portfolio_artifacts",
]
