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
from alpharank.portfolio.attribution import (
    portfolio_return_attribution,
    reference_return_attribution,
)
from alpharank.portfolio.benchmark import (
    BENCHMARK_CONVENTIONS,
    SPY_PRICE_RETURN,
    SPY_TOTAL_RETURN,
    benchmark_convention,
    completed_through_month,
    monthly_benchmark_returns,
)
from alpharank.portfolio.comparison import align_return_series, reference_monthly_series
from alpharank.portfolio.costs import TransactionCostModel, transaction_cost_components
from alpharank.portfolio.contracts import (
    CAUSAL_TIMING_REQUIRED_COLUMNS,
    HOLDINGS_REQUIRED_COLUMNS,
    MONTHLY_REQUIRED_COLUMNS,
    validate_holdings,
    validate_causal_timing,
    validate_monthly_returns,
)
from alpharank.portfolio.performance import (
    advanced_performance_statistics,
    annual_returns,
    legacy_report_statistics,
    performance_statistics,
)
from alpharank.portfolio.lineage import (
    compare_input_hashes,
    compare_ticker_exclusions,
    input_hashes_from_manifest,
    require_matching_data_contexts,
    require_matching_price_eligibility,
    require_matching_ticker_exclusions,
    ticker_exclusions_from_manifest,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.terminal_returns import (
    SUCCESSOR_PRICE_COLUMNS,
    TERMINAL_EVENT_COLUMNS,
    TERMINAL_EVENT_TYPES,
    TerminalReturnResult,
    resolve_terminal_shareholder_returns,
)

__all__ = [
    "CAUSAL_TIMING_REQUIRED_COLUMNS",
    "HOLDINGS_REQUIRED_COLUMNS",
    "MONTHLY_REQUIRED_COLUMNS",
    "BENCHMARK_CONVENTIONS",
    "SPY_PRICE_RETURN",
    "SPY_TOTAL_RETURN",
    "SUCCESSOR_PRICE_COLUMNS",
    "TERMINAL_EVENT_COLUMNS",
    "TERMINAL_EVENT_TYPES",
    "TerminalReturnResult",
    "TransactionCostModel",
    "align_return_series",
    "advanced_performance_statistics",
    "annual_returns",
    "benchmark_convention",
    "completed_through_month",
    "capped_inverse_risk_weights",
    "compare_input_hashes",
    "compare_ticker_exclusions",
    "constrained_inverse_risk_weights",
    "equal_weights",
    "legacy_report_statistics",
    "monthly_benchmark_returns",
    "input_hashes_from_manifest",
    "performance_statistics",
    "portfolio_turnover",
    "portfolio_return_attribution",
    "reference_monthly_series",
    "reference_return_attribution",
    "resolve_terminal_shareholder_returns",
    "require_matching_data_contexts",
    "require_matching_price_eligibility",
    "require_matching_ticker_exclusions",
    "select_ranked_candidates",
    "simulate_weighted_portfolio",
    "validate_causal_timing",
    "ticker_exclusions_from_manifest",
    "transaction_cost_components",
    "validate_holdings",
    "validate_monthly_returns",
    "write_common_portfolio_artifacts",
]
