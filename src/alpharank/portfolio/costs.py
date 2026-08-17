"""Versionable transaction-cost scenarios for the shared simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TransactionCostModel:
    scenario_id: str
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    commission_bps: float = 0.0
    minimum_fee_currency: float = 0.0
    portfolio_value_currency: float = 1_000_000.0
    fx_bps: float = 0.0
    fx_turnover_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("Transaction cost scenario_id must be non-empty.")
        numeric = (
            self.spread_bps,
            self.slippage_bps,
            self.impact_bps,
            self.commission_bps,
            self.minimum_fee_currency,
            self.portfolio_value_currency,
            self.fx_bps,
            self.fx_turnover_fraction,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Transaction cost parameters must be finite.")
        if any(value < 0.0 for value in numeric):
            raise ValueError("Transaction cost parameters must be non-negative.")
        if self.portfolio_value_currency <= 0.0:
            raise ValueError("portfolio_value_currency must be positive.")
        if self.fx_turnover_fraction > 1.0:
            raise ValueError("fx_turnover_fraction cannot exceed one.")


def transaction_cost_components(
    turnover: float, model: TransactionCostModel
) -> dict[str, float | str]:
    """Return reconciled costs as fractions of portfolio value."""

    if not math.isfinite(turnover) or turnover < 0.0:
        raise ValueError("turnover must be finite and non-negative.")
    spread = turnover * model.spread_bps / 10_000.0
    slippage = turnover * model.slippage_bps / 10_000.0
    impact = turnover * model.impact_bps / 10_000.0
    commission_variable = turnover * model.commission_bps / 10_000.0
    commission_minimum = (
        model.minimum_fee_currency / model.portfolio_value_currency
        if turnover > 0.0
        else 0.0
    )
    commission = max(commission_variable, commission_minimum)
    fx = (
        turnover
        * model.fx_turnover_fraction
        * model.fx_bps
        / 10_000.0
    )
    total = spread + slippage + impact + commission + fx
    return {
        "cost_scenario_id": model.scenario_id,
        "spread_cost": spread,
        "slippage_cost": slippage,
        "impact_cost": impact,
        "commission_cost": commission,
        "fx_cost": fx,
        "transaction_cost": total,
    }
