from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.governance import (
    EconomicPrefixError,
    compare_economic_prefix,
    require_stable_economic_prefix,
)


def _holdings(*, weight_delta: float = 0.0, extend: bool = False) -> pl.DataFrame:
    rows = [
        {
            "strategy": "alpha",
            "decision_month": date(2020, 1, 1),
            "holding_month": date(2020, 2, 1),
            "ticker": "A",
            "target_weight": 0.6 + weight_delta,
            "realized_return": 0.10,
            "benchmark_return": 0.02,
            "selection_rank": 1,
        },
        {
            "strategy": "alpha",
            "decision_month": date(2020, 1, 1),
            "holding_month": date(2020, 2, 1),
            "ticker": "B",
            "target_weight": 0.4 - weight_delta,
            "realized_return": -0.05,
            "benchmark_return": 0.02,
            "selection_rank": 2,
        },
    ]
    if extend:
        rows.append(
            {
                "strategy": "alpha",
                "decision_month": date(2020, 2, 1),
                "holding_month": date(2020, 3, 1),
                "ticker": "C",
                "target_weight": 1.0,
                "realized_return": 0.03,
                "benchmark_return": 0.01,
                "selection_rank": 1,
            }
        )
    return pl.DataFrame(rows)


def _monthly(*, return_delta: float = 0.0, extend: bool = False) -> pl.DataFrame:
    rows = [
        {
            "strategy": "alpha",
            "decision_month": date(2020, 1, 1),
            "holding_month": date(2020, 2, 1),
            "gross_return": 0.04 + return_delta,
            "turnover": 1.0,
            "transaction_cost": 0.001,
            "net_return": 0.039 + return_delta,
            "benchmark_return": 0.02,
            "active_return": 0.019 + return_delta,
            "relative_return": 1.039 / 1.02 - 1.0 + return_delta,
            "n_positions": 2,
        }
    ]
    if extend:
        rows.append(
            {
                "strategy": "alpha",
                "decision_month": date(2020, 2, 1),
                "holding_month": date(2020, 3, 1),
                "gross_return": 0.03,
                "turnover": 1.0,
                "transaction_cost": 0.001,
                "net_return": 0.029,
                "benchmark_return": 0.01,
                "active_return": 0.019,
                "relative_return": 1.029 / 1.01 - 1.0,
                "n_positions": 1,
            }
        )
    return pl.DataFrame(rows)


def test_economic_prefix_is_bitwise_stable() -> None:
    report = compare_economic_prefix(
        reference_holdings=_holdings(),
        candidate_holdings=_holdings(extend=True),
        reference_monthly=_monthly(),
        candidate_monthly=_monthly(extend=True),
        numeric_tolerance=0.0,
        tolerance_justification=None,
    )

    assert report["passed"] is True
    assert report["through_holding_month"] == "2020-02-01"
    assert report["frames"]["holdings"]["reference_sha256"] == report[
        "frames"
    ]["holdings"]["candidate_sha256"]
    assert all(
        value["maximum_absolute_difference"] == 0.0
        for value in report["frames"]["monthly"]["numeric_columns"].values()
    )


def test_economic_prefix_uses_approved_numeric_tolerance() -> None:
    passing = compare_economic_prefix(
        reference_holdings=_holdings(),
        candidate_holdings=_holdings(weight_delta=5e-13),
        reference_monthly=_monthly(),
        candidate_monthly=_monthly(return_delta=5e-13),
    )
    failing = compare_economic_prefix(
        reference_holdings=_holdings(),
        candidate_holdings=_holdings(weight_delta=2e-12),
        reference_monthly=_monthly(),
        candidate_monthly=_monthly(return_delta=2e-12),
    )

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["frames"]["holdings"]["numeric_columns"]["target_weight"][
        "maximum_absolute_difference"
    ] == pytest.approx(2e-12)


def test_economic_prefix_rejects_changed_selection() -> None:
    changed = _holdings().with_columns(
        pl.when(pl.col("ticker") == "B")
        .then(pl.lit("C"))
        .otherwise(pl.col("ticker"))
        .alias("ticker")
    )

    with pytest.raises(EconomicPrefixError, match="holdings"):
        require_stable_economic_prefix(
            reference_holdings=_holdings(),
            candidate_holdings=changed,
            reference_monthly=_monthly(),
            candidate_monthly=_monthly(),
        )


def test_positive_tolerance_requires_documented_justification() -> None:
    with pytest.raises(ValueError, match="requires a justification"):
        compare_economic_prefix(
            reference_holdings=_holdings(),
            candidate_holdings=_holdings(),
            reference_monthly=_monthly(),
            candidate_monthly=_monthly(),
            numeric_tolerance=1e-12,
            tolerance_justification=None,
        )
