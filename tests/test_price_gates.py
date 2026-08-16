from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.data.prices.contracts import PriceGatePolicy
from alpharank.data.prices.gates import audit_price_candidate, validate_price_candidate


def _prices(adjusted: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["MSFT.US"] * len(adjusted),
            "date": [date(2026, 3, 23), date(2026, 3, 24)][: len(adjusted)],
            "adjusted_close": adjusted,
        }
    )


def _lineage(adjusted: list[float], vintages: list[str], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["MSFT.US"] * len(adjusted),
            "date": [date(2026, 3, 23), date(2026, 3, 24)][: len(adjusted)],
            "close": closes,
            "adjusted_close": adjusted,
            "source": ["yfinance"] * len(adjusted),
            "source_vintage_id": vintages,
        }
    )


def test_uniform_dividend_restatement_does_not_create_material_return_revision() -> None:
    previous = _prices([383.0, 372.73999])
    candidate = _prices([382.172272, 371.934418])
    lineage = _lineage(candidate["adjusted_close"].to_list(), ["full", "full"], [383.0, 372.73999])

    result = audit_price_candidate(
        previous_prices=previous,
        candidate_prices=candidate,
        candidate_lineage=lineage,
        active_tickers=["MSFT"],
        expected_through="2026-08-10",
    )

    assert result.report["transition_factor_findings"] == 0
    assert result.report["historical_daily_return_revisions_over_threshold"] == 0
    assert result.report["passed"] is True


def test_microsoft_mixed_vintage_seam_is_rejected() -> None:
    previous = _prices([383.0, 371.934418])
    candidate = _prices([383.0, 371.934418])
    lineage = _lineage(candidate["adjusted_close"].to_list(), ["old", "new"], [383.0, 372.73999])
    policy = PriceGatePolicy(allow_historical_price_revisions=True)

    result = audit_price_candidate(
        previous_prices=previous,
        candidate_prices=candidate,
        candidate_lineage=lineage,
        active_tickers=["MSFT"],
        expected_through="2026-08-10",
        policy=policy,
    )

    assert result.report["mixed_active_yahoo_vintage_tickers"] == ["MSFT.US"]
    assert result.report["transition_factor_findings"] == 1
    with pytest.raises(RuntimeError, match="publication gates"):
        validate_price_candidate(result)


def test_missing_inactive_eodhd_key_is_rejected() -> None:
    candidate = _prices([100.0, 101.0])
    lineage = _lineage([100.0, 101.0], ["full", "full"], [100.0, 101.0])
    expected_seed = pl.DataFrame(
        {"ticker": ["OLD.US"], "date": [date(2020, 1, 2)]}
    )

    result = audit_price_candidate(
        previous_prices=None,
        candidate_prices=candidate,
        candidate_lineage=lineage,
        active_tickers=["MSFT"],
        expected_eodhd_keys=expected_seed,
        expected_through="2026-08-10",
    )

    assert result.report["missing_inactive_eodhd_seed_keys"] == 1
    assert "eodhd_seed_coverage_lost" in result.report["blocking_reasons"]


def test_active_universe_requires_one_global_yahoo_vintage() -> None:
    candidate = pl.DataFrame(
        {
            "ticker": ["MSFT.US", "AAPL.US"],
            "date": [date(2026, 3, 23), date(2026, 3, 23)],
            "adjusted_close": [100.0, 200.0],
        }
    )
    lineage = candidate.with_columns(
        pl.col("adjusted_close").alias("close"),
        pl.lit("yfinance").alias("source"),
        pl.Series("source_vintage_id", ["run_1", "run_2"]),
    )

    result = audit_price_candidate(
        previous_prices=None,
        candidate_prices=candidate,
        candidate_lineage=lineage,
        active_tickers=["MSFT", "AAPL"],
        expected_through="2026-03-23",
    )

    assert sorted(result.report["active_global_yahoo_vintage_ids"]) == [
        "run_1",
        "run_2",
    ]
    assert "active_universe_not_one_global_yahoo_vintage" in result.report[
        "blocking_reasons"
    ]
