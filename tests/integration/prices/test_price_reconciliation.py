from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.ingestion.price_publication_candidate import (
    PricePublicationContext,
    build_price_publication_candidate,
)
from alpharank.data.ingestion.refresh_policy import SourceRefreshPolicy
from alpharank.data.prices import (
    HybridPriceResult,
    PriceReconciliationContext,
    reconcile_validated_price_history,
)
from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    PRICE_LINEAGE_COLUMNS,
    PRICE_VALUE_COLUMNS,
)


def _lineage(
    dates: list[str],
    adjusted_close: list[float],
    *,
    run_id: str,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": dates,
            "open": adjusted_close,
            "high": adjusted_close,
            "low": adjusted_close,
            "close": adjusted_close,
            "volume": [100.0] * len(dates),
            "adjusted_close": adjusted_close,
            "ticker": ["A.US"] * len(dates),
            "source": ["yfinance"] * len(dates),
            "dataset": ["prices_yfinance"] * len(dates),
            "ingestion_run_id": [run_id] * len(dates),
            "ingested_at": ["2026-08-27T10:00:00+00:00"] * len(dates),
            "source_vintage_id": [run_id] * len(dates),
            "return_source_vintage_id": [run_id] * len(dates),
            "adjustment_policy_version": [ADJUSTMENT_POLICY_VERSION] * len(dates),
            "adjustment_bridge_factor": [1.0] * len(dates),
            "eodhd_seed_sha256": ["seed"] * len(dates),
            "correction_overlay_id": [None] * len(dates),
        }
    ).select(PRICE_LINEAGE_COLUMNS)


def test_reconciliation_keeps_validated_rows_and_appends_provider_return() -> None:
    previous = _lineage(["2026-08-24", "2026-08-25"], [100.0, 101.0], run_id="old")
    provider = _lineage(
        ["2026-08-24", "2026-08-25", "2026-08-26"],
        [200.0, 204.0, 208.08],
        run_id="run_27",
    )

    result = reconcile_validated_price_history(
        previous_validated_lineage=previous,
        current_yahoo_observation=provider,
        context=PriceReconciliationContext(
            active_tickers=("A",),
            preserved_terminal_tickers=(),
            incomplete_provider_tickers=(),
            run_id="run_27",
        ),
    )

    selected = result.prices.sort("date")["adjusted_close"].to_list()
    assert selected == pytest.approx([100.0, 101.0, 103.02])
    assert result.lineage.head(2).equals(previous, null_equal=True)
    assert result.report["previous_validated_rows_changed"] == 0
    assert result.report["return_extension_rows"] == 1
    assert result.extension_audit["provider_daily_return"].to_list() == pytest.approx([0.02])


def test_publication_resolves_provider_revision_without_overwriting_history() -> None:
    previous = _lineage(["2026-08-10", "2026-08-11"], [100.0, 101.0], run_id="old")
    provider = _lineage(
        ["2026-08-10", "2026-08-11", "2026-08-12"],
        [200.0, 204.0, 208.08],
        run_id="run_27",
    )
    candidate = build_price_publication_candidate(
        HybridPriceResult(
            prices=provider.select(PRICE_VALUE_COLUMNS),
            lineage=provider,
            composition_report={},
        ),
        provider,
        previous,
        context=PricePublicationContext(
            active_tickers=("A",),
            preserved_terminal_tickers=(),
            expected_eodhd_keys=pl.DataFrame(),
            expected_through="2026-08-27",
            run_id="run_27",
            policy=SourceRefreshPolicy().price_gate_policy(),
            previous_comparison_prices=previous.select(PRICE_VALUE_COLUMNS),
        ),
    )

    assert candidate.provider_gate.report["passed"] is False
    assert candidate.gate.report["passed"] is True
    assert candidate.gate.report["resolved_provider_blocking_reasons"] == [
        "unreviewed_historical_return_revisions"
    ]
    assert candidate.hybrid.prices["adjusted_close"].to_list() == pytest.approx(
        [100.0, 101.0, 103.02]
    )


def test_reconciliation_blocks_publication_when_provider_anchor_is_missing() -> None:
    previous = _lineage(["2026-08-10", "2026-08-11"], [100.0, 101.0], run_id="old")
    provider = _lineage(["2026-08-10", "2026-08-12"], [200.0, 208.08], run_id="run_27")

    result = reconcile_validated_price_history(
        previous_validated_lineage=previous,
        current_yahoo_observation=provider,
        context=PriceReconciliationContext(
            active_tickers=("A",),
            preserved_terminal_tickers=(),
            incomplete_provider_tickers=(),
            run_id="run_27",
        ),
    )

    assert result.report["passed"] is False
    assert result.report["blocking_reasons"] == ["unresolved_validated_return_extension"]
    assert result.lineage.equals(previous, null_equal=True)


def test_reconciliation_retains_reviewed_incomplete_provider_prefix() -> None:
    previous = _lineage(["2026-08-10", "2026-08-11"], [100.0, 101.0], run_id="old")
    provider = _lineage(["2026-08-10", "2026-08-12"], [200.0, 208.08], run_id="run_27")

    result = reconcile_validated_price_history(
        previous_validated_lineage=previous,
        current_yahoo_observation=provider,
        context=PriceReconciliationContext(
            active_tickers=("A",),
            preserved_terminal_tickers=(),
            incomplete_provider_tickers=("A",),
            run_id="run_27",
        ),
    )

    assert result.report["passed"] is True
    assert result.report["retained_incomplete_provider_tickers"] == [
        {"ticker": "A.US", "reason": "provider_anchor_for_validated_tail_missing"}
    ]
    assert result.lineage.equals(previous, null_equal=True)


def test_reconciliation_uses_carried_def_anchor_for_new_provider_dates() -> None:
    previous = _lineage(["2026-08-10", "2026-08-11"], [100.0, 101.0], run_id="old")
    carried_anchor = previous.tail(1)
    provider_tail = _lineage(["2026-08-12"], [103.02], run_id="run_27")
    definitive_resolution = pl.concat([carried_anchor, provider_tail])

    result = reconcile_validated_price_history(
        previous_validated_lineage=previous,
        current_yahoo_observation=definitive_resolution,
        context=PriceReconciliationContext(
            active_tickers=("A",),
            preserved_terminal_tickers=(),
            incomplete_provider_tickers=("A",),
            run_id="run_27",
        ),
    )

    assert result.report["passed"] is True
    assert result.report["retained_incomplete_provider_tickers"] == []
    assert result.report["return_extension_rows"] == 1
    assert result.prices.sort("date")["adjusted_close"].to_list() == pytest.approx(
        [100.0, 101.0, 103.02]
    )
    assert result.extension_audit.select(
        "validated_anchor_date", "date", "provider_daily_return"
    ).to_dicts() == [
        {
            "validated_anchor_date": "2026-08-11",
            "date": "2026-08-12",
            "provider_daily_return": pytest.approx(0.02),
        }
    ]
