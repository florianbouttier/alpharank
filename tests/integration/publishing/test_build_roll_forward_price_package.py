from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.prices.contracts import ADJUSTMENT_POLICY_VERSION, PRICE_LINEAGE_COLUMNS
from alpharank.data.publishing.acquired_price_run import (
    REQUIRED_ACQUISITION_SOURCES,
    load_acquisition_run_manifest,
    resolve_acquired_previous_lineage,
)
from alpharank.data.publishing.price_package_inputs import (
    prepare_benchmark_prices,
    resolve_active_resolution_vintage_id,
)
from alpharank.data.publishing.price_package_output import PricePackageRequest
from alpharank.data.publishing.price_roll_forward import _prepare_roll_forward_evidence


def _price_lineage(
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
            "ingested_at": ["2026-08-27T07:06:54+00:00"] * len(dates),
            "source_vintage_id": [run_id] * len(dates),
            "return_source_vintage_id": [run_id] * len(dates),
            "adjustment_policy_version": [ADJUSTMENT_POLICY_VERSION] * len(dates),
            "adjustment_bridge_factor": [1.0] * len(dates),
            "eodhd_seed_sha256": ["seed"] * len(dates),
            "correction_overlay_id": [None] * len(dates),
        }
    ).select(PRICE_LINEAGE_COLUMNS)


def test_builder_binds_audited_carries_to_full_ingestion_run() -> None:
    fresh_yahoo = pl.DataFrame(
        {
            "ticker": ["AVB.US", "AVB.US"],
            "ingestion_run_id": ["20260816_103942", "20260819_220746"],
        }
    )

    assert (
        resolve_active_resolution_vintage_id(
            run_id="20260819_220746",
            fresh_yahoo=fresh_yahoo,
        )
        == "20260819_220746"
    )


def test_builder_rejects_a_fresh_vintage_without_current_run_observation() -> None:
    fresh_yahoo = pl.DataFrame(
        {
            "ticker": ["AVB.US"],
            "ingestion_run_id": ["20260816_103942"],
        }
    )

    with pytest.raises(RuntimeError, match="no observation"):
        resolve_active_resolution_vintage_id(
            run_id="20260819_220746",
            fresh_yahoo=fresh_yahoo,
        )


def test_deferred_builder_preserves_validated_returns_before_appending_tail(
    tmp_path: Path,
) -> None:
    previous = _price_lineage(
        ["2024-08-10", "2024-08-11"],
        [100.0, 101.0],
        run_id="validated",
    )
    provider = _price_lineage(
        ["2024-08-10", "2024-08-11", "2026-08-26"],
        [200.0, 204.0, 208.08],
        run_id="20260827_070654",
    )
    previous_path = tmp_path / "previous.parquet"
    provider_path = tmp_path / "provider.parquet"
    seed_path = tmp_path / "seed.parquet"
    constituents_path = tmp_path / "constituents.csv"
    registry_path = tmp_path / "reviewed_moves.json"
    previous.write_parquet(previous_path)
    provider.write_parquet(provider_path)
    previous.select(
        "date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"
    ).write_parquet(seed_path)
    pl.DataFrame({"Date": ["2026-08-01"], "Ticker": ["A"]}).write_csv(constituents_path)
    registry_path.write_text(
        '{"registry_id":"reviewed_extreme_price_moves_test_v1","events":[]}',
        encoding="utf-8",
    )
    request = PricePackageRequest(
        run_id="20260827_070654",
        source_refresh_contract={"source_semantics": {}},
        previous_lineage_path=previous_path,
        previous_resolution="test",
        previous_composition_id=None,
        fresh_yahoo_path=provider_path,
        benchmark_path=tmp_path / "unused-benchmark.parquet",
        constituents_path=constituents_path,
        eodhd_seed_path=seed_path,
        output_dir=tmp_path / "unused-output",
        expected_through="2026-08-27",
        start_date="2005-01-01",
        preserve_terminal_tickers=(),
        constituent_registry_path=tmp_path / "unused-terminal-registry.json",
        reviewed_move_registry_path=registry_path,
    )

    evidence = _prepare_roll_forward_evidence(request)

    assert evidence.revision_gate.report["passed"] is True
    assert evidence.revision_gate.report["resolved_provider_blocking_reasons"] == [
        "unreviewed_historical_return_revisions"
    ]
    assert evidence.result.prices.sort("date")["adjusted_close"].to_list() == pytest.approx(
        [100.0, 101.0, 103.02]
    )


def test_builder_accepts_completed_quarantined_acquisition_without_network(
    tmp_path: Path,
) -> None:
    run_id = "20260827_070654"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    previous_lineage = tmp_path / "previous.parquet"
    pl.DataFrame({"ticker": ["RDDT.US"], "date": ["2024-10-29"]}).write_parquet(previous_lineage)
    sources = [
        {
            "source": source,
            "status": ("downloaded_quarantined" if source == "yahoo_prices" else "downloaded"),
            "downloaded_rows": 1,
        }
        for source in sorted(REQUIRED_ACQUISITION_SOURCES)
    ]
    (run_dir / "acquisition_status.json").write_text(
        json.dumps(
            {
                "contract": "alpharank_source_acquisition_status_v1",
                "run_id": run_id,
                "phase": "all_declared_sources_attempted_before_publication_decision",
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "source_refresh_contract.json").write_text(
        json.dumps(
            {
                "snapshot_scope": "full_ingestion",
                "policy": {"require_eodhd_price_seed": True},
                "previous_validated_price_lineage": {"path": str(previous_lineage)},
                "source_semantics": {
                    "yfinance_prices": {
                        "network_missing_tickers": [],
                        "benchmark_network_missing_tickers": [],
                    },
                    "sec_companyfacts": {"active_network_complete": True},
                    "sec_submissions": {"active_network_complete": True},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_acquisition_run_manifest(run_dir)

    assert manifest["run_id"] == run_id
    assert resolve_acquired_previous_lineage(manifest, explicit=None) == previous_lineage


def test_builder_rejects_incomplete_acquisition_phase(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260827_070654"
    run_dir.mkdir()
    (run_dir / "acquisition_status.json").write_text(
        json.dumps(
            {
                "contract": "alpharank_source_acquisition_status_v1",
                "run_id": run_dir.name,
                "phase": "prices_only",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "source_refresh_contract.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="did not finish"):
        load_acquisition_run_manifest(run_dir)


def test_builder_binds_acquired_benchmark_to_same_run(tmp_path: Path) -> None:
    path = tmp_path / "prices_spy_yfinance.parquet"
    pl.DataFrame(
        {
            "ticker": ["SPY.US"],
            "date": ["2026-08-26"],
            "adjusted_close": [700.0],
            "close": [700.0],
            "open": [699.0],
            "high": [701.0],
            "low": [698.0],
            "volume": [1_000_000.0],
            "ingestion_run_id": ["20260827_070654"],
        }
    ).write_parquet(path)

    result = prepare_benchmark_prices(path, expected_run_id="20260827_070654")

    assert result.columns == [
        "ticker",
        "date",
        "adjusted_close",
        "close",
        "open",
        "high",
        "low",
        "volume",
    ]
    with pytest.raises(RuntimeError, match="not run-bound"):
        prepare_benchmark_prices(path, expected_run_id="20260827_999999")
