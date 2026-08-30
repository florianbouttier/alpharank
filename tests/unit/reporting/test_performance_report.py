from __future__ import annotations

import base64
import gzip
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.portfolio.performance import portfolio_period_statistics
from alpharank.reporting.performance_report import (
    BENCHMARK_STRATEGY,
    PERFORMANCE_METRIC_FIELDS,
    STRATEGY_SPECS,
    PerformanceReportInputs,
    build_performance_report_payload,
    write_performance_report,
)


def test_payload_uses_canonical_window_kpi_and_keeps_every_holding(tmp_path: Path) -> None:
    inputs = _report_inputs(tmp_path)
    payload = build_performance_report_payload(
        inputs,
        generated_at_utc=datetime(2026, 8, 29, 15, 0, tzinfo=timezone.utc),
    )

    expected = portfolio_period_statistics(
        [0.01, -0.02],
        benchmark_returns=[0.005, 0.006],
        turnovers=[0.2, 0.3],
        transaction_costs=[0.0002, 0.0003],
        position_counts=[1, 1],
        maximum_position_weights=[1.0, 1.0],
        maximum_sector_weights=[1.0, 1.0],
    )
    window = payload["metric_windows"]["2026-01-01|2026-02-01"]
    legacy_frequency = window[payload["strategy_order"].index("Legacy · Frequency")]
    metric_index = payload["metric_fields"].index("cagr")

    assert payload["calendar"] == {
        "months": 2,
        "start": "2026-01-01",
        "end": "2026-02-01",
        "available_months": ["2026-01-01", "2026-02-01"],
        "available_start_months": ["2026-01-01"],
        "available_end_months": ["2026-02-01"],
    }
    assert legacy_frequency[metric_index] == pytest.approx(expected["cagr"])
    assert len(payload["strategy_order"]) == 11
    assert len(payload["holdings"]) == 20
    assert BENCHMARK_STRATEGY in payload["strategy_order"]
    assert payload["data_quality"]["sector_coverage_by_strategy"][
        "Legacy · Frequency"
    ] == pytest.approx(1.0)


def test_html_is_self_contained_and_embeds_a_valid_compressed_payload(tmp_path: Path) -> None:
    paths = write_performance_report(
        _report_inputs(tmp_path),
        output_dir=tmp_path / "report",
        generated_at_utc=datetime(2026, 8, 29, 15, 0, tzinfo=timezone.utc),
    )
    html = paths["report"].read_text(encoding="utf-8")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    match = re.search(r"const PAYLOAD_GZIP_BASE64=(\"[^\"]+\")", html)

    assert match is not None
    payload = json.loads(gzip.decompress(base64.b64decode(json.loads(match.group(1)))))
    assert payload["metric_fields"] == list(PERFORMANCE_METRIC_FIELDS)
    assert payload["lineage"]["composition_id"] == "test-composition"
    assert manifest["report"]["sha256"]
    assert "https://" not in html
    assert "http://" not in html
    assert "Viridis" not in html  # The requested scale is encoded locally, not loaded.
    assert ".loading[hidden]" in html
    assert 'id="curve-multiselect"' in html
    assert 'id="metric-head"' in html
    assert 'id="cumulative-heatmap"' in html
    assert 'id="incremental-heatmap"' in html
    assert 'id="strategy-select"' not in html
    assert 'matrixWindows("cumulative")' in html
    assert 'matrixWindows("incremental")' in html
    assert "Tous les KPI des courbes affichées" in html
    assert "state.curves.map(strategy" in html
    assert ".chart-grid { display: grid; grid-template-columns: 1fr;" in html


def _report_inputs(tmp_path: Path) -> PerformanceReportInputs:
    common_dir = tmp_path / "common"
    legacy_dir = tmp_path / "legacy"
    common_dir.mkdir()
    legacy_dir.mkdir()
    months = [date(2026, 1, 1), date(2026, 2, 1)]
    common_strategies = sorted(
        {spec.source_strategy for spec in STRATEGY_SPECS if spec.source == "common"}
    )
    _monthly(common_strategies, months).write_parquet(
        common_dir / "comparison_common_monthly.parquet"
    )
    _monthly(["Combined_Equal"], months).write_parquet(legacy_dir / "legacy_common_monthly.parquet")
    _holdings(
        [name for name in common_strategies if name != "SPY total return"], months
    ).write_parquet(common_dir / "comparison_common_holdings.parquet")
    _holdings(["Combined_Equal"], months).write_parquet(
        legacy_dir / "legacy_common_holdings.parquet"
    )
    (common_dir / "manifest.json").write_text(
        json.dumps(
            {
                "comparison_eligible": True,
                "publication_eligible": False,
                "methodology_status": "test",
                "timing_contract": "decision_month=t; holding_month=t+1",
                "transaction_cost_policy": {"bps_times_turnover": 10},
                "status": "test_replay",
                "comparison_profile": {"name": "test"},
                "runtime_provenance": {"git": {"head": "abc123"}},
            }
        ),
        encoding="utf-8",
    )
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_manifest.write_text(
        json.dumps({"composition_id": "test-composition", "snapshot_dir": "/snapshot"}),
        encoding="utf-8",
    )
    return PerformanceReportInputs(common_dir, legacy_dir, snapshot_manifest)


def _monthly(strategies: list[str], months: list[date]) -> pl.DataFrame:
    rows = []
    for strategy in strategies:
        returns = [0.005, 0.006] if strategy == "SPY total return" else [0.01, -0.02]
        for index, month in enumerate(months):
            rows.append(
                {
                    "strategy": strategy,
                    "decision_month": date(
                        month.year - (month.month == 1),
                        12 if month.month == 1 else month.month - 1,
                        1,
                    ),
                    "holding_month": month,
                    "net_return": returns[index],
                    "benchmark_return": [0.005, 0.006][index],
                    "turnover": 0.0 if strategy == "SPY total return" else [0.2, 0.3][index],
                    "transaction_cost": 0.0
                    if strategy == "SPY total return"
                    else [0.0002, 0.0003][index],
                    "n_positions": 0 if strategy == "SPY total return" else 1,
                    "maximum_position_weight": 0.0 if strategy == "SPY total return" else 1.0,
                    "maximum_sector_weight": 0.0 if strategy == "SPY total return" else 1.0,
                }
            )
    return pl.DataFrame(rows)


def _holdings(strategies: list[str], months: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "strategy": strategy,
                "decision_month": date(
                    month.year - (month.month == 1), 12 if month.month == 1 else month.month - 1, 1
                ),
                "holding_month": month,
                "ticker": f"{strategy[:3].upper()}{index}",
                "target_weight": 1.0,
                "realized_return": [0.01, -0.02][index],
                "selection_rank": 1,
                "score": 0.5,
                "sector": "Test",
                "n_models": 1,
            }
            for strategy in strategies
            for index, month in enumerate(months)
        ]
    )
