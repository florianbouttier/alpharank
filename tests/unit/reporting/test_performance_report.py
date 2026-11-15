from __future__ import annotations

import base64
import gzip
import json
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from shutil import copyfile

import polars as pl
import pytest

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio.performance import portfolio_period_statistics
from alpharank.replay.sector_allocation import SECTOR_CHAMPION_POLICY_ID
from alpharank.replay.sector_backtest import REFERENCE_STRATEGIES, SECTOR_NATIVE, SECTOR_TREND
from alpharank.reporting.performance_report import (
    BENCHMARK_STRATEGY,
    PERFORMANCE_METRIC_FIELDS,
    STRATEGY_SPECS,
    PerformanceReportInputs,
    SectorReportInputs,
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
    assert payload["current_portfolio"]["as_of_date"] == "2026-03-28"
    assert payload["current_portfolio"]["decision_month"] == "2026-02-01"
    assert payload["current_portfolio"]["holding_month"] == "2026-03-01"
    assert payload["current_portfolio"]["status"] == "in_force_return_pending"
    assert len(payload["current_portfolio"]["holdings"]) == 10
    assert {row["strategy"] for row in payload["current_portfolio"]["holdings"]} == {
        spec.label for spec in STRATEGY_SPECS if spec.label != BENCHMARK_STRATEGY
    }
    assert all(row["realized_return"] is None for row in payload["current_portfolio"]["holdings"])
    assert BENCHMARK_STRATEGY in payload["strategy_order"]
    assert payload["data_quality"]["sector_coverage_by_strategy"][
        "Legacy · Frequency"
    ] == pytest.approx(1.0)
    composer = payload["portfolio_composer"]
    assert len(composer["strategy_order"]) == 10
    assert len(composer["combination_masks"]) == 1_023
    assert composer["combination_masks"][:3] == [1, 2, 3]
    assert composer["monthly_returns"][0][:3] == pytest.approx([0.01, 0.01, 0.01])
    assert len(composer["metric_windows"]["2026-01-01|2026-02-01"]) == 1_023
    assert composer["strategy_correlation_windows"]["2026-01-01|2026-02-01"][0][1] == pytest.approx(
        1.0
    )
    assert "correlation" in composer["metric_fields"]
    assert composer["policy"] == {
        "method": "monthly_equal_weight_strategy_sleeves",
        "rebalance_frequency": "monthly",
        "input_returns": "net_return après les frais propres à chaque stratégie",
        "additional_inter_sleeve_cost": 0.0,
        "correlation": "Pearson sur les rendements mensuels de la fenêtre",
        "relative_wealth": "richesse composée divisée par la richesse SPY",
        "status": "diagnostic post-hoc non promu",
    }
    frontiers = payload["strategy_frontiers"]
    assert frontiers["strategy_order"] == [
        "Legacy · Frequency",
        "Boosting · Top 5",
        "Boosting tendance · Top 5",
    ]
    assert len(frontiers["weights"]) == 168
    assert all(sum(weights) == pytest.approx(1.0) for weights in frontiers["weights"])
    frontier_window = frontiers["windows"]["2026-01-01|2026-02-01"]
    assert len(frontier_window["candidate_metrics"]) == 168
    assert set(frontier_window["recommendations"]) == {
        "minimum_volatility",
        "maximum_sharpe",
        "maximum_calmar",
    }
    assert frontiers["policy"]["weight_step"] == pytest.approx(0.05)
    assert frontiers["policy"]["maximum_strategy_weight"] == pytest.approx(0.70)


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
    assert manifest["portfolio_composer"]["combinations"] == 1_023
    assert manifest["strategy_frontiers"]["candidate_allocations"] == 168
    assert "https://" not in html
    assert "http://" not in html
    assert "Viridis" not in html  # The requested scale is encoded locally, not loaded.
    assert ".loading[hidden]" in html
    assert 'id="analysis-studio"' in html
    assert 'data-analysis-mode="strategies"' in html
    assert 'data-analysis-mode="composer"' in html
    assert 'data-analysis-mode="frontier"' in html
    assert 'id="curve-multiselect"' in html
    assert 'id="composer-multiselect"' in html
    assert 'id="studio-kpis"' in html
    assert 'data-chart-view="wealth"' in html
    assert 'data-chart-view="drawdown"' in html
    assert 'data-chart-view="relative"' in html
    assert 'id="studio-chart"' in html
    assert 'id="studio-chart-tooltip"' in html
    assert 'id="studio-outcome"' in html
    assert 'id="metric-head"' in html
    assert 'id="cumulative-heatmap"' in html
    assert 'id="incremental-heatmap"' in html
    assert 'id="composer-options"' in html
    assert 'id="composer-correlation-matrix"' in html
    assert 'id="composer-boosting-pair"' in html
    assert 'id="frontier-studio"' in html
    assert 'id="frontier-recommendations"' in html
    assert 'id="frontier-volatility-chart"' in html
    assert 'id="frontier-drawdown-chart"' in html
    assert 'id="frontier-anchor-body"' in html
    assert 'id="current-portfolio"' in html
    assert 'id="current-portfolio-strategy"' in html
    assert 'id="current-holdings-body"' in html
    assert "Portefeuille en vigueur" in html
    assert "Rendement mensuel non réalisé" in html
    assert 'id="strategy-select"' not in html
    assert 'id="composer-wealth-chart"' not in html
    assert 'id="composer-drawdown-chart"' not in html
    assert 'id="composer-relative-chart"' not in html
    assert '<section class="section" id="composer">' not in html
    assert 'matrixWindows("cumulative")' in html
    assert 'matrixWindows("incremental")' in html
    assert "Tous les KPI de la sélection" in html
    assert "function setAnalysisMode(mode)" in html
    assert "function drawStudioChart()" in html
    assert "function renderStudioOutcome()" in html
    assert "function comparisonDeltaText(field, value, benchmark)" in html
    assert "function studioDisplaySeries()" in html
    assert "aucun coût supplémentaire entre poches" in html
    assert "function composerMetricValue(field)" in html
    assert "function renderComposerCorrelation()" in html
    assert "function composerRelativeWealthSeries()" in html
    assert "function renderFrontierStudio()" in html
    assert "function drawFrontierScatter(canvas, model)" in html
    assert "function resizeStudioCharts()" in html
    ids = re.findall(r'id="([^"]+)"', html)
    assert len(ids) == len(set(ids))


def test_dashboard_shell_keeps_the_complete_information_contract(tmp_path: Path) -> None:
    paths = write_performance_report(
        _report_inputs(tmp_path),
        output_dir=tmp_path / "dashboard",
        generated_at_utc=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
    )
    html = paths["report"].read_text(encoding="utf-8")
    match = re.search(r"const PAYLOAD_GZIP_BASE64=(\"[^\"]+\")", html)

    assert match is not None
    payload = json.loads(gzip.decompress(base64.b64decode(json.loads(match.group(1)))))
    for element_id in (
        "theme-toggle",
        "coverage-series",
        "coverage-kpis",
        "coverage-months",
        "coverage-holdings",
    ):
        assert f'id="{element_id}"' in html
    assert "Comparer au SPY" in html
    assert 'class="skip-link"' in html
    assert "function renderCoverage()" in html
    assert "function initializeTheme()" in html
    assert "function renderStrategyQuickPicks()" in html
    assert "function setQuickPeriod(year)" in html
    assert 'id="strategy-quick-picks"' in html
    assert 'data-period-start-year="2020"' in html
    assert 'class="scoreboard"' in html
    assert 'class="kpi-card"' not in html
    assert 'html[data-theme="dark"]' in html
    assert len(payload["strategy_order"]) == 11
    assert len(payload["metric_fields"]) == 33
    assert len(payload["methodologies"]) >= 5
    assert payload["holdings"]
    assert payload["current_portfolio"]["holdings"]
    assert payload["strategy_frontiers"]["windows"]


def test_sector_profile_keeps_all_realized_series_and_separates_last_basket(tmp_path: Path) -> None:
    inputs = _sector_report_inputs(tmp_path)
    payload = build_performance_report_payload(
        inputs, generated_at_utc=datetime(2026, 11, 15, tzinfo=timezone.utc)
    )

    trend = "Boosting tendance · Top 5 sectoriel"
    window = payload["metric_windows"]["2026-01-01|2026-02-01"]
    trend_values = window[payload["strategy_order"].index(trend)]
    assert set(payload["strategy_order"]) == {spec.label for spec in STRATEGY_SPECS} | {
        "Boosting · Top 5 sectoriel",
        trend,
    }
    composer = payload["portfolio_composer"]
    assert len(composer["combination_masks"]) == 63 and trend in composer["strategy_order"]
    assert len(payload["monthly"]) == 26
    assert payload["contracts"]["report_task"] == "REPORT-019"
    assert trend_values[PERFORMANCE_METRIC_FIELDS.index("maximum_sector_weight")] == 0.2
    assert payload["status"]["publication_eligible"] is False
    assert "Consumer Defensive" in next(
        card["summary"] for card in payload["methodologies"] if "sectoriels" in card["title"]
    )
    pending = payload["current_portfolio"]
    assert pending["status"] == "archived_replay_return_pending"
    assert pending["holding_month"] == "2026-03-01"
    trend_pending = [row for row in pending["holdings"] if row["strategy"] == trend]
    assert len(trend_pending) == len({row["sector"] for row in trend_pending}) == 5
    assert all(row["realized_return"] is None for row in trend_pending)


def test_sector_profile_rejects_common_manifest_from_another_vintage(tmp_path: Path) -> None:
    inputs = _sector_report_inputs(tmp_path)
    common_manifest = inputs.common_replay_dir / "manifest.json"
    _write_json(common_manifest, {**json.loads(common_manifest.read_text()), "vintage": "other"})

    with pytest.raises(ValueError, match="did not use the supplied common replay"):
        build_performance_report_payload(
            inputs, generated_at_utc=datetime(2026, 11, 15, tzinfo=timezone.utc)
        )


def _sector_report_inputs(root: Path) -> PerformanceReportInputs:
    inputs = _report_inputs(root)
    backtest, release = root / "sector_backtest", root / "sector_release"
    backtest.mkdir()
    release.mkdir()
    taxonomy = Path("configs/data_contracts/sec_sic_sector_taxonomy_v1.json")
    (root / taxonomy).parent.mkdir(parents=True)
    copyfile(Path(__file__).resolve().parents[3] / taxonomy, root / taxonomy)
    sector_history = release / "sector_history.parquet"
    pl.DataFrame({"Sector": ["Energy"]}).write_parquet(sector_history)
    sector_digest = sha256_path(sector_history)
    files = {sector_history.name: {"sha256": sector_digest}}
    _write_json(release / "manifest.json", {"release_id": "test-sector-id", "files": files})
    allocation_inputs = {
        "predictions_sha256": "test-predictions",
        "terminal_registry_sha256": "test-terminal-registry",
        "sector_history_sha256": sector_digest,
    }
    file_hashes, manifest_hashes = _seal_common_fixture(inputs, allocation_inputs)
    months = [date(2026, 1, 1), date(2026, 2, 1)]
    monthly_path = backtest / "monthly_returns.parquet"
    _monthly([*REFERENCE_STRATEGIES, SECTOR_NATIVE, SECTOR_TREND], months).with_columns(
        pl.when(pl.col("strategy").is_in([SECTOR_NATIVE, SECTOR_TREND]))
        .then(0.2)
        .otherwise(pl.col("maximum_sector_weight"))
        .alias("maximum_sector_weight")
    ).write_parquet(monthly_path)
    holdings_path = backtest / "sector_holdings.parquet"
    _holdings([SECTOR_NATIVE, SECTOR_TREND], months).with_columns(
        pl.lit("Energy").alias("sector")
    ).write_parquet(holdings_path)
    for strategy, name in ((SECTOR_NATIVE, "native"), (SECTOR_TREND, "trend")):
        allocation_path = _write_allocation(root / name, strategy, allocation_inputs)
        manifest_path = allocation_path.with_name("manifest.json")
        file_hashes[str(allocation_path.relative_to(root))] = sha256_path(allocation_path)
        manifest_hashes[str(manifest_path.relative_to(root))] = sha256_path(manifest_path)
    _write_json(
        backtest / "manifest.json",
        {
            "contract_version": 1,
            "status": "validated_r_and_d",
            "promotion_eligible": False,
            "methodology": {"transaction_cost_bps_times_turnover": 10},
            "inputs": file_hashes,
            "input_manifests": manifest_hashes,
            "files": {path.name: sha256_path(path) for path in (monthly_path, holdings_path)},
        },
    )
    equal = root / "aligned-legacy-equal.parquet"
    _monthly(["Combined_Equal"], months).write_parquet(equal)
    return replace(inputs, sector=SectorReportInputs(backtest, release, root, equal))


def _seal_common_fixture(
    inputs: PerformanceReportInputs, allocation_inputs: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    legacy_manifest = inputs.legacy_run_dir / "data_input_manifest.json"
    _write_json(legacy_manifest, {"snapshot_id": "test-composition"})
    snapshot_copy = inputs.legacy_run_dir / "input_snapshot" / "snapshot_manifest.json"
    snapshot_copy.parent.mkdir()
    copyfile(inputs.snapshot_manifest, snapshot_copy)
    common_dir = inputs.common_replay_dir
    manifest_path = common_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = {
        "legacy_run_manifest": {"sha256": sha256_path(legacy_manifest)},
        "boosting_predictions": {"sha256": allocation_inputs["predictions_sha256"]},
    }
    manifest["terminal_entry_gate"] = {
        "registry_sha256": allocation_inputs["terminal_registry_sha256"]
    }
    manifest["artifacts"] = {
        role: {"sha256": sha256_path(common_dir / filename)}
        for role, filename in (
            ("holdings", "comparison_common_holdings.parquet"),
            ("boosting_live_score_holdings", "boosting_live_score_holdings.parquet"),
        )
    }
    _write_json(manifest_path, manifest)
    monthly_path = common_dir / "comparison_common_monthly.parquet"
    return {str(monthly_path): sha256_path(monthly_path)}, {
        str(manifest_path): sha256_path(manifest_path)
    }


def _write_allocation(root: Path, strategy: str, allocation_inputs: dict[str, str]) -> Path:
    root.mkdir()
    path = root / "sector_champion_holdings.parquet"
    sectors = ["Energy", "Technology", "Healthcare", "Industrials", "Utilities"]
    _holdings([strategy], [date(2026, 3, 1)]).drop("sector", "ticker").join(
        pl.DataFrame({"Sector": sectors, "ticker": [f"T{i}" for i in range(5)]}), how="cross"
    ).with_columns(
        pl.lit("horizon_pending").alias("target_status_1m"),
        pl.lit(None, dtype=pl.Float64).alias("realized_return"),
        pl.lit(0.2).alias("target_weight"),
    ).write_parquet(path)
    _write_json(
        root / "manifest.json",
        {
            "strategy": strategy,
            "policy_id": SECTOR_CHAMPION_POLICY_ID,
            "inputs": allocation_inputs,
        },
    )
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    current_month = date(2026, 3, 1)
    live_strategies = sorted(
        {spec.source_strategy for spec in STRATEGY_SPECS if spec.family.startswith("Boosting")}
    )
    _holdings(live_strategies, [current_month]).write_parquet(
        common_dir / "boosting_live_score_holdings.parquet"
    )
    pl.DataFrame(
        [
            {
                "portfolio_model": strategy,
                "year_month": current_month,
                "ticker": f"{strategy[:3].upper()}-CURRENT",
                "weight_normalized": 1.0,
                "Sector": "Test",
                "n_models": 1,
            }
            for strategy in ("Combined_Frequency", "Combined_Equal")
        ]
    ).write_parquet(legacy_dir / "legacy_detailed_returns_polars.parquet")
    as_of_evidence = tmp_path / "price_return_extension_audit.parquet"
    pl.DataFrame(
        {
            "ticker": ["TEST"],
            "date": ["2026-03-28"],
            "selected_adjusted_close": [100.0],
        }
    ).write_parquet(as_of_evidence)
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
    return PerformanceReportInputs(
        common_dir,
        legacy_dir,
        snapshot_manifest,
        portfolio_as_of_evidence=as_of_evidence,
    )


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
