from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio.combinations import (
    EQUAL_WEIGHT_ADDITIONAL_COST,
    EQUAL_WEIGHT_REBALANCE_FREQUENCY,
    EqualWeightCombinationGrid,
)
from alpharank.portfolio.comparison import (
    performance_by_start_year,
)
from alpharank.replay.sector_backtest import SECTOR_NATIVE, SECTOR_TREND
from alpharank.reporting._performance_report_analytics import (
    build_report_analytics,
    strategy_frontier_payload,
)
from alpharank.reporting._performance_report_methods import methodology_cards
from alpharank.reporting._performance_report_sector import (
    SectorReportInputs,
    load_sector_report_data,
)
from alpharank.reporting.strategy_analysis.report import (
    write_daily_analysis_report,
    write_monthly_archive_report,
)

REPORT_SCHEMA_VERSION = 5
BENCHMARK_STRATEGY = "SPY · Total return"
COMPOSER_METRIC_FIELDS = (
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "sortino",
    "correlation",
)
PERFORMANCE_METRIC_FIELDS = (
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "positive_month_rate",
    "sortino",
    "calmar",
    "annualized_excess_return",
    "tracking_error",
    "information_ratio",
    "beta",
    "alpha",
    "correlation",
    "benchmark_hit_rate",
    "var_95",
    "cvar_95",
    "omega",
    "up_capture",
    "down_capture",
    "skewness",
    "excess_kurtosis",
    "average_monthly_turnover",
    "annualized_turnover",
    "total_transaction_cost",
    "annualized_transaction_cost",
    "average_positions",
    "minimum_positions",
    "maximum_positions",
    "average_maximum_position_weight",
    "maximum_single_name_weight",
    "average_maximum_sector_weight",
    "maximum_sector_weight",
)


@dataclass(frozen=True)
class StrategySpec:
    label: str
    source: str
    source_strategy: str
    family: str
    status: str
    color: str


STRATEGY_SPECS = (
    StrategySpec(
        "Legacy · Frequency",
        "common",
        "Legacy",
        "Legacy",
        "production_method_candidate_replay",
        "#111d55",
    ),
    StrategySpec(
        "Legacy · Equal",
        "legacy",
        "Combined_Equal",
        "Legacy",
        "historical_aggregation",
        "#4d5f99",
    ),
    StrategySpec("Boosting · Top 5", "common", "Boosting Top 5", "Boosting natif", "rd", "#265511"),
    StrategySpec(
        "Boosting · Top 10", "common", "Boosting Top 10", "Boosting natif", "rd", "#3f751f"
    ),
    StrategySpec(
        "Boosting · Top 15", "common", "Boosting Top 15", "Boosting natif", "rd", "#619c35"
    ),
    StrategySpec(
        "Boosting · Top 20", "common", "Boosting Top 20", "Boosting natif", "rd", "#8bc45b"
    ),
    StrategySpec(
        "Boosting tendance · Top 5",
        "common",
        "Boosting Top 5 | Causal trend",
        "Boosting tendance",
        "post_hoc_rd",
        "#1f6a7a",
    ),
    StrategySpec(
        "Boosting tendance · Top 10",
        "common",
        "Boosting Top 10 | Causal trend",
        "Boosting tendance",
        "post_hoc_rd",
        "#2b8396",
    ),
    StrategySpec(
        "Boosting tendance · Top 15",
        "common",
        "Boosting Top 15 | Causal trend",
        "Boosting tendance",
        "post_hoc_rd",
        "#46a4b7",
    ),
    StrategySpec(
        "Boosting tendance · Top 20",
        "common",
        "Boosting Top 20 | Causal trend",
        "Boosting tendance",
        "post_hoc_rd",
        "#72c2d0",
    ),
    StrategySpec(
        BENCHMARK_STRATEGY,
        "common",
        "SPY total return",
        "Benchmark",
        "benchmark",
        "#9b8816",
    ),
)

SECTOR_STRATEGY_SPECS = (
    *STRATEGY_SPECS[:-1],
    StrategySpec(
        "Boosting · Top 5 sectoriel",
        "common",
        SECTOR_NATIVE,
        "Boosting natif",
        "post_hoc_rd",
        "#a65a29",
    ),
    StrategySpec(
        "Boosting tendance · Top 5 sectoriel",
        "common",
        SECTOR_TREND,
        "Boosting tendance",
        "post_hoc_rd",
        "#7752a4",
    ),
    STRATEGY_SPECS[-1],
)


@dataclass(frozen=True)
class PerformanceReportInputs:
    common_replay_dir: Path
    legacy_run_dir: Path
    snapshot_manifest: Path
    portfolio_as_of_evidence: Path
    sector: SectorReportInputs | None = None


def build_performance_report_payload(
    inputs: PerformanceReportInputs,
    *,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    """Build a browser payload without recalculating KPI outside the common engine."""

    source_paths = _source_paths(inputs)
    (
        common_manifest,
        snapshot_manifest,
        common_monthly,
        legacy_monthly,
        common_holdings,
        legacy_holdings,
    ) = _load_report_inputs(source_paths)

    sector = None
    specs = STRATEGY_SPECS
    if inputs.sector is not None:
        sector = load_sector_report_data(
            inputs.sector,
            common_dir=inputs.common_replay_dir,
            common_manifest=common_manifest,
            reference_monthly=common_monthly,
        )
        legacy_manifest = inputs.legacy_run_dir / "data_input_manifest.json"
        if (
            sha256_path(legacy_manifest)
            != common_manifest["sources"]["legacy_run_manifest"]["sha256"]
        ):
            raise ValueError("Legacy run does not match the sealed common replay.")
        if sha256_path(inputs.snapshot_manifest) != sha256_path(
            inputs.legacy_run_dir / "input_snapshot/snapshot_manifest.json"
        ):
            raise ValueError("Snapshot manifest does not match the sealed Legacy run.")
        source_paths.update(sector.source_paths, legacy_run_manifest=legacy_manifest)
        common_monthly = pl.concat([common_monthly, sector.monthly], how="diagonal_relaxed")
        common_holdings = pl.concat([common_holdings, sector.holdings], how="diagonal_relaxed")
        specs = SECTOR_STRATEGY_SPECS

    monthly = _assemble_monthly(common_monthly, legacy_monthly, specs=specs)
    holdings = _assemble_holdings(
        common_holdings,
        legacy_holdings,
        start_month=monthly["holding_month"].min(),
        end_month=monthly["holding_month"].max(),
        specs=specs,
    )
    current_portfolio = _current_portfolio_payload(
        inputs, specs=specs, sector_pending=sector.pending if sector is not None else None
    )
    contracts = {
        "timing": common_manifest.get("timing_contract"),
        "transaction_cost": common_manifest.get("transaction_cost_policy"),
        "benchmark": "SPY total return depuis adjusted_close",
        "missing_return": "Sélection avant rendement réalisé ; absence sélectionnée = arrêt.",
        "kpi_engine": "alpharank.portfolio.performance.portfolio_period_statistics",
        "report_task": "REPORT-014",
    }
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc.isoformat().replace("+00:00", "Z"),
        "title": "AlphaRank · Rapport de backtest complet",
        "status": _report_status(common_manifest),
        "current_portfolio": current_portfolio,
        "methodologies": methodology_cards(sector),
        "contracts": contracts,
        "lineage": _report_lineage(common_manifest, snapshot_manifest, source_paths),
    }
    if sector is not None:
        _set_sector_context(payload, sector.release_id, sector.taxonomy_id)
    return payload | _report_analysis(monthly, holdings, specs=specs)


def _report_analysis(
    monthly: pl.DataFrame, holdings: pl.DataFrame, *, specs: tuple[StrategySpec, ...]
) -> dict[str, Any]:
    sector_coverage = _sector_coverage(holdings)
    if specs is SECTOR_STRATEGY_SPECS:
        sector_coverage[BENCHMARK_STRATEGY] = 0.0  # ETF constituent sectors are not observed.
    monthly = _mask_unobservable_sector_weights(monthly, sector_coverage)
    strategy_order = [spec.label for spec in specs]
    series, metric_windows, composer, frontiers = build_report_analytics(
        monthly,
        strategy_order=strategy_order,
        benchmark_strategy=BENCHMARK_STRATEGY,
        performance_metric_fields=PERFORMANCE_METRIC_FIELDS,
        composer_metric_fields=COMPOSER_METRIC_FIELDS,
    )
    start_month, end_month = _report_bounds(monthly)
    start_year_metrics = performance_by_start_year(
        series,
        benchmark_strategy=BENCHMARK_STRATEGY,
        strategy_order=strategy_order,
        first_year=start_month.year,
        end_month=end_month,
    )
    return {
        "calendar": _report_calendar(monthly, metric_windows, start_month, end_month),
        "strategy_order": strategy_order,
        "strategies": [_strategy_payload(spec) for spec in specs],
        "metric_fields": list(PERFORMANCE_METRIC_FIELDS),
        "metric_windows": metric_windows,
        "portfolio_composer": _composer_payload(composer),
        "strategy_frontiers": strategy_frontier_payload(frontiers),
        "start_year_metrics": _start_year_rows(start_year_metrics),
        "monthly": _monthly_rows(monthly),
        "holdings": _holding_rows(holdings),
        "data_quality": {"sector_coverage_by_strategy": sector_coverage},
    }


def _set_sector_context(payload: dict[str, Any], release_id: str, taxonomy_id: str) -> None:
    payload["title"] = "AlphaRank · Toutes les stratégies face au SPY"
    payload["headline"] = "Toutes les stratégies face au SPY"
    payload["subtitle"] = (
        "Legacy, Boosting natif et tendance, avec leurs variantes Top 5 sectorielles. "
        "« Tout » commence au premier mois commun, pas en 2010."
    )
    payload["default_curves"] = [spec.label for spec in SECTOR_STRATEGY_SPECS]
    payload["status"]["publication_eligible"] = False
    payload["status"]["message"] = (
        "Replay commun et REPLAY-014 · R&D, aucun nouveau signal. "
        "Legacy Equal utilise le même calendrier et les mêmes frais."
    )
    contracts = payload["contracts"]
    contracts.update(
        report_task="REPORT-019", sector_release=release_id, sector_taxonomy=taxonomy_id
    )
    contracts["transaction_cost"] = dict(
        contracts["transaction_cost"],
        strategies=[spec.source_strategy for spec in SECTOR_STRATEGY_SPECS[:-1]],
    )


def _report_bounds(monthly: pl.DataFrame) -> tuple[date, date]:
    start_month = monthly["holding_month"].min()
    end_month = monthly["holding_month"].max()
    if not isinstance(start_month, date) or not isinstance(end_month, date):
        raise ValueError("The report calendar must contain at least one valid date.")
    return start_month, end_month


def _load_report_inputs(
    paths: dict[str, Path],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    return (
        _load_json(paths["common_manifest"]),
        _load_json(paths["snapshot_manifest"]),
        pl.read_parquet(paths["common_monthly"]),
        pl.read_parquet(paths["legacy_monthly"]),
        pl.read_parquet(paths["common_holdings"]),
        pl.read_parquet(paths["legacy_holdings"]),
    )


def _report_status(common_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison_eligible": common_manifest.get("comparison_eligible"),
        "publication_eligible": common_manifest.get("publication_eligible"),
        "methodology_status": common_manifest.get("methodology_status"),
        "message": (
            "Comparaison mécanique valide sur un snapshot candidat ; "
            "les variantes Boosting ne sont pas promues. Les secteurs ne "
            "sont pas renseignés dans ces holdings : leurs KPI sont indisponibles."
        ),
    }


def _report_calendar(
    monthly: pl.DataFrame,
    metric_windows: dict[str, list[list[float | None]]],
    start_month: Any,
    end_month: Any,
) -> dict[str, Any]:
    return {
        "months": monthly["holding_month"].n_unique(),
        "start": start_month.isoformat(),
        "end": end_month.isoformat(),
        "available_months": [
            value.isoformat() for value in monthly["holding_month"].unique().sort()
        ],
        "available_start_months": _window_boundaries(metric_windows, side="start"),
        "available_end_months": _window_boundaries(metric_windows, side="end"),
    }


def _report_lineage(
    common_manifest: dict[str, Any],
    snapshot_manifest: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "composition_id": snapshot_manifest.get("composition_id"),
        "snapshot_dir": snapshot_manifest.get("snapshot_dir"),
        "replay_git_commit": common_manifest.get("runtime_provenance", {})
        .get("git", {})
        .get("head"),
        "common_replay_status": common_manifest.get("status"),
        "common_replay_profile": common_manifest.get("comparison_profile"),
        "source_files": [
            {"role": role, "path": str(path), "sha256": sha256_path(path)}
            for role, path in source_paths.items()
        ],
    }


def write_performance_report(
    inputs: PerformanceReportInputs,
    *,
    output_dir: Path,
    generated_at_utc: datetime,
    analysis_frequency: Literal["daily", "monthly_archive"] = "monthly_archive",
) -> dict[str, Path]:
    """Write the self-contained report and its compact lineage manifest."""

    payload = build_performance_report_payload(inputs, generated_at_utc=generated_at_utc)
    if analysis_frequency not in {"daily", "monthly_archive"}:
        raise ValueError("Unsupported analysis frequency.")
    if analysis_frequency == "daily":
        return write_daily_analysis_report(
            payload, output_dir=output_dir, generated_at_utc=generated_at_utc
        )
    return write_monthly_archive_report(payload, output_dir=output_dir)


def _assemble_monthly(
    common: pl.DataFrame, legacy: pl.DataFrame, *, specs: tuple[StrategySpec, ...]
) -> pl.DataFrame:
    months = common["holding_month"].unique().sort()
    parts: list[pl.DataFrame] = []
    for spec in specs:
        source = common if spec.source == "common" else legacy
        part = source.filter(
            (pl.col("strategy") == spec.source_strategy)
            & pl.col("holding_month").is_between(months.min(), months.max())
        ).with_columns(pl.lit(spec.label).alias("strategy"))
        if not part["holding_month"].sort().equals(months):
            raise ValueError(f"Strategy {spec.label!r} does not cover the common calendar")
        parts.append(part)
    return pl.concat(parts, how="diagonal_relaxed").sort(["holding_month", "strategy"])


def _assemble_holdings(
    common: pl.DataFrame,
    legacy: pl.DataFrame,
    *,
    start_month: Any,
    end_month: Any,
    specs: tuple[StrategySpec, ...],
) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for spec in specs:
        if spec.label == BENCHMARK_STRATEGY:
            continue
        source = common if spec.source == "common" else legacy
        part = source.filter(
            (pl.col("strategy") == spec.source_strategy)
            & pl.col("holding_month").is_between(start_month, end_month)
        ).with_columns(pl.lit(spec.label).alias("strategy"))
        parts.append(part)
    return pl.concat(parts, how="diagonal_relaxed").sort(
        ["holding_month", "strategy", "target_weight", "ticker"],
        descending=[False, False, True, False],
    )


def _current_portfolio_payload(
    inputs: PerformanceReportInputs,
    *,
    specs: tuple[StrategySpec, ...],
    sector_pending: pl.DataFrame | None,
) -> dict[str, Any]:
    live = pl.read_parquet(inputs.common_replay_dir / "boosting_live_score_holdings.parquet")
    if sector_pending is not None:
        live = pl.concat([live, sector_pending], how="diagonal_relaxed")
    legacy = pl.read_parquet(inputs.legacy_run_dir / "legacy_detailed_returns_polars.parquet")
    holdings = pl.concat(
        [
            _current_boosting_holdings(live, specs=specs),
            _current_legacy_holdings(legacy, specs=specs),
        ],
        how="diagonal_relaxed",
    ).sort(
        ["strategy", "target_weight", "ticker"],
        descending=[False, True, False],
    )
    expected = {spec.label for spec in specs if spec.label != BENCHMARK_STRATEGY}
    observed = set(holdings["strategy"].unique().to_list())
    if observed != expected:
        raise ValueError(f"Current portfolio strategies differ: {sorted(expected - observed)}")
    decision_month = _one_date(holdings, "decision_month")
    holding_month = _one_date(holdings, "holding_month")
    as_of_date = _market_evidence_date(inputs.portfolio_as_of_evidence)
    if not _date_is_in_month(as_of_date, holding_month):
        raise ValueError("Current portfolio market evidence is outside its holding month")
    return {
        "as_of_date": as_of_date.isoformat(),
        "decision_month": decision_month.isoformat(),
        "holding_month": holding_month.isoformat(),
        "status": (
            "archived_replay_return_pending"
            if sector_pending is not None
            else "in_force_return_pending"
        ),
        "holdings": _holding_rows(holdings),
    }


def _current_boosting_holdings(
    live: pl.DataFrame, *, specs: tuple[StrategySpec, ...]
) -> pl.DataFrame:
    required = {"strategy", "decision_month", "holding_month", "ticker", "target_weight"}
    missing = sorted(required - set(live.columns))
    if missing:
        raise ValueError(f"Boosting live holdings are missing: {missing}")
    parts: list[pl.DataFrame] = []
    for spec in specs:
        if not spec.family.startswith("Boosting"):
            continue
        part = live.filter(pl.col("strategy") == spec.source_strategy)
        if part.is_empty():
            raise ValueError(f"Current Boosting portfolio is missing {spec.source_strategy}")
        parts.append(_select_current_holding_fields(part, strategy=spec.label))
    return pl.concat(parts, how="diagonal_relaxed")


def _current_legacy_holdings(
    detailed: pl.DataFrame, *, specs: tuple[StrategySpec, ...]
) -> pl.DataFrame:
    required = {"portfolio_model", "year_month", "ticker", "weight_normalized"}
    missing = sorted(required - set(detailed.columns))
    if missing:
        raise ValueError(f"Legacy current holdings are missing: {missing}")
    current_month = detailed["year_month"].cast(pl.Date, strict=False).max()
    parts: list[pl.DataFrame] = []
    for spec in specs:
        if spec.family != "Legacy":
            continue
        source = "Combined_Frequency" if spec.source == "common" else spec.source_strategy
        part = detailed.filter(
            (pl.col("portfolio_model") == source)
            & (pl.col("year_month").cast(pl.Date, strict=False) == current_month)
        ).with_columns(
            pl.col("year_month").cast(pl.Date, strict=False).alias("holding_month"),
            pl.col("weight_normalized").cast(pl.Float64).alias("target_weight"),
        )
        if part.is_empty():
            raise ValueError(f"Current Legacy portfolio is missing {source}")
        part = part.with_columns(
            pl.col("holding_month").dt.offset_by("-1mo").alias("decision_month")
        )
        parts.append(_select_current_holding_fields(part, strategy=spec.label))
    return pl.concat(parts, how="diagonal_relaxed")


def _select_current_holding_fields(frame: pl.DataFrame, *, strategy: str) -> pl.DataFrame:
    return frame.select(
        pl.lit(strategy).alias("strategy"),
        pl.col("decision_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.String),
        pl.col("target_weight").cast(pl.Float64),
        pl.lit(None, dtype=pl.Float64).alias("realized_return"),
        _optional_holding_column(frame, "selection_rank", pl.Int64),
        _optional_holding_column(frame, "score", pl.Float64),
        _optional_holding_column(frame, "sector", pl.String, fallback="Sector"),
        _optional_holding_column(frame, "n_models", pl.Int64),
    )


def _optional_holding_column(
    frame: pl.DataFrame,
    name: str,
    dtype: pl.DataType | type[pl.DataType],
    *,
    fallback: str | None = None,
) -> pl.Expr:
    source = name if name in frame.columns else fallback
    if source is not None and source in frame.columns:
        return pl.col(source).cast(dtype, strict=False).alias(name)
    return pl.lit(None, dtype=dtype).alias(name)


def _one_date(frame: pl.DataFrame, column: str) -> date:
    values = frame[column].drop_nulls().unique().to_list()
    if len(values) != 1 or not isinstance(values[0], date):
        raise ValueError(f"Current portfolio needs one {column}, found {values}")
    return values[0]


def _market_evidence_date(path: Path) -> date:
    evidence = pl.read_parquet(path)
    if "date" not in evidence.columns:
        raise ValueError("Portfolio as-of evidence needs a date column")
    date_expression = (
        pl.col("date").str.to_date(strict=False)
        if evidence.schema["date"] == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    value = evidence.select(date_expression.max()).item()
    if not isinstance(value, date):
        raise ValueError("Portfolio as-of evidence contains no valid market date")
    return value


def _date_is_in_month(value: date, month: date) -> bool:
    return value.year == month.year and value.month == month.month


def _sector_coverage(holdings: pl.DataFrame) -> dict[str, float]:
    return {
        str(row["strategy"]): float(row["sector_rows"] / row["rows"])
        for row in holdings.group_by("strategy")
        .agg(
            pl.len().alias("rows"),
            pl.col("sector").is_not_null().sum().alias("sector_rows"),
        )
        .iter_rows(named=True)
    }


def _mask_unobservable_sector_weights(
    monthly: pl.DataFrame,
    coverage: dict[str, float],
) -> pl.DataFrame:
    incomplete = [strategy for strategy, ratio in coverage.items() if ratio < 1.0]
    if not incomplete:
        return monthly
    return monthly.with_columns(
        pl.when(pl.col("strategy").is_in(incomplete))
        .then(None)
        .otherwise(pl.col("maximum_sector_weight"))
        .cast(pl.Float64)
        .alias("maximum_sector_weight")
    )


def _monthly_rows(monthly: pl.DataFrame) -> list[dict[str, Any]]:
    fields = (
        "strategy",
        "decision_month",
        "holding_month",
        "net_return",
        "benchmark_return",
        "turnover",
        "transaction_cost",
        "n_positions",
        "maximum_position_weight",
        "maximum_sector_weight",
    )
    return [_json_row(row) for row in monthly.select(fields).iter_rows(named=True)]


def _composer_payload(grid: EqualWeightCombinationGrid) -> dict[str, Any]:
    return {
        "strategy_order": list(grid.strategy_order),
        "combination_masks": list(grid.combination_masks),
        "months": [month.isoformat() for month in grid.months],
        "monthly_returns": grid.monthly_returns.tolist(),
        "metric_fields": list(grid.metric_fields),
        "metric_windows": grid.metric_windows,
        "strategy_correlation_windows": grid.strategy_correlation_windows,
        "policy": {
            "method": "monthly_equal_weight_strategy_sleeves",
            "rebalance_frequency": EQUAL_WEIGHT_REBALANCE_FREQUENCY,
            "input_returns": "net_return après les frais propres à chaque stratégie",
            "additional_inter_sleeve_cost": EQUAL_WEIGHT_ADDITIONAL_COST,
            "correlation": "Pearson sur les rendements mensuels de la fenêtre",
            "relative_wealth": "richesse composée divisée par la richesse SPY",
            "status": "diagnostic post-hoc non promu",
        },
    }


def _holding_rows(holdings: pl.DataFrame) -> list[dict[str, Any]]:
    optional = [
        column
        for column in ("selection_rank", "score", "sector", "n_models")
        if column in holdings.columns
    ]
    fields = [
        "strategy",
        "decision_month",
        "holding_month",
        "ticker",
        "target_weight",
        "realized_return",
        *optional,
    ]
    return [_json_row(row) for row in holdings.select(fields).iter_rows(named=True)]


def _start_year_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    fields = (
        "requested_start_year",
        "strategy",
        "effective_start_month",
        "end_month",
        "months",
        "coverage",
        "cagr",
        "annualized_volatility",
        "max_drawdown",
    )
    return [_json_row(row) for row in frame.select(fields).iter_rows(named=True)]


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in row.items()}


def _json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value


def _strategy_payload(spec: StrategySpec) -> dict[str, str]:
    return {
        "label": spec.label,
        "family": spec.family,
        "status": spec.status,
        "color": spec.color,
    }


def _window_boundaries(
    windows: dict[str, list[list[float | None]]],
    *,
    side: str,
) -> list[str]:
    index = 0 if side == "start" else 1
    return sorted({key.split("|")[index] for key in windows})


def _source_paths(inputs: PerformanceReportInputs) -> dict[str, Path]:
    return {
        "common_manifest": inputs.common_replay_dir / "manifest.json",
        "common_monthly": inputs.common_replay_dir / "comparison_common_monthly.parquet",
        "common_holdings": inputs.common_replay_dir / "comparison_common_holdings.parquet",
        "boosting_live_holdings": (
            inputs.common_replay_dir / "boosting_live_score_holdings.parquet"
        ),
        "legacy_monthly": (
            inputs.sector.legacy_equal_monthly
            if inputs.sector is not None
            else inputs.legacy_run_dir / "legacy_common_monthly.parquet"
        ),
        "legacy_holdings": inputs.legacy_run_dir / "legacy_common_holdings.parquet",
        "legacy_detailed_holdings": (
            inputs.legacy_run_dir / "legacy_detailed_returns_polars.parquet"
        ),
        "portfolio_as_of_evidence": inputs.portfolio_as_of_evidence,
        "snapshot_manifest": inputs.snapshot_manifest,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return payload
