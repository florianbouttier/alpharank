from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.portfolio.combinations import (
    EQUAL_WEIGHT_ADDITIONAL_COST,
    EQUAL_WEIGHT_REBALANCE_FREQUENCY,
    EqualWeightCombinationGrid,
    equal_weight_strategy_combination_grid,
)
from alpharank.portfolio.comparison import (
    performance_by_start_year,
    subperiod_portfolio_metric_grid,
)
from alpharank.reporting._performance_report_html import render_performance_report_html

REPORT_SCHEMA_VERSION = 3
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


@dataclass(frozen=True)
class PerformanceReportInputs:
    common_replay_dir: Path
    legacy_run_dir: Path
    snapshot_manifest: Path


def build_performance_report_payload(
    inputs: PerformanceReportInputs,
    *,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    """Build a browser payload without recalculating KPI outside the common engine."""

    (
        common_manifest,
        snapshot_manifest,
        common_monthly,
        legacy_monthly,
        common_holdings,
        legacy_holdings,
    ) = _load_report_inputs(inputs)

    monthly = _assemble_monthly(common_monthly, legacy_monthly)
    holdings = _assemble_holdings(
        common_holdings,
        legacy_holdings,
        start_month=monthly["holding_month"].min(),
        end_month=monthly["holding_month"].max(),
    )
    sector_coverage = _sector_coverage(holdings)
    monthly = _mask_unobservable_sector_weights(monthly, sector_coverage)
    strategy_order = [spec.label for spec in STRATEGY_SPECS]
    series = {
        strategy: monthly.filter(pl.col("strategy") == strategy).sort("holding_month")
        for strategy in strategy_order
    }
    metric_windows = subperiod_portfolio_metric_grid(
        series,
        benchmark_strategy=BENCHMARK_STRATEGY,
        strategy_order=strategy_order,
        metric_fields=PERFORMANCE_METRIC_FIELDS,
        calendar_year_boundaries_only=True,
    )
    composer = equal_weight_strategy_combination_grid(
        series,
        benchmark_strategy=BENCHMARK_STRATEGY,
        strategy_order=[strategy for strategy in strategy_order if strategy != BENCHMARK_STRATEGY],
        metric_fields=COMPOSER_METRIC_FIELDS,
    )
    start_month, end_month = _report_bounds(monthly)
    start_year_metrics = performance_by_start_year(
        series,
        benchmark_strategy=BENCHMARK_STRATEGY,
        strategy_order=strategy_order,
        first_year=start_month.year,
        end_month=end_month,
    )
    source_paths = _source_paths(inputs)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc.isoformat().replace("+00:00", "Z"),
        "title": "AlphaRank · Rapport de backtest complet",
        "status": _report_status(common_manifest),
        "calendar": _report_calendar(monthly, metric_windows, start_month, end_month),
        "strategy_order": strategy_order,
        "strategies": [_strategy_payload(spec) for spec in STRATEGY_SPECS],
        "metric_fields": list(PERFORMANCE_METRIC_FIELDS),
        "metric_windows": metric_windows,
        "portfolio_composer": _composer_payload(composer),
        "start_year_metrics": _start_year_rows(start_year_metrics),
        "monthly": _monthly_rows(monthly),
        "holdings": _holding_rows(holdings),
        "data_quality": {"sector_coverage_by_strategy": sector_coverage},
        "methodologies": _methodology_cards(),
        "contracts": {
            "timing": common_manifest.get("timing_contract"),
            "transaction_cost": common_manifest.get("transaction_cost_policy"),
            "benchmark": "SPY total return depuis adjusted_close",
            "missing_return": "Sélection avant rendement réalisé ; absence sélectionnée = arrêt.",
            "kpi_engine": "alpharank.portfolio.performance.portfolio_period_statistics",
            "report_task": "REPORT-008",
        },
        "lineage": _report_lineage(common_manifest, snapshot_manifest, source_paths),
    }


def _report_bounds(monthly: pl.DataFrame) -> tuple[date, date]:
    start_month = monthly["holding_month"].min()
    end_month = monthly["holding_month"].max()
    if not isinstance(start_month, date) or not isinstance(end_month, date):
        raise ValueError("The report calendar must contain at least one valid date.")
    return start_month, end_month


def _load_report_inputs(
    inputs: PerformanceReportInputs,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    return (
        _load_json(inputs.common_replay_dir / "manifest.json"),
        _load_json(inputs.snapshot_manifest),
        pl.read_parquet(inputs.common_replay_dir / "comparison_common_monthly.parquet"),
        pl.read_parquet(inputs.legacy_run_dir / "legacy_common_monthly.parquet"),
        pl.read_parquet(inputs.common_replay_dir / "comparison_common_holdings.parquet"),
        pl.read_parquet(inputs.legacy_run_dir / "legacy_common_holdings.parquet"),
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
            {"role": role, "path": str(path), "sha256": _sha256(path)}
            for role, path in source_paths.items()
        ],
    }


def write_performance_report(
    inputs: PerformanceReportInputs,
    *,
    output_dir: Path,
    generated_at_utc: datetime,
) -> dict[str, Path]:
    """Write the self-contained report and its compact lineage manifest."""

    payload = build_performance_report_payload(inputs, generated_at_utc=generated_at_utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backtest_performance_report.html"
    report_path.write_text(render_performance_report_html(payload), encoding="utf-8")
    manifest_path = output_dir / "backtest_performance_report_manifest.json"
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": payload["generated_at_utc"],
        "report": {
            "path": str(report_path.resolve()),
            "sha256": _sha256(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "calendar": payload["calendar"],
        "strategies": payload["strategy_order"],
        "metric_fields": payload["metric_fields"],
        "portfolio_composer": {
            "method": payload["portfolio_composer"]["policy"]["method"],
            "combinations": len(payload["portfolio_composer"]["combination_masks"]),
            "metric_fields": payload["portfolio_composer"]["metric_fields"],
            "correlation_windows": len(
                payload["portfolio_composer"]["strategy_correlation_windows"]
            ),
        },
        "lineage": payload["lineage"],
        "status": payload["status"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"report": report_path, "manifest": manifest_path}


def _assemble_monthly(common: pl.DataFrame, legacy: pl.DataFrame) -> pl.DataFrame:
    start = common["holding_month"].min()
    end = common["holding_month"].max()
    parts: list[pl.DataFrame] = []
    for spec in STRATEGY_SPECS:
        source = common if spec.source == "common" else legacy
        part = source.filter(
            (pl.col("strategy") == spec.source_strategy)
            & pl.col("holding_month").is_between(start, end)
        ).with_columns(pl.lit(spec.label).alias("strategy"))
        if part.height != common["holding_month"].n_unique():
            raise ValueError(f"Strategy {spec.label!r} does not cover the common calendar")
        parts.append(part)
    return pl.concat(parts, how="diagonal_relaxed").sort(["holding_month", "strategy"])


def _assemble_holdings(
    common: pl.DataFrame,
    legacy: pl.DataFrame,
    *,
    start_month: Any,
    end_month: Any,
) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for spec in STRATEGY_SPECS:
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


def _methodology_cards() -> list[dict[str, Any]]:
    return [
        {
            "title": "Legacy · Frequency",
            "status": "Méthode de production rejouée sur un snapshot candidat",
            "summary": "Trend-following relatif, quatre pistes walk-forward puis votes pondérés.",
            "pseudo_code": [
                "calculer les EMA relatives à SPY avec les prix connus à t",
                "optimiser chaque piste uniquement sur les mois antérieurs",
                "appliquer univers historique, liquidité et plafond sectoriel",
                "agréger les votes puis détenir le panier en t+1",
            ],
        },
        {
            "title": "Legacy · Equal",
            "status": "Agrégation historique",
            "summary": "Même recherche Legacy, mais moyenne égale des quatre trajectoires.",
            "pseudo_code": [
                "produire les quatre portefeuilles Legacy causaux",
                "donner le même poids à chaque trajectoire",
                "agréger les poids par titre",
                "simuler avec le même calendrier et les mêmes frais",
            ],
        },
        {
            "title": "Boosting natif",
            "status": "R&D comparable, non promu",
            "summary": "XGBoost classe les titres avec les EMA relatives causales de chaque fold.",
            "pseudo_code": [
                "geler les paires EMA disponibles au cutoff du fold",
                "entraîner, valider et calibrer sans toucher au test",
                "scorer toutes les lignes OOS du mois t",
                "prendre Top N avant de joindre le rendement de t+1",
            ],
        },
        {
            "title": "Boosting filtré par tendance",
            "status": "Diagnostic post-hoc, non promu",
            "summary": "Le score ne change pas ; une majorité stricte d’EMA positives borne l’univers.",
            "pseudo_code": [
                "reconstruire les ratios EMA causaux du fold",
                "exiger toutes les paires et une majorité strictement positive",
                "conserver le score Boosting natif inchangé",
                "classer puis prendre Top N dans l’univers filtré",
            ],
        },
        {
            "title": "SPY total return",
            "status": "Benchmark commun",
            "summary": "Benchmark dividendes réinvestis depuis adjusted_close, sans frais simulés.",
            "pseudo_code": [
                "résoudre SPY sur le même calendrier mensuel",
                "calculer le rendement total adjusted_close",
                "exclure le mois courant incomplet",
                "apparier chaque mois aux stratégies avant les KPI",
            ],
        },
        {
            "title": "Portefeuille composé",
            "status": "Laboratoire post-hoc, non promu",
            "summary": ("Équipondération mensuelle des stratégies cochées, comparée au SPY."),
            "pseudo_code": [
                "choisir au moins une poche de stratégie",
                "attribuer 1 / N du capital à chaque poche chaque mois",
                "moyenner leurs rendements nets sans coût inter-poche ajouté",
                "lire KPI, corrélations et richesse relative pré-calculés face au SPY",
            ],
        },
    ]


def _source_paths(inputs: PerformanceReportInputs) -> dict[str, Path]:
    return {
        "common_manifest": inputs.common_replay_dir / "manifest.json",
        "common_monthly": inputs.common_replay_dir / "comparison_common_monthly.parquet",
        "common_holdings": inputs.common_replay_dir / "comparison_common_holdings.parquet",
        "legacy_monthly": inputs.legacy_run_dir / "legacy_common_monthly.parquet",
        "legacy_holdings": inputs.legacy_run_dir / "legacy_common_holdings.parquet",
        "snapshot_manifest": inputs.snapshot_manifest,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
