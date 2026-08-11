#!/usr/bin/env python3
"""Build the AlphaRank research and production dashboard."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.portfolio.performance import advanced_performance_statistics
from alpharank.portfolio.attribution import (
    portfolio_return_attribution,
    reference_return_attribution,
)
from alpharank.portfolio.lineage import (
    input_hashes_from_manifest,
    load_manifest,
    require_matching_data_contexts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPN_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/"
    "legacy_ema_top5_vs_top10_quarantine_v7_20260726"
)
CHAMPION_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/"
    "legacy_ema_long_history_ticker_quarantine_v6_20260726"
)
DIAGNOSTICS_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_ema_fold_full_shap_20260810"
)
RISK_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/"
    "legacy_ema_risk_overlay_ticker_quarantine_v6_20260726"
)
SCREENING_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/screening_clean_20260725"
)
EMA_SCREENING_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_winners_pit_ema_only_20260725"
)
LIVE_DIR = PROJECT_ROOT / (
    "outputs/live_alpha/"
    "ema_classification_h6_202606_20260727_production_candidate_v3"
)
HISTORICAL_LEGACY_COMMON_DIR = PROJECT_ROOT / (
    "outputs/common_portfolio_replays/"
    "legacy_20260713_201639_spy_total_return"
)
CURRENT_LEGACY_COMMON_DIR = PROJECT_ROOT / (
    "outputs/common_portfolio_replays/"
    "legacy_20260727_221253_spy_total_return"
)
LATEST_OPEN_SOURCE_MANIFEST = (
    PROJECT_ROOT / "data/open_source/official/manifests/latest_run.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/research_dashboard/"
    "legacy_ema_alpha_central_20260810_full_shap"
)
SHAP_OVERVIEW_ROWS_PER_FOLD = 80


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    return _clean(pl.read_csv(path, try_parse_dates=True).to_dicts())


def _parquet_records(path: Path) -> list[dict[str, Any]]:
    return _clean(pl.read_parquet(path).to_dicts())


def _full_precision_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Serialize audit-critical returns without per-row decimal truncation."""

    return [
        {
            key: value.isoformat() if isinstance(value, (date, datetime)) else value
            for key, value in row.items()
        }
        for row in frame.to_dicts()
    ]


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _maximum_date(path: Path, column: str) -> str:
    value = pl.scan_parquet(path).select(pl.col(column).max()).collect().item()
    return str(value)


def _historical_lineage(monthly: pl.DataFrame) -> dict[str, Any]:
    champion_manifest_path = CHAMPION_DIR / "manifest.json"
    risk_manifest_path = RISK_DIR / "manifest.json"
    topn_manifest_path = TOPN_DIR / "manifest.json"
    report_manifest_path = TOPN_DIR / "alpha_shap_portfolio_manifest.json"
    champion = load_manifest(champion_manifest_path)
    risk = load_manifest(risk_manifest_path)
    topn = load_manifest(topn_manifest_path)
    report = load_manifest(report_manifest_path)
    champion_hashes = input_hashes_from_manifest(champion)
    context_check = require_matching_data_contexts(
        champion_manifest_path,
        risk_manifest_path,
        required_keys=set(champion_hashes),
    )
    if topn["source_risk_manifest_sha256"] != _hash(risk_manifest_path):
        raise ValueError("Top-N artifact does not match its declared risk manifest.")

    legacy_detailed = _project_path(champion["legacy_detailed_returns_path"])
    legacy_monthly = _project_path(champion["legacy_monthly_returns_path"])
    if champion["legacy_detailed_returns_sha256"] != _hash(legacy_detailed):
        raise ValueError("Champion Legacy detailed hash no longer matches its source.")
    if report["legacy_holdings_sha256"] != _hash(legacy_detailed):
        raise ValueError("Dashboard Legacy holdings and champion use different packages.")

    common_manifest = load_manifest(HISTORICAL_LEGACY_COMMON_DIR / "manifest.json")
    common = _canonical_legacy_monthly(HISTORICAL_LEGACY_COMMON_DIR).rename(
        {
            "legacy_return": "source_legacy_return",
            "spy_return": "source_spy_return",
        }
    )
    comparison = monthly.select(
        "holding_month", "legacy_return", "spy_return"
    ).join(
        common,
        on="holding_month",
        how="inner",
    )
    legacy_maximum_error = float(
        (comparison["legacy_return"] - comparison["source_legacy_return"])
        .abs()
        .max()
    )
    spy_maximum_error = float(
        (comparison["spy_return"] - comparison["source_spy_return"])
        .abs()
        .max()
    )
    if (
        comparison.height != monthly.height
        or legacy_maximum_error > 1e-12
        or spy_maximum_error > 1e-12
    ):
        raise ValueError(
            "Dashboard Legacy/SPY series is not an exact canonical replay of "
            "the champion data package."
        )

    data_dir = _project_path(champion["config"]["data_dir"])
    return {
        "label": "Backtest historique",
        "status": "same_snapshot_verified",
        "snapshot_id": data_dir.parent.name,
        "input_snapshot": str(data_dir.relative_to(PROJECT_ROOT)),
        "price_max_date": _maximum_date(data_dir / "US_Finalprice.parquet", "date"),
        "universe_max_month": str(
            pl.read_csv(data_dir / "SP500_Constituents.csv")
            .select(pl.col("Date").str.to_date(strict=False).max())
            .item()
        ),
        "last_test_decision_month": str(
            pl.read_parquet(CHAMPION_DIR / "classification_h06/predictions.parquet")
            ["decision_month"]
            .max()
        ),
        "last_holding_month": str(monthly["holding_month"].max()),
        "legacy_series_rows_verified": comparison.height,
        "legacy_series_maximum_error": legacy_maximum_error,
        "spy_series_maximum_error": spy_maximum_error,
        "benchmark": common_manifest["benchmark"],
        "note": "Legacy et Boosting : snapshot identique ; SPY total return ajusté",
        "matching_input_keys": context_check["matching_keys"],
    }


def _canonical_legacy_monthly(root: Path) -> pl.DataFrame:
    monthly = pl.read_csv(
        root / "legacy_common_total_return_monthly.csv",
        try_parse_dates=True,
    )
    return (
        monthly.filter(
            pl.col("strategy").is_in(
                ["Combined_Frequency", "Combined_Equal", "SPY total return"]
            )
        )
        .pivot(on="strategy", index="holding_month", values="net_return")
        .rename(
            {
                "Combined_Frequency": "legacy_return",
                "Combined_Equal": "legacy_equal_return",
                "SPY total return": "spy_return",
            }
        )
        .sort("holding_month")
    )


def _performance_attribution_payload() -> list[dict[str, Any]]:
    alpha_holdings = (
        pl.read_parquet(TOPN_DIR / "allocation_holdings.parquet")
        .filter(
            pl.col("strategy").is_in(
                ["alpha_top5_equal", "alpha_top10_equal"]
            )
        )
        .with_columns(
            pl.col("portfolio_weight").alias("target_weight"),
            pl.col("future_return_1m").alias("realized_return"),
            pl.col("benchmark_future_return_1m").alias("benchmark_return"),
        )
        .select(
            "strategy",
            "decision_month",
            "holding_month",
            "ticker",
            "target_weight",
            "realized_return",
            "benchmark_return",
            "sector",
        )
    )
    alpha_monthly = (
        pl.read_csv(TOPN_DIR / "allocation_monthly.csv", try_parse_dates=True)
        .filter(
            pl.col("strategy").is_in(
                ["alpha_top5_equal", "alpha_top10_equal"]
            )
        )
        .with_columns(
            (pl.col("net_return") - pl.col("benchmark_return")).alias(
                "active_return"
            ),
            (
                (1.0 + pl.col("net_return"))
                / (1.0 + pl.col("benchmark_return"))
                - 1.0
            ).alias("relative_return"),
        )
    )
    alpha_attribution = portfolio_return_attribution(
        alpha_holdings,
        alpha_monthly,
    )

    legacy_holdings = pl.read_parquet(
        HISTORICAL_LEGACY_COMMON_DIR
        / "legacy_common_total_return_holdings.parquet"
    ).filter(pl.col("strategy") == "Combined_Frequency")
    legacy_monthly = pl.read_csv(
        HISTORICAL_LEGACY_COMMON_DIR
        / "legacy_common_total_return_monthly.csv",
        try_parse_dates=True,
    )
    legacy_attribution = portfolio_return_attribution(
        legacy_holdings,
        legacy_monthly.filter(pl.col("strategy") == "Combined_Frequency"),
    )
    spy_attribution = reference_return_attribution(
        legacy_monthly.filter(pl.col("strategy") == "SPY total return"),
        component="SPY",
    )
    combined = pl.concat(
        [alpha_attribution, legacy_attribution, spy_attribution],
        how="diagonal_relaxed",
    ).filter(
        pl.col("holding_month").is_between(date(2011, 8, 1), date(2025, 11, 1))
    )
    compact = combined.select(
        pl.col("strategy").alias("s"),
        pl.col("holding_month").alias("m"),
        pl.col("component").alias("c"),
        pl.col("component_type").alias("k"),
        pl.col("simple_return_contribution").alias("v"),
        pl.col("log_return_contribution").alias("l"),
        pl.col("effective_weight").alias("w"),
        pl.col("realized_return").alias("r"),
    )
    return _full_precision_records(compact)


def _current_legacy_lineage() -> dict[str, Any]:
    manifest = load_manifest(CURRENT_LEGACY_COMMON_DIR / "manifest.json")
    run_dir = Path(manifest["run_dir"])
    data_dir = run_dir / "input_snapshot"
    return {
        "label": "Legacy autonome validé",
        "status": "canonical_legacy_replay",
        "snapshot_id": manifest["run_id"],
        "input_snapshot": str(data_dir.relative_to(PROJECT_ROOT)),
        "price_max_date": _maximum_date(data_dir / "US_Finalprice.parquet", "date"),
        "last_holding_month": manifest["benchmark"]["completed_through_month"],
        "benchmark": manifest["benchmark"],
        "note": "Dernier replay Legacy validé ; non comparable au Boosting historique",
    }


def _live_lineage() -> dict[str, Any]:
    live_manifest_path = LIVE_DIR / "manifest.json"
    live = load_manifest(live_manifest_path)
    data_dir = _project_path(live["config"]["data_dir"])
    legacy_manifest_path = data_dir.parent / "data_input_manifest.json"
    live_hashes = input_hashes_from_manifest(live)
    context_check = require_matching_data_contexts(
        legacy_manifest_path,
        live_manifest_path,
        required_keys=set(live_hashes),
    )
    legacy_detailed = Path(live["legacy_detailed_returns"]["path"])
    if live["legacy_detailed_returns"]["sha256"] != _hash(legacy_detailed):
        raise ValueError("Live Alpha and live Legacy do not share the declared package.")
    return {
        "label": "Candidat live",
        "status": "same_snapshot_verified",
        "snapshot_id": data_dir.parent.name,
        "input_snapshot": str(data_dir.relative_to(PROJECT_ROOT)),
        "price_max_date": _maximum_date(data_dir / "US_Finalprice.parquet", "date"),
        "decision_month": live["decision_month"],
        "holding_month": live["holding_month"],
        "matching_input_keys": context_check["matching_keys"],
        "note": "Legacy et Boosting live : snapshot identique",
    }


def _latest_data_lineage() -> dict[str, Any]:
    manifest = load_manifest(LATEST_OPEN_SOURCE_MANIFEST)
    snapshot = Path(manifest["official_dir"]).parent / manifest["published_output_snapshot"]
    return {
        "label": "Dernières données disponibles",
        "status": "historized_not_model_rerun",
        "snapshot_id": snapshot.name,
        "input_snapshot": str(snapshot.relative_to(PROJECT_ROOT)),
        "ingestion_run_id": manifest["run_id"],
        "ingested_at": manifest["ingested_at"],
        "price_max_date": _maximum_date(snapshot / "US_Finalprice.parquet", "date"),
        "universe_max_month": str(
            pl.read_csv(snapshot / "SP500_Constituents.csv")
            .select(pl.col("Date").str.to_date(strict=False).max())
            .item()
        ),
        "note": "Historisé, mais aucun backtest commun n'est encore publié",
    }


def _dashboard_monthly_source() -> Path:
    common = TOPN_DIR / "portfolio_common_monthly.csv"
    return common if common.exists() else TOPN_DIR / "monthly_portfolio_returns.csv"


def _dashboard_monthly() -> pl.DataFrame:
    source = _dashboard_monthly_source()
    monthly = pl.read_csv(source, try_parse_dates=True)
    if "strategy" not in monthly.columns:
        return monthly
    return (
        monthly.filter(
            pl.col("strategy").is_in(
                [
                    "alpha_top5_equal",
                    "alpha_top10_equal",
                    "Legacy",
                    "SPY total return",
                ]
            )
        )
        .pivot(on="strategy", index="holding_month", values="net_return")
        .rename(
            {
                "alpha_top5_equal": "alpha_top5_return",
                "alpha_top10_equal": "alpha_top10_return",
                "Legacy": "legacy_return",
                "SPY total return": "spy_return",
            }
        )
        .sort("holding_month")
    )


def _compact_shap_rows(
    samples: pl.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in samples.iter_rows(named=True):
        payload.append(
            {
                "m": _clean(row["decision_month"]),
                "t": row["ticker"],
                "f": int(row["fold"]),
                "v": [_clean(row.get(f"value__{feature}")) for feature in features],
                "s": [_clean(row.get(f"shap__{feature}")) for feature in features],
            }
        )
    return payload


def _monthly_shap_payload(
    samples_path: Path,
    lexicon_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    lexicon = pl.read_csv(lexicon_path).sort("importance_rank")
    features = lexicon["feature"].to_list()
    samples = pl.read_parquet(samples_path).sort(
        ["decision_month", "ticker"]
    )
    overview_parts = []
    for fold in samples["fold"].unique().sort().to_list():
        fold_rows = samples.filter(pl.col("fold") == fold)
        overview_parts.append(
            fold_rows.sample(
                n=min(SHAP_OVERVIEW_ROWS_PER_FOLD, fold_rows.height),
                seed=42 + int(fold),
                shuffle=False,
            )
        )
    overview = pl.concat(overview_parts).sort(["decision_month", "ticker"])
    month_counts = (
        samples.group_by("decision_month", "fold")
        .len(name="rows")
        .sort("decision_month")
    )
    return (
        _compact_shap_rows(overview, features),
        _clean(month_counts.to_dicts()),
        _clean(lexicon.to_dicts()),
        features,
    )


def _write_monthly_shap_sidecars(
    samples_path: Path,
    features: list[str],
    target_dir: Path,
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    samples = pl.read_parquet(samples_path).sort(["decision_month", "ticker"])
    files: list[dict[str, Any]] = []
    for month_rows in samples.partition_by("decision_month", maintain_order=True):
        month = _clean(month_rows["decision_month"][0])
        path = target_dir / f"{month[:7]}.json.gz"
        encoded = json.dumps(
            {"month": month, "rows": _compact_shap_rows(month_rows, features)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
        files.append(
            {
                "month": month,
                "rows": month_rows.height,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _hash(path),
            }
        )
    manifest = {
        "source": str(samples_path.relative_to(PROJECT_ROOT)),
        "source_sha256": _hash(samples_path),
        "features": len(features),
        "rows": samples.height,
        "months": len(files),
        "files": files,
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _fold_diagnostics() -> list[dict[str, Any]]:
    root = DIAGNOSTICS_DIR / "classification_h06"
    metrics = pl.read_csv(root / "fold_metrics.csv", try_parse_dates=True)
    splits = pl.read_csv(root / "fold_feature_manifest.csv", try_parse_dates=True)
    return _clean(splits.join(metrics, on="fold").sort("fold").to_dicts())


PERIOD_METRIC_FIELDS = (
    "total_return", "cagr", "annualized_volatility", "sharpe", "sortino",
    "calmar", "max_drawdown", "information_ratio", "beta", "alpha",
    "correlation", "benchmark_hit_rate", "var_95", "cvar_95", "omega",
    "up_capture", "down_capture",
)


def _period_metrics(monthly: pl.DataFrame) -> dict[str, list[list[float | None]]]:
    series = ("alpha_top5_return", "legacy_return", "spy_return")
    benchmark = monthly["spy_return"].to_numpy()
    months = monthly["holding_month"].to_list()
    output: dict[str, list[list[float | None]]] = {}
    for start in range(len(months)):
        for end in range(start, len(months)):
            rows: list[list[float | None]] = []
            for column in series:
                stats = advanced_performance_statistics(
                    monthly[column].to_numpy()[start : end + 1],
                    benchmark_returns=benchmark[start : end + 1],
                )
                rows.append([_clean(stats[field]) for field in PERIOD_METRIC_FIELDS])
            output[f"{months[start].isoformat()}|{months[end].isoformat()}"] = rows
    return output


def _drawdowns(monthly: pl.DataFrame) -> list[dict[str, Any]]:
    columns = {
        "alpha_top5_return": "Top 5 égal",
        "alpha_top10_return": "Top 10 égal",
        "legacy_return": "Legacy",
        "spy_return": "SPY",
    }
    wealth = {label: 1.0 for label in columns.values()}
    peak = wealth.copy()
    rows: list[dict[str, Any]] = []
    for row in monthly.sort("holding_month").iter_rows(named=True):
        item: dict[str, Any] = {"month": _clean(row["holding_month"])}
        for column, label in columns.items():
            wealth[label] *= 1.0 + float(row[column])
            peak[label] = max(peak[label], wealth[label])
            item[label] = {
                "wealth": round(wealth[label], 8),
                "drawdown": round(wealth[label] / peak[label] - 1.0, 8),
            }
        rows.append(item)
    return rows


def _regimes(monthly: pl.DataFrame) -> list[dict[str, Any]]:
    periods = [
        ("2011–2015", date(2011, 1, 1), date(2015, 12, 31)),
        ("2016–2020", date(2016, 1, 1), date(2020, 12, 31)),
        ("2021–2025", date(2021, 1, 1), date(2025, 12, 31)),
    ]
    columns = {
        "alpha_top5_return": "Top 5 égal",
        "alpha_top10_return": "Top 10 égal",
        "legacy_return": "Legacy",
        "spy_return": "SPY",
    }
    output: list[dict[str, Any]] = []
    for label, start, end in periods:
        frame = monthly.filter(
            pl.col("holding_month").is_between(start, end)
        )
        for column, series in columns.items():
            values = frame[column].to_list()
            wealth = 1.0
            peak = 1.0
            max_drawdown = 0.0
            for value in values:
                wealth *= 1.0 + float(value)
                peak = max(peak, wealth)
                max_drawdown = min(max_drawdown, wealth / peak - 1.0)
            years = len(values) / 12
            cagr = wealth ** (1 / years) - 1 if years else None
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values)
                / max(1, len(values) - 1)
            )
            volatility = math.sqrt(variance * 12)
            output.append(
                {
                    "period": label,
                    "series": series,
                    "months": len(values),
                    "cagr": cagr,
                    "volatility": volatility,
                    "sharpe": (cagr - 0.02) / volatility
                    if volatility
                    else None,
                    "max_drawdown": max_drawdown,
                }
            )
    return _clean(output)


def _live_payload() -> dict[str, Any]:
    manifest = json.loads((LIVE_DIR / "manifest.json").read_text())
    return {
        "manifest": _clean(manifest),
        "top5": _records(LIVE_DIR / "portfolio_top5.csv"),
        "top10": _records(LIVE_DIR / "portfolio_top10.csv"),
        "legacy": _records(
            LIVE_DIR / "legacy_portfolio_same_holding_month.csv"
        ),
    }


def build_payload() -> tuple[dict[str, Any], list[Path]]:
    champion_manifest = load_manifest(CHAMPION_DIR / "manifest.json")
    champion_data_manifest = (
        _project_path(champion_manifest["config"]["data_dir"]).parent
        / "data_input_manifest.json"
    )
    live_manifest = load_manifest(LIVE_DIR / "manifest.json")
    live_data_manifest = (
        _project_path(live_manifest["config"]["data_dir"]).parent
        / "data_input_manifest.json"
    )
    source_files = [
        CHAMPION_DIR / "manifest.json",
        champion_data_manifest,
        RISK_DIR / "manifest.json",
        TOPN_DIR / "manifest.json",
        TOPN_DIR / "alpha_shap_portfolio_manifest.json",
        TOPN_DIR / "monthly_portfolios.parquet",
        TOPN_DIR / "allocation_holdings.parquet",
        TOPN_DIR / "allocation_monthly.csv",
        _dashboard_monthly_source(),
        TOPN_DIR / "performance_legacy_convention.csv",
        TOPN_DIR / "annual_returns_wide.csv",
        TOPN_DIR / "cost_sensitivity.csv",
        TOPN_DIR / "paired_block_bootstrap.csv",
        TOPN_DIR / "promotion_gates.csv",
        TOPN_DIR / "rank_bucket_diagnostics.csv",
        TOPN_DIR / "alpha_shap_feature_lexicon.csv",
        DIAGNOSTICS_DIR / "classification_h06/shap_samples.parquet",
        DIAGNOSTICS_DIR / "classification_h06/fold_metrics.csv",
        DIAGNOSTICS_DIR / "classification_h06/fold_feature_manifest.csv",
        CHAMPION_DIR / "model_horizon_summary.csv",
        SCREENING_DIR / "model_horizon_summary.csv",
        EMA_SCREENING_DIR / "model_horizon_summary.csv",
        RISK_DIR / "risk_model_metrics.csv",
        RISK_DIR / "allocation_performance_legacy_convention.csv",
        RISK_DIR / "allocation_acceptance_gates.csv",
        LIVE_DIR / "manifest.json",
        live_data_manifest,
        LATEST_OPEN_SOURCE_MANIFEST,
        LIVE_DIR / "portfolio_top5.csv",
        LIVE_DIR / "portfolio_top10.csv",
        LIVE_DIR / "legacy_portfolio_same_holding_month.csv",
        HISTORICAL_LEGACY_COMMON_DIR / "manifest.json",
        HISTORICAL_LEGACY_COMMON_DIR / "legacy_common_total_return_holdings.parquet",
        HISTORICAL_LEGACY_COMMON_DIR / "legacy_common_total_return_monthly.csv",
        CURRENT_LEGACY_COMMON_DIR / "manifest.json",
        CURRENT_LEGACY_COMMON_DIR / "legacy_common_total_return_monthly.csv",
    ]
    missing = [path for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dashboard sources:\n" + "\n".join(map(str, missing))
        )
    shap, shap_month_counts, lexicon, features = _monthly_shap_payload(
        DIAGNOSTICS_DIR / "classification_h06/shap_samples.parquet",
        TOPN_DIR / "alpha_shap_feature_lexicon.csv",
    )
    shap_rows = sum(row["rows"] for row in shap_month_counts)
    monthly = _dashboard_monthly()
    attribution = _performance_attribution_payload()
    lineage = {
        "historical": _historical_lineage(monthly),
        "legacy_current": _current_legacy_lineage(),
        "live": _live_lineage(),
        "latest": _latest_data_lineage(),
    }
    payload = {
        "meta": {
            "created": datetime.now().astimezone().isoformat(),
            "test_start": str(monthly["holding_month"].min()),
            "test_end": str(monthly["holding_month"].max()),
            "test_months": monthly.height,
            "shap_rows": shap_rows,
            "shap_overview_rows": len(shap),
            "shap_features": len(features),
            "shap_months": len(shap_month_counts),
            "attribution_rows": len(attribution),
        },
        "performance": _records(
            TOPN_DIR / "performance_legacy_convention.csv"
        ),
        "monthly": _full_precision_records(monthly),
        "attribution": attribution,
        "legacy_current_monthly": _full_precision_records(
            _canonical_legacy_monthly(CURRENT_LEGACY_COMMON_DIR)
        ),
        "lineage": _clean(lineage),
        "period_metric_fields": PERIOD_METRIC_FIELDS,
        "period_metrics": _period_metrics(monthly),
        "wealth": _drawdowns(monthly),
        "regimes": _regimes(monthly),
        "annual": _records(TOPN_DIR / "annual_returns_wide.csv"),
        "holdings": _parquet_records(TOPN_DIR / "monthly_portfolios.parquet"),
        "costs": _records(TOPN_DIR / "cost_sensitivity.csv"),
        "bootstrap": _records(TOPN_DIR / "paired_block_bootstrap.csv"),
        "promotion": _records(TOPN_DIR / "promotion_gates.csv"),
        "buckets": _records(TOPN_DIR / "rank_bucket_diagnostics.csv"),
        "screening": _records(SCREENING_DIR / "model_horizon_summary.csv"),
        "ema_screening": _records(
            EMA_SCREENING_DIR / "model_horizon_summary.csv"
        ),
        "champion": _records(CHAMPION_DIR / "model_horizon_summary.csv"),
        "folds": _fold_diagnostics(),
        "risk_models": _records(RISK_DIR / "risk_model_metrics.csv"),
        "risk_performance": _records(
            RISK_DIR / "allocation_performance_legacy_convention.csv"
        ),
        "risk_gates": _records(
            RISK_DIR / "allocation_acceptance_gates.csv"
        ),
        "shap": shap,
        "shap_month_counts": shap_month_counts,
        "lexicon": lexicon,
        "features": features,
        "live": _live_payload(),
    }
    return payload, source_files


HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank — Centre de recherche Legacy & Boosting</title>
<style>
:root{--bg:#F5F6FA;--panel:#FFF;--ink:#141927;--muted:#687083;--line:#DDE1EA;--navy:#111D55;--navy2:#26387E;--gold:#9B8816;--green:#16794B;--red:#B03A45;--bluewash:#EEF1FB;--goldwash:#F7F3DD;--shadow:0 1px 2px rgba(17,29,85,.04)}
[data-theme="dark"]{--bg:#11141D;--panel:#1A1E29;--ink:#F2F3F7;--muted:#9EA6B7;--line:#313746;--navy:#8998E2;--navy2:#A7B2ED;--gold:#D6C45B;--green:#55B888;--red:#EC7A83;--bluewash:#20283E;--goldwash:#2B291B}
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.48 "IBM Plex Sans",sans-serif}
button,select,input{font:inherit;color:inherit}.mono,.metric strong,td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.shell{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;background:var(--navy);color:#fff;padding:24px 16px;display:flex;flex-direction:column}
.brand{font:600 18px "IBM Plex Mono";letter-spacing:-.03em;padding:0 10px 24px;border-bottom:1px solid rgba(255,255,255,.18)}.brand small{display:block;font:500 10px "IBM Plex Sans";opacity:.65;letter-spacing:.12em;text-transform:uppercase;margin-top:5px}
.tabs{display:grid;gap:4px;margin-top:20px}.tab{border:0;background:transparent;color:rgba(255,255,255,.72);text-align:left;padding:10px 12px;border-radius:4px;cursor:pointer;font-weight:600}.tab:hover,.tab.active{background:rgba(255,255,255,.13);color:#fff}.tab span{font-family:"IBM Plex Mono";font-size:10px;opacity:.55;margin-right:8px}
.aside-foot{margin-top:auto;padding:14px 10px 0;border-top:1px solid rgba(255,255,255,.18);font-size:11px;color:rgba(255,255,255,.65)}
main{min-width:0}.topbar{height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:10}.topbar .crumb{color:var(--muted)}.topbar button{border:1px solid var(--line);background:var(--panel);padding:7px 10px;border-radius:4px;cursor:pointer}
.page{display:none;padding:28px;max-width:1500px;margin:auto}.page.active{display:block}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.11em;color:var(--navy);font-weight:700}.hero h1{font-size:34px;line-height:1.08;letter-spacing:-.035em;margin:7px 0 10px}.hero p{color:var(--muted);max-width:1000px;font-size:16px}.hero{margin-bottom:22px}
.grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.panel{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);padding:18px}.panel h2,.panel h3{margin:0 0 12px;letter-spacing:-.02em}.panel h2{font-size:19px}.panel h3{font-size:15px}.panel p{color:var(--muted)}.metric{min-height:116px}.metric span{font-size:12px;color:var(--muted)}.metric strong{font-size:27px;display:block;margin:7px 0 3px;letter-spacing:-.04em}.metric small{color:var(--muted)}
.callout{border-left:4px solid var(--navy);background:var(--bluewash);padding:14px 16px;margin:14px 0}.callout.warn{border-color:var(--gold);background:var(--goldwash)}.callout.bad{border-color:var(--red)}.callout strong{display:block;margin-bottom:3px}
.section-head{display:flex;justify-content:space-between;align-items:end;gap:16px;margin:30px 0 12px}.section-head h2{font-size:22px;margin:0}.section-head p{margin:4px 0 0;color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.controls label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}select,input{border:1px solid var(--line);background:var(--panel);padding:8px 10px;border-radius:3px}
.table-wrap{overflow:auto;border:1px solid var(--line);background:var(--panel)}table{width:100%;border-collapse:collapse;min-width:780px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);z-index:1;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}td.num{text-align:right}tr:last-child td{border-bottom:0}.badge{display:inline-block;font:600 10px "IBM Plex Mono";padding:3px 6px;border:1px solid currentColor;border-radius:2px}.pass{color:var(--green)}.fail{color:var(--red)}.gold{color:var(--gold)}.navy{color:var(--navy)}
.chart{height:330px;position:relative}.chart svg{width:100%;height:100%;display:block}.chart text{fill:var(--muted);font:10px "IBM Plex Mono"}.chart .gridline{stroke:var(--line);stroke-width:1}.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:12px}.legend i{width:10px;height:10px;display:inline-block;margin-right:5px}
.portfolio-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.portfolio-card{border-top:4px solid var(--navy)}.portfolio-card.legacy{border-color:var(--gold)}.ticker-row{display:grid;grid-template-columns:54px 1fr auto;gap:8px;padding:9px 0;border-bottom:1px solid var(--line);align-items:center}.ticker-row:last-child{border-bottom:0}.ticker{font:600 12px "IBM Plex Mono"}.action{font:600 9px "IBM Plex Mono";padding:2px 4px;border:1px solid currentColor}.enter{color:var(--green)}.exit{color:var(--red)}.keep{color:var(--muted)}
.shap-layout{display:grid;grid-template-columns:minmax(340px,.9fr) minmax(500px,1.4fr);gap:14px}.shap-bar{display:grid;grid-template-columns:minmax(150px,1fr) 4fr 72px;gap:8px;align-items:center;margin:7px 0;font:11px "IBM Plex Mono"}.bar-track{height:10px;background:var(--bluewash)}.bar-fill{height:100%;background:var(--navy)}canvas{width:100%;height:400px;border:1px solid var(--line);background:var(--panel)}
.two-col-doc{columns:2 340px;column-gap:32px}.doc-block{break-inside:avoid;margin-bottom:18px}.doc-block h3{margin:0 0 5px}.doc-block p,.doc-block li{color:var(--muted)}code{font-family:"IBM Plex Mono";font-size:11px;color:var(--navy)}
.source-list{font:11px "IBM Plex Mono";word-break:break-all}.fine{font-size:11px;color:var(--muted)}.spacer{height:8px}
.method-flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}.method-step{position:relative;padding:14px 12px;border-top:3px solid var(--navy)}.method-step b{display:block;font:600 11px "IBM Plex Mono";color:var(--navy);margin-bottom:7px}.method-step p{font-size:12px;margin:0}.period-bar{position:sticky;top:64px;z-index:9;display:flex;flex-wrap:wrap;align-items:end;gap:10px;padding:13px 14px;margin:0 0 14px;background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow)}.period-bar .field{display:grid;gap:4px}.period-bar .field span{font:600 10px "IBM Plex Mono";text-transform:uppercase;color:var(--muted)}.preset{border:1px solid var(--line);background:var(--panel);padding:8px 10px;border-radius:3px;cursor:pointer}.preset.active{background:var(--navy);border-color:var(--navy);color:#fff}.period-status{margin-left:auto;font:500 11px "IBM Plex Mono";color:var(--muted)}
.advanced-table td:first-child{font-weight:600}.advanced-table .subtle{display:block;font:10px "IBM Plex Mono";color:var(--muted);margin-top:2px}.metric-help{border-bottom:1px dashed var(--muted);cursor:help}.episodes{display:grid;gap:7px}.episode{display:grid;grid-template-columns:1.25fr .7fr .7fr .9fr;gap:8px;padding:9px 0;border-bottom:1px solid var(--line);font:11px "IBM Plex Mono"}.episode:last-child{border-bottom:0}.method-note{display:grid;grid-template-columns:1fr 1fr;gap:14px}.formula-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.formula{padding:12px;background:var(--bluewash);border-left:3px solid var(--navy)}.formula b{display:block;margin-bottom:4px}.formula code{display:block;margin:5px 0;white-space:normal}.range-warning{color:var(--red);font-weight:600}
.segmented{display:inline-flex;border:1px solid var(--line);background:var(--panel)}.segmented button{border:0;border-right:1px solid var(--line);background:transparent;padding:8px 12px;cursor:pointer}.segmented button:last-child{border-right:0}.segmented button.active{background:var(--navy);color:#fff}.waterfall-scroll{overflow-x:auto;border:1px solid var(--line);background:var(--panel)}.waterfall-chart{height:430px;min-width:100%}.waterfall-chart svg{display:block;height:100%}.waterfall-chart text{font:10px "IBM Plex Mono";fill:var(--muted)}.waterfall-chart .wf-value{font-weight:600;fill:var(--ink)}.attribution-note{display:grid;grid-template-columns:1.2fr .8fr;gap:14px}.attribution-note code{font-size:12px}.attribution-table td:first-child{font-family:"IBM Plex Mono"}
@media(max-width:1100px){.g4{grid-template-columns:repeat(2,1fr)}.g3,.portfolio-cards{grid-template-columns:1fr 1fr}.shap-layout{grid-template-columns:1fr}}
@media(max-width:1100px){.method-flow{grid-template-columns:repeat(3,1fr)}.formula-grid{grid-template-columns:1fr 1fr}}
@media(max-width:760px){.shell{display:block}aside{position:sticky;height:auto;padding:12px;z-index:30}.brand{padding:2px 5px 10px}.tabs{display:flex;overflow:auto;margin-top:8px}.tab{white-space:nowrap;padding:8px}.aside-foot{display:none}.topbar{top:104px;height:52px;padding:0 14px}.page{padding:18px 12px}.hero h1{font-size:28px}.g4,.g3,.g2,.portfolio-cards,.method-flow,.formula-grid,.method-note,.attribution-note{grid-template-columns:1fr}.chart{height:270px}.two-col-doc{columns:1}.section-head{align-items:start;flex-direction:column}.period-bar{top:156px}.period-status{width:100%;margin-left:0}.episode{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="shell">
<aside>
  <div class="brand">ALPHARANK<small>Research control center</small></div>
  <nav class="tabs">
    <button class="tab active" data-page="overview"><span>01</span>Synthèse</button>
    <button class="tab" data-page="models"><span>02</span>Modèles & horizons</button>
    <button class="tab" data-page="backtest"><span>03</span>Backtest approfondi</button>
    <button class="tab" data-page="portfolios"><span>04</span>Actions par mois</button>
    <button class="tab" data-page="shap"><span>05</span>SHAP mensuel</button>
    <button class="tab" data-page="risk"><span>06</span>Risque</button>
    <button class="tab" data-page="live"><span>07</span>Production 2026</button>
    <button class="tab" data-page="docs"><span>08</span>Documentation & audit</button>
  </nav>
  <div class="aside-foot">Une source unique de lecture.<br>Résultats hors-échantillon, lineage et limites.</div>
</aside>
<main>
<header class="topbar"><div class="crumb">Legacy EMA / Boosting / <b id="crumb">Synthèse</b></div><button id="theme">◐ Thème</button></header>

<section class="page active" id="overview">
 <div class="hero"><div class="eyebrow">Décision actuelle · recherche non promue automatiquement</div><h1>Le boosting produit un alpha fort, mais la preuve reste conditionnelle.</h1><p>Le backtest historique compare Legacy et Boosting sur le même snapshot figé du 13 juillet 2026. Il s'arrête en novembre 2025 parce que cet ancien univers finissait en avril 2026. Le candidat live utilise séparément le snapshot validé du 27 juillet.</p></div>
 <div class="grid g4" id="headline-metrics"></div>
 <div class="callout warn"><strong>Conclusion de gouvernance</strong>Top 5 égal reste la meilleure allocation testée sur l'ancien snapshot commun. Ce résultat n'est pas encore la comparaison sur les dernières données historisées. Le live de juillet 2026 est un candidat opérationnel, pas une nouvelle preuve hors-échantillon.</div>
 <div class="section-head"><div><h2>Lineage des trois contextes</h2><p>Un snapshot commun est obligatoire à l'intérieur de chaque comparaison ; les périodes historiques, live et données disponibles ne sont jamais fusionnées silencieusement.</p></div></div>
 <div class="grid g3" id="lineage-contexts"></div>
 <div class="section-head"><div><h2>Ce qui est démontré</h2><p>Lecture synthétique, sans confondre performance et validité causale.</p></div></div>
 <div class="grid g3">
  <article class="panel"><h3>Signal</h3><p>La classification de surperformance à 6 mois, basée uniquement sur les EMA gagnantes de Legacy et leurs transformations mensuelles, est le champion retenu.</p></article>
  <article class="panel"><h3>Allocation</h3><p>Les cinq meilleurs scores, équipondérés à 20 %, dominent Top 10, Legacy et SPY sur ce backtest. Les rangs 6–10 diluent nettement le signal.</p></article>
  <article class="panel"><h3>Risque</h3><p>Les têtes volatilité/downside sont informatives, mais leur utilisation en pondération ne satisfait pas les garde-fous complets.</p></article>
 </div>
 <div class="section-head"><div><h2>Chaîne de décision</h2><p>Du modèle à l'action, puis au contrôle.</p></div></div>
 <div class="grid g4">
  <article class="panel"><div class="eyebrow">1 · Features</div><h3>EMA relatives Legacy</h3><p>Paires gagnantes disponibles point-in-time dans chaque fold, ratio action/SPY, puis ratio brut, rang, z-score et quartiles.</p></article>
  <article class="panel"><div class="eyebrow">2 · Modèle</div><h3>XGBoost classification H6</h3><p>Probabilité qu'une action soit dans le décile supérieur de surperformance future à six mois.</p></article>
  <article class="panel"><div class="eyebrow">3 · Portefeuille</div><h3>Top 5 égal</h3><p>Classement par score alpha brut, cinq titres, 20 % chacun, rebalancement mensuel.</p></article>
  <article class="panel"><div class="eyebrow">4 · Contrôle</div><h3>Legacy + SPY</h3><p>CAGR, Sharpe Legacy, drawdown, pire année, coûts et bootstrap apparié.</p></article>
 </div>
</section>

<section class="page" id="models">
 <div class="hero"><div class="eyebrow">Méthodes et validation temporelle</div><h1>Deux algorithmes distincts, un même contrat de portefeuille</h1><p>Legacy construit directement un panier par règles EMA et optimisation annuelle. Le boosting apprend une cible de surperformance future, puis transforme ses scores hors-échantillon en cinq positions équipondérées.</p></div>
 <div class="grid g2 method-note">
  <article class="panel"><div class="eyebrow">Méthode 1 · Legacy</div><h2>Quatre modèles Optuna, vote par fréquence</h2><p>Chaque fin de mois, Legacy calcule des ratios d'EMA du prix relatif action/SPY, applique les filtres fondamentaux et sectoriels point-in-time, puis exécute quatre branches <code>11/12/21/22</code>. Une action reçoit un poids proportionnel au nombre de branches qui la retiennent. Le panier décidé au mois t est détenu au mois t+1.</p><p class="fine">Référence publiée : <code>Combined_Frequency</code>. Coût historique : 0 pb. Les rendements manquants sont exclus puis les poids disponibles renormalisés.</p></article>
  <article class="panel"><div class="eyebrow">Méthode 2 · Boosting</div><h2>Classification XGBoost H6, allocation Top 5</h2><p>Le modèle utilise uniquement les paires EMA gagnantes connues avant chaque fold. La cible vaut 1 pour le décile supérieur de surperformance cumulée future contre SPY à six mois. Le modèle est entraîné sur le passé, calibré sur les six mois suivants, figé sur le test, puis les cinq scores mensuels les plus élevés sont équipondérés.</p><p class="fine">Backtest : 15 folds walk-forward, 10 pb × turnover, détention un mois malgré une cible H6. Aucun mois test ne sert au fit ni au choix des variables de son fold.</p></article>
 </div>
 <div class="section-head"><div><h2>Les 15 ensembles train → calibration → test</h2><p>Choisissez un ensemble pour voir ses périodes, volumes et écarts de métriques.</p></div><div class="controls"><label>Ensemble test</label><select id="fold-select"></select></div></div>
 <div class="grid g4" id="fold-metrics"></div>
 <div class="table-wrap"><table><thead><tr><th>Split</th><th>Période</th><th>Lignes</th><th>ROC AUC</th><th>PR AUC</th><th>Lift PR</th><th>NDCG@10</th><th>Excès Top 5 1m</th><th>Overlap Legacy</th></tr></thead><tbody id="fold-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Screening des objectifs et horizons</h2><p>Comparaison de classification, régression, ranking et teacher sur les horizons disponibles.</p></div></div>
 <div class="controls"><label>Jeu d'entrée</label><select id="model-dataset"><option value="ema">EMA gagnantes Legacy</option><option value="broad">Screening large initial</option></select><label>Méthode</label><select id="model-method"><option value="all">Toutes</option></select></div>
 <div class="callout"><strong>Champion retenu : classification, horizon 6 mois</strong>Le choix final ne repose pas sur le seul ROC AUC : PR AUC, lift sur la prévalence, rendement Top 5 à un mois, overlap Legacy, calibration et stabilité temporelle sont lus ensemble.</div>
 <div class="table-wrap"><table><thead><tr><th>Méthode</th><th>Horizon</th><th>Folds</th><th>Rows test</th><th>ROC AUC</th><th>PR AUC</th><th>Lift PR</th><th>Spearman</th><th>NDCG@10</th><th>RMSE norm.</th><th>Top5 excès H</th><th>Top5 excès 1m</th><th>Overlap Legacy</th></tr></thead><tbody id="model-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Lecture des tâches</h2></div></div>
 <div class="grid g4">
  <article class="panel"><h3>Classification</h3><p>Classe positive = décile supérieur de surperformance future contre SPY. Mesures : ROC AUC, PR AUC, lift, Brier et log-loss.</p></article>
  <article class="panel"><h3>Régression</h3><p>Prévoit directement la surperformance future. Mesures : RMSE, RMSE normalisée, MAE, R² et IC mensuel.</p></article>
  <article class="panel"><h3>Ranking</h3><p>Optimise l'ordre relatif par mois. Mesures : NDCG@5/10/20 et lift contre l'absence de signal.</p></article>
  <article class="panel"><h3>Teacher</h3><p>Essaie de reproduire le choix Legacy. Utile comme diagnostic d'imitation, insuffisant pour prouver l'alpha futur.</p></article>
 </div>
</section>

<section class="page" id="backtest">
 <div class="hero"><div class="eyebrow">Audit interactif · période commune</div><h1>Comprendre puis disséquer le backtest</h1><p>Le modèle apprend à six mois, mais le portefeuille est reclassé et détenu un mois. Choisissez ensuite n'importe quelle période : tous les indicateurs, graphiques et années sont recalculés uniquement sur les mois communs sélectionnés.</p></div>
 <div class="section-head"><div><h2>La mécanique exacte, de la donnée au rendement</h2><p>Ce que sait le modèle à chaque étape — et ce qu'il ne sait pas encore.</p></div></div>
 <div class="method-flow">
  <article class="panel method-step"><b>01 · mois t</b><h3>Photo causale</h3><p>Prix disponibles à la fin du mois de décision. Ratio action/SPY et EMA uniquement jusqu'à cette date.</p></article>
  <article class="panel method-step"><b>02 · features PIT</b><h3>Liste croissante</h3><p>Seulement les gagnantes connues avant le fold : 3 paires/15 variables au fold 1, jusqu'à 37/185 au fold 15.</p></article>
  <article class="panel method-step"><b>03 · cible H6</b><h3>Surperformance future</h3><p>Classe 1 si l'action finit dans le décile supérieur de surperformance cumulée contre SPY à six mois.</p></article>
  <article class="panel method-step"><b>04 · fold externe</b><h3>Train → validation → test</h3><p>Le modèle est figé avant son bloc test, généralement douze mois. Aucun mois test ne sert au fit.</p></article>
  <article class="panel method-step"><b>05 · mois t+1</b><h3>Top 5 équipondéré</h3><p>Classement par score brut, cinq actions à 20 %, puis nouvelle décision le mois suivant.</p></article>
  <article class="panel method-step"><b>06 · mesure</b><h3>Rendement 1 mois</h3><p>Rendement réalisé du mois de détention, net de 10 pb × turnover, comparé aux mêmes mois Legacy et SPY.</p></article>
 </div>
 <div class="method-note">
  <div class="callout"><strong>Pourquoi une cible 6 mois mais une détention 1 mois ?</strong>H6 stabilise le signal de surperformance. Le portefeuille est néanmoins rescored chaque mois : il exploite ce signal long en renouvelant mensuellement ses cinq convictions les plus fortes.</div>
  <div class="callout warn"><strong>Ce que signifie « hors-échantillon » ici</strong>15 modèles successifs couvrent le test. Un modèle est entraîné avant chaque fold, pas chaque mois du fold. La dernière fenêtre est plus courte. Une période choisie ci-dessous ne réentraîne rien : elle tranche uniquement les prédictions OOS déjà figées.</div>
 </div>
 <div class="period-bar" aria-label="Sélecteur de période du backtest">
  <label class="field"><span>Début</span><select id="bt-start"></select></label>
  <label class="field"><span>Fin</span><select id="bt-end"></select></label>
  <button class="preset active" data-period="full">Tout</button>
  <button class="preset" data-period="120">10 ans</button>
  <button class="preset" data-period="60">5 ans</button>
  <button class="preset" data-period="36">3 ans</button>
  <span class="period-status" id="bt-period-status"></span>
 </div>
 <div id="snapshot-reconciliation" class="callout warn"></div>
 <div class="grid g4" id="backtest-metrics"></div>
 <div hidden><table><tbody id="perf-rows"></tbody><tbody id="regime-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Tableau de bord avancé sur la période</h2><p>Survolez les noms soulignés pour la définition courte ; les formules complètes sont plus bas.</p></div></div>
 <div class="table-wrap"><table class="advanced-table"><thead><tr><th>Méthode</th><th>Rendement total</th><th>CAGR</th><th>Vol.</th><th><span class="metric-help" title="(CAGR - 2 %) / volatilité annualisée">Sharpe Legacy</span></th><th><span class="metric-help" title="(CAGR - 2 %) / downside deviation annualisée">Sortino</span></th><th><span class="metric-help" title="CAGR / valeur absolue du max drawdown">Calmar</span></th><th>Max DD</th><th><span class="metric-help" title="Excès annualisé moyen / tracking error">Information ratio</span></th><th>Beta SPY</th><th>Alpha ann.</th><th>Corr. SPY</th><th>Hit vs SPY</th><th>VaR 95 %</th><th>CVaR 95 %</th></tr></thead><tbody id="advanced-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Richesse cumulée</h2><p>Base 1 au début de la période choisie.</p></div></div>
 <article class="panel"><div class="legend" id="wealth-legend"></div><div class="chart" id="wealth-chart"></div></article>
 <div class="section-head"><div><h2>Drawdowns</h2><p>Baisse depuis le plus haut atteint à l'intérieur de la période.</p></div></div>
 <article class="panel"><div class="chart" id="dd-chart"></div></article>
 <div class="section-head"><div><h2>Indicateurs glissants</h2><p>Détecter si l'avantage est durable ou concentré dans quelques régimes.</p></div><div class="controls"><label>Fenêtre</label><select id="rolling-window"><option value="12">12 mois</option><option value="24">24 mois</option><option value="36" selected>36 mois</option><option value="60">60 mois</option></select><label>Indicateur</label><select id="rolling-metric"><option value="sharpe">Sharpe Legacy</option><option value="excess_spy">Excès annualisé vs SPY</option><option value="excess_legacy">Excès annualisé vs Legacy</option><option value="volatility">Volatilité</option><option value="drawdown">Max drawdown fenêtre</option></select></div></div>
 <article class="panel"><div class="chart" id="rolling-chart"></div></article>
 <div class="section-head"><div><h2>Épisodes de drawdown</h2><p>Pic, creux, récupération et durée. Les trois pires épisodes par méthode.</p></div></div>
 <div class="grid g2" id="drawdown-episodes"></div>
 <div class="section-head"><div><h2>Rendements par année dans la sélection</h2><p>* année partielle ou tronquée par le filtre.</p></div></div>
 <div class="table-wrap"><table id="annual-table"></table></div>
 <div class="section-head"><div><h2>Extrêmes et forme de distribution</h2></div></div>
 <div class="table-wrap"><table><thead><tr><th>Méthode</th><th>Meilleur mois</th><th>Pire mois</th><th>Mois positifs</th><th>Skewness</th><th>Kurtosis excès</th><th>Omega 0 %</th><th>Capture haussière</th><th>Capture baissière</th></tr></thead><tbody id="distribution-rows"></tbody></table></div>
 <div class="section-head"><div><h2>D'où vient le CAGR ?</h2><p>Waterfall exact des actions, des coûts, des années et des mois sur la période choisie.</p></div></div>
 <div class="controls">
  <label>Méthode</label><select id="attr-strategy"></select>
  <label>Décomposition</label><div class="segmented" id="attr-axis"><button class="active" data-axis="actions">Actions</button><button data-axis="years">Années</button><button data-axis="months">Mois</button></div>
  <span id="attr-scope-wrap"><label>Détail actions</label> <select id="attr-scope"></select></span>
 </div>
 <div class="grid g4" id="attr-metrics"></div>
 <div class="attribution-note">
  <div class="callout"><strong>Réconciliation exacte</strong><code>Σ contributions log annualisées → exp(somme) − 1 = CAGR</code><br>Les contributions log sont additives. La barre « effet composé » fait le passage non linéaire vers le CAGR. Les impacts marginaux par action ne sont pas additifs.</div>
  <div class="callout warn"><strong>Lecture par mois</strong>Le drill-down d'un mois explique son rendement mensuel, sans l'annualiser artificiellement. L'axe « Mois » mesure la contribution de chaque mois au CAGR de toute la période sélectionnée.</div>
 </div>
 <div class="waterfall-scroll"><div class="waterfall-chart" id="attr-waterfall"></div></div>
 <div class="section-head"><div><h2 id="attr-table-title">Détail exhaustif</h2><p id="attr-table-note"></p></div><div class="controls"><label>Filtrer</label><input id="attr-search" type="search" placeholder="Ticker, année ou mois"></div></div>
 <div class="table-wrap"><table class="attribution-table"><thead><tr><th>Composante</th><th>Type</th><th>Mois actifs</th><th>Contribution log</th><th>Impact CAGR marginal</th><th>Contribution simple cumulée</th><th>Poids effectif moyen</th></tr></thead><tbody id="attr-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Robustesse et dilution</h2></div></div>
 <div class="grid g2">
  <article class="panel"><h3>Bootstrap apparié, blocs de 12 mois</h3><div id="bootstrap-summary"></div><p class="fine">2 000 réplications. L'intervalle préserve mieux l'autocorrélation qu'un bootstrap mensuel naïf.</p></article>
  <article class="panel"><h3>Rangs 1–5 contre 6–10</h3><div id="bucket-summary"></div></article>
  <article class="panel"><h3>Sensibilité aux coûts</h3><div id="cost-summary"></div></article>
  <article class="panel"><h3>Garde-fous Top 10</h3><div id="promotion-summary"></div></article>
 </div>
 <div class="section-head"><div><h2>Lexique des indicateurs</h2><p>Conventions utilisées dans le calcul interactif.</p></div></div>
 <div class="formula-grid">
  <div class="formula"><b>CAGR</b><code>(richesse finale)^(12 / N mois) − 1</code><span class="fine">Taux composé annualisé sur la période filtrée.</span></div>
  <div class="formula"><b>Sharpe Legacy</b><code>(CAGR − 2 %) / volatilité annualisée</code><span class="fine">Même convention que les rapports Legacy ; différente du Sharpe arithmétique académique.</span></div>
  <div class="formula"><b>Sortino</b><code>(CAGR − 2 %) / downside deviation</code><span class="fine">Ne pénalise que les rendements mensuels négatifs.</span></div>
  <div class="formula"><b>Calmar</b><code>CAGR / |max drawdown|</code><span class="fine">Rendement composé obtenu par unité de perte maximale observée.</span></div>
  <div class="formula"><b>Information ratio</b><code>12 × moyenne(R − B) / [écart-type(R − B) × √12]</code><span class="fine">B = SPY ; cohérence de l'excès plutôt que performance absolue.</span></div>
  <div class="formula"><b>Alpha / Beta</b><code>beta = cov(R,B)/var(B) ; alpha = 12×[R−Rf−beta(B−Rf)]</code><span class="fine">Régression mensuelle simple contre SPY, taux sans risque 2 % annualisé.</span></div>
  <div class="formula"><b>VaR / CVaR 95 %</b><code>quantile 5 % ; moyenne des mois ≤ VaR</code><span class="fine">Mesures historiques mensuelles, sans hypothèse de normalité.</span></div>
  <div class="formula"><b>Capture</b><code>moyenne stratégie / moyenne SPY, mois SPY ±</code><span class="fine">Participation aux marchés haussiers et baissiers ; interpréter avec le nombre de mois.</span></div>
  <div class="formula"><b>Omega 0 %</b><code>somme gains mensuels / |somme pertes mensuelles|</code><span class="fine">Rapport gains/pertes au seuil mensuel de zéro.</span></div>
 </div>
 <div class="callout bad"><strong>Règle d'interprétation</strong>Une fenêtre courte peut afficher des ratios extrêmes et trompeurs. Sous 36 mois, le rapport signale la fragilité ; sous 12 mois, CAGR, Sharpe, Sortino et alpha ne doivent pas servir à promouvoir un modèle.</div>
</section>

<section class="page" id="portfolios">
 <div class="hero"><div class="eyebrow">Audit mensuel des décisions</div><h1>Quelles actions sont détenues chaque mois ?</h1><p>Choisissez un mois de détention. Les badges ENTER / KEEP sont calculés par rapport au mois précédent ; les sorties sont listées séparément. Les rendements affichés sont réalisés sur le mois de détention.</p></div>
 <div class="controls"><label>Mois de détention</label><select id="holding-month"></select><span id="month-returns" class="mono"></span></div>
 <div class="portfolio-cards" id="portfolio-cards"></div>
</section>

<section class="page" id="shap">
 <div class="hero"><div class="eyebrow">Explication hors-échantillon</div><h1>SHAP par ensemble de test et par mois</h1><p>Le détail d'un mois contient toutes les actions effectivement scorées, sans échantillonnage. La vue « tous les mois » reste un échantillon global de navigation, explicitement signalé pour préserver la rapidité du rapport.</p></div>
 <div class="callout warn"><strong>Point de méthode essentiel</strong>Le backtest ne réentraîne pas tous les mois : un modèle est ajusté une fois par fold, puis utilisé sur son bloc test (généralement 12 mois). Le filtre mensuel explique donc un mois hors-échantillon sous le modèle de son fold. Le live, lui, est réentraîné à chaque exécution mensuelle.</div>
 <div class="controls"><label>Ensemble test</label><select id="shap-fold"><option value="all">Tous les ensembles</option></select><label>Mois de décision</label><select id="shap-month"></select><label>Variable individuelle</label><select id="shap-feature"></select></div>
 <div class="grid g4" id="shap-metrics"></div>
 <div class="shap-layout">
  <article class="panel"><h2>Importance |SHAP| décroissante</h2><p class="fine">Recalculée pour le mois sélectionné. Unité : marge brute XGBoost / log-odds, avant calibration isotone.</p><div id="shap-bars"></div></article>
  <article class="panel"><h2>Beeswarm — Top 15</h2><p class="fine">Axe horizontal = contribution SHAP ; couleur = valeur de la variable, faible → élevée.</p><canvas id="beeswarm" width="1000" height="620"></canvas></article>
 </div>
 <div class="section-head"><div><h2 id="individual-title">Analyse individuelle</h2><p>Valeur de la variable contre contribution SHAP.</p></div></div>
 <canvas id="individual" width="1200" height="500"></canvas>
 <div class="section-head"><div><h2>Détail des observations</h2><p id="shap-detail-note"></p></div></div>
 <div class="table-wrap"><table><thead><tr><th>Mois décision</th><th>Ticker</th><th>Fold</th><th>Valeur</th><th>SHAP</th></tr></thead><tbody id="shap-detail"></tbody></table></div>
 <div class="section-head"><div><h2>Lexique exact des 185 variables</h2></div><div class="controls"><input id="lexicon-search" placeholder="Rechercher une variable…"></div></div>
 <div class="table-wrap"><table><thead><tr><th>Rang global</th><th>Variable</th><th>|SHAP| global</th><th>Transformation</th><th>Spans EMA</th><th>Unité</th><th>Définition exacte</th><th>Interprétation</th></tr></thead><tbody id="lexicon-rows"></tbody></table></div>
</section>

<section class="page" id="risk">
 <div class="hero"><div class="eyebrow">Deuxième tête · contrôle du risque</div><h1>Volatilité, downside et allocation</h1><p>Les têtes de risque sont entraînées uniquement sur des rendements quotidiens strictement futurs. Elles servent à tester des poids inverse-volatilité ou inverse-downside sur la même sélection alpha.</p></div>
 <div class="callout bad"><strong>Aucun overlay risque n'est promu</strong>Certaines variantes améliorent le Sharpe, mais aucune ne passe simultanément le drawdown, le coût, la perte de CAGR et, le cas échéant, la concentration sectorielle.</div>
 <div class="section-head"><div><h2>Qualité prédictive des têtes risque</h2></div></div>
 <div class="table-wrap"><table><thead><tr><th>Tête</th><th>Horizon</th><th>Tâche</th><th>Target</th><th>Rows test</th><th>Spearman</th><th>RMSE</th><th>RMSE norm.</th><th>MAE</th><th>ROC AUC</th><th>PR AUC</th><th>Lift PR</th></tr></thead><tbody id="risk-model-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Allocations risque vs références</h2></div></div>
 <div class="table-wrap"><table><thead><tr><th>Méthode</th><th>CAGR</th><th>Vol.</th><th>Sharpe</th><th>Max DD</th><th>Pire année</th></tr></thead><tbody id="risk-perf-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Garde-fous d'acceptation</h2></div></div>
 <div id="risk-gates" class="grid g2"></div>
</section>

<section class="page" id="live">
 <div class="hero"><div class="eyebrow">Candidat de production · juillet 2026</div><h1>Portefeuille calculé sur les prix disponibles fin juin</h1><p>Le modèle est réentraîné sur l'historique causal disponible, calibré sur juillet–décembre 2025, puis score l'univers connu du mois de détention juillet 2026.</p></div>
 <div class="grid g4" id="live-metrics"></div>
 <div class="callout warn"><strong>Ce n'est pas un test scellé</strong>Les métriques affichées sont celles de la fenêtre de validation 2025-07 à 2025-12. Le portefeuille juillet 2026 est une décision live, sans rendement futur encore utilisé.</div>
 <div class="portfolio-cards" id="live-portfolios"></div>
 <div class="section-head"><div><h2>Lineage de production</h2></div></div>
 <div class="table-wrap"><table><tbody id="live-lineage"></tbody></table></div>
</section>

<section class="page" id="docs">
 <div class="hero"><div class="eyebrow">Référence centrale</div><h1>Méthode, biais, conventions et actions</h1><p>Cette page documente ce qu'il faut savoir pour interpréter, reproduire et faire évoluer les résultats sans fuite de données.</p></div>
 <div class="two-col-doc">
  <div class="doc-block"><h3>Legacy</h3><p>Méthode déterministe fondée sur des paires d'EMA relatives action/SPY sélectionnées historiquement, puis sur un portefeuille concentré. Elle est toujours conservée comme référence côte à côte.</p></div>
  <div class="doc-block"><h3>Features exactes</h3><p>Pour chaque paire (a,b) : <code>EMA_a(P_action/P_SPY) / EMA_b(P_action/P_SPY)</code>, puis ratio brut, rang centile mensuel, z-score cross-sectionnel et indicateurs top/bottom quartile. Aucune fondamentale dans le champion présenté.</p></div>
  <div class="doc-block"><h3>Target alpha H6</h3><p>Classe positive si la surperformance cumulée future contre SPY à six mois appartient au décile supérieur du mois. Le modèle classe les actions ; la probabilité calibrée sert à l'interprétation, le score brut au ranking.</p></div>
  <div class="doc-block"><h3>Géométrie temporelle</h3><p>En historique : entraînement ancien, validation/calibration suivante, puis bloc test chronologique. Le modèle est réentraîné entre les folds, pas entre les mois d'un même bloc. Le dernier fold peut être plus court.</p></div>
  <div class="doc-block"><h3>Anti-fuite</h3><ul><li>Features datées au mois de décision.</li><li>Labels entièrement mûrs avant la fin du train.</li><li>Calibration apprise avant le test.</li><li>Univers et exclusions explicités.</li><li>Comparaisons sur mois communs.</li><li>Prix incohérents mis en quarantaine sur toute leur trajectoire.</li></ul></div>
  <div class="doc-block"><h3>SHAP</h3><p>Chaque vue mensuelle explique exhaustivement toutes les actions scorées, soit 361 à 497 observations. Seule la vue globale utilise un échantillon de 80 lignes par fold, explicitement signalé.</p></div>
  <div class="doc-block"><h3>Risque</h3><p>Volatilité réalisée = écart-type annualisé des rendements quotidiens futurs. Downside = racine annualisée de la moyenne des carrés des rendements négatifs futurs. Classe high-vol = top 20 % cross-sectionnel de volatilité future.</p></div>
  <div class="doc-block"><h3>Coûts et Sharpe</h3><p>Le backtest principal applique 10 pb multipliés par le turnover. La sensibilité couvre plusieurs coûts. Le Sharpe dit « Legacy » emploie un taux sans risque annuel fixe de 2 % et le CAGR au numérateur.</p></div>
  <div class="doc-block"><h3>Ce qu'il ne faut pas conclure</h3><p>Une forte performance ne prouve pas que le signal survivra hors de la période étudiée. Multiple testing, révisions d'univers, survivorship historique imparfait et faible nombre de régimes restent des risques.</p></div>
  <div class="doc-block"><h3>Décision recommandée</h3><p>Conserver Top 5 égal comme challenger live surveillé, Legacy comme contrôle et SPY comme benchmark. Ne promouvoir Top 10 ou un overlay risque qu'après passage des gates sur une nouvelle période réellement scellée.</p></div>
 </div>
 <div class="section-head"><div><h2>Sources et intégrité</h2><p>Le manifeste adjacent contient les empreintes SHA-256 de chaque source.</p></div></div>
 <div class="source-list" id="sources"></div>
</section>
</main></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("payload").textContent);
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const pct=(v,d=1)=>v==null?"—":(100*v).toFixed(d)+" %";
const num=(v,d=2)=>v==null?"—":Number(v).toFixed(d);
const money=(v)=>v==null?"—":"$"+Number(v).toFixed(2);
const labels={alpha_top5_equal:"Top 5 égal",alpha_top10_equal:"Top 10 égal","SPY total return":"SPY",Legacy:"Legacy"};
const color={"Top 5 égal":"#111D55","Top 10 égal":"#6071B5","Legacy":"#9B8816","Legacy · snapshot 13/07":"#9B8816","SPY":"#657083"};
function metric(label,value,note){return `<article class="panel metric"><span>${label}</span><strong>${value}</strong><small>${note||""}</small></article>`}
function badge(ok){return `<span class="badge ${ok?"pass":"fail"}">${ok?"PASS":"FAIL"}</span>`}
function td(v,cls="num"){return `<td class="${cls}">${v}</td>`}

$$(".tab").forEach(b=>b.onclick=()=>{$$(".tab,.page").forEach(x=>x.classList.remove("active"));b.classList.add("active");$("#"+b.dataset.page).classList.add("active");$("#crumb").textContent=b.textContent.trim().replace(/^\\d+/,"");window.scrollTo(0,0);if(b.dataset.page==="backtest")renderBacktest();if(b.dataset.page==="shap")renderShap()});
$("#theme").onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==="dark"?"":"dark";if($("#shap").classList.contains("active"))renderShap();if($("#backtest").classList.contains("active"))renderBacktest()};

const perfBy=s=>D.performance.find(x=>x.series===s);
const p5=perfBy("alpha_top5_equal"),p10=perfBy("alpha_top10_equal"),leg=perfBy("Legacy"),spy=perfBy("SPY total return");
$("#headline-metrics").innerHTML=metric("Top 5 · CAGR",pct(p5.cagr),"net 10 pb × turnover")+metric("Legacy · CAGR",pct(leg.cagr),"snapshot commun du 13/07")+metric("SPY · CAGR",pct(spy.cagr),"total return · adjusted_close")+metric("Top 5 · Sharpe",num(p5.sharpe),"Legacy convention");
const lineageCards=[D.lineage.historical,D.lineage.legacy_current,D.lineage.live,D.lineage.latest];
$("#lineage-contexts").innerHTML=lineageCards.map(x=>`<article class="panel"><div class="eyebrow">${x.status==="same_snapshot_verified"?"Snapshot contrôlé":x.status==="canonical_legacy_replay"?"Replay canonique":"Données plus récentes"}</div><h3>${x.label}</h3><p class="mono">${x.snapshot_id}</p><p>Prix max : <b>${x.price_max_date}</b><br>${x.last_holding_month?`Dernière détention complète : <b>${x.last_holding_month.slice(0,7)}</b><br>`:""}${x.holding_month?`Détention live : <b>${x.holding_month.slice(0,7)}</b><br>`:""}${x.universe_max_month?`Univers : <b>${x.universe_max_month.slice(0,7)}</b>`:""}</p><p class="fine">${x.note}</p></article>`).join("");
$("#backtest-metrics").innerHTML=metric("Top 5 · CAGR",pct(p5.cagr),`Δ Legacy ${pct(p5.cagr-leg.cagr)}`)+metric("Top 5 · Sharpe",num(p5.sharpe),`Legacy ${num(leg.sharpe)}`)+metric("Top 5 · Max DD",pct(p5.max_drawdown),`Legacy ${pct(leg.max_drawdown)}`)+metric("Top 10 · CAGR",pct(p10.cagr),`Δ Top 5 ${pct(p10.cagr-p5.cagr)}`);

function modelRows(){
 const rows=$("#model-dataset").value==="ema"?D.ema_screening:D.screening, method=$("#model-method").value;
 $("#model-rows").innerHTML=rows.filter(r=>method==="all"||r.method===method).map(r=>`<tr><td>${r.method}${r.method==="classification"&&r.horizon===6?' <span class="badge pass">CHAMPION</span>':""}</td>${td(r.horizon+"m")}${td(r.folds)}${td(r.test_rows)}${td(num(r.roc_auc,3))}${td(num(r.pr_auc_average_precision,3))}${td(num(r.pr_auc_lift_vs_prevalence,2))}${td(num(r.spearman_ic,3))}${td(num(r.ndcg_at_10,3))}${td(num(r.normalized_rmse,3))}${td(pct(r.top5_horizon_excess))}${td(pct(r.top5_one_month_excess))}${td(pct(r.top5_legacy_overlap))}</tr>`).join("");
}
const methods=[...new Set(D.ema_screening.map(x=>x.method))];$("#model-method").innerHTML+=[...methods].map(x=>`<option>${x}</option>`).join("");$("#model-dataset").onchange=modelRows;$("#model-method").onchange=modelRows;modelRows();

$("#fold-select").innerHTML=D.folds.map(x=>`<option value="${x.fold}">Fold ${x.fold} · test ${x.test_start.slice(0,7)} → ${x.test_end.slice(0,7)}</option>`).join("");
function renderFold(){
 const f=D.folds.find(x=>x.fold===Number($("#fold-select").value))||D.folds[0];
 $("#fold-metrics").innerHTML=metric("Train",`${f.train_start.slice(0,7)} → ${f.train_cutoff.slice(0,7)}`,`${f.train_rows} lignes · labels H6 mûrs`)+metric("Calibration",`${f.validation_start.slice(0,7)} → ${f.validation_end.slice(0,7)}`,`${f.validation_rows} lignes · early stopping + isotonic`)+metric("Test",`${f.test_start.slice(0,7)} → ${f.test_end.slice(0,7)}`,`${f.test_rows} lignes · strictement hors-échantillon`)+metric("Variables",f.kept_feature_count,`${f.winner_pair_count} paires EMA connues au cutoff`);
 const splits=[['Train','train',f.train_start,f.train_cutoff,f.train_rows],['Calibration','validation',f.validation_start,f.validation_end,f.validation_rows],['Test OOS','test',f.test_start,f.test_end,f.test_rows]];
 $("#fold-rows").innerHTML=splits.map(([label,prefix,start,end,rows])=>`<tr><td>${label}</td><td class="mono">${start.slice(0,7)} → ${end.slice(0,7)}</td>${td(rows)}${td(num(f[prefix+'_roc_auc'],3))}${td(num(f[prefix+'_pr_auc_average_precision'],3))}${td(num(f[prefix+'_pr_auc_lift_vs_prevalence'],2))}${td(num(f[prefix+'_ndcg_at_10'],3))}${td(pct(f[prefix+'_top5_one_month_excess']))}${td(pct(f[prefix+'_top5_legacy_overlap']))}</tr>`).join("");
}
$("#fold-select").onchange=renderFold;renderFold();

const perfOrder=["alpha_top5_equal","alpha_top10_equal",...D.performance.map(x=>x.series).filter(x=>!["alpha_top5_equal","alpha_top10_equal","Legacy","SPY total return"].includes(x)),"Legacy","SPY total return"];
$("#perf-rows").innerHTML=perfOrder.map(s=>perfBy(s)).filter(Boolean).map(r=>`<tr><td>${labels[r.series]||r.series.replaceAll("_"," ")}</td>${td(pct(r.cagr))}${td(num(r.sharpe))}${td(pct(r.annualized_volatility))}${td(pct(r.max_drawdown))}${td(`${r.worst_full_calendar_year} · ${pct(r.worst_full_calendar_year_return)}`)}${td(pct(r.average_turnover))}${td(pct(r.average_maximum_position_weight))}${td(pct(r.maximum_sector_weight))}</tr>`).join("");
$("#regime-rows").innerHTML=D.regimes.map(r=>`<tr><td>${r.period}</td><td>${r.series}</td>${td(r.months)}${td(pct(r.cagr))}${td(pct(r.volatility))}${td(num(r.sharpe))}${td(pct(r.max_drawdown))}</tr>`).join("");
const annualCols=["year","months","alpha_top5_equal","alpha_top10_equal","Legacy","SPY total return"];
$("#annual-table").innerHTML=`<thead><tr>${annualCols.map(x=>`<th>${labels[x]||x}</th>`).join("")}</tr></thead><tbody>`+D.annual.map(r=>`<tr>${annualCols.map((x,i)=>i<2?`<td>${r[x]}${x==="year"&&r.months<12?"*":""}</td>`:td(pct(r[x]))).join("")}</tr>`).join("")+"</tbody>";

function drawLine(id,field){
 const el=$("#"+id),w=Math.max(700,el.clientWidth||900),h=310,p={l:52,r:18,t:16,b:34},series=["Top 5 égal","Top 10 égal","Legacy","SPY"];
 let vals=D.wealth.flatMap(r=>series.map(s=>r[s][field])),min=Math.min(...vals),max=Math.max(...vals);if(field==="wealth")min=0;
 const x=i=>p.l+i*(w-p.l-p.r)/(D.wealth.length-1),y=v=>p.t+(max-v)*(h-p.t-p.b)/(max-min||1);
 let svg=`<svg viewBox="0 0 ${w} ${h}">`;
 for(let j=0;j<5;j++){const v=min+(max-min)*j/4,yy=y(v);svg+=`<line class="gridline" x1="${p.l}" y1="${yy}" x2="${w-p.r}" y2="${yy}"/><text x="4" y="${yy+3}">${field==="wealth"?v.toFixed(1):pct(v,0)}</text>`}
 series.forEach(s=>{const pts=D.wealth.map((r,i)=>`${x(i)},${y(r[s][field])}`).join(" ");svg+=`<polyline points="${pts}" fill="none" stroke="${color[s]}" stroke-width="${s==="Top 5 égal"?2.5:1.7}"/>`});
 [0,Math.floor(D.wealth.length/2),D.wealth.length-1].forEach(i=>svg+=`<text x="${x(i)}" y="${h-8}" text-anchor="${i===0?"start":i===D.wealth.length-1?"end":"middle"}">${D.wealth[i].month.slice(0,7)}</text>`);el.innerHTML=svg+"</svg>";
}
$("#wealth-legend").innerHTML=["Top 5 égal","Top 10 égal","Legacy","SPY"].map(s=>`<span><i style="background:${color[s]}"></i>${s}</span>`).join("");
const boot=D.bootstrap.filter(x=>x.strategy==="alpha_top5_equal"&&["Legacy","SPY total return"].includes(x.comparator));
$("#bootstrap-summary").innerHTML=boot.map(x=>`<p><b>vs ${labels[x.comparator]||x.comparator}</b><br>Δ Sharpe observé <span class="mono">${num(x.observed_sharpe_difference,3)}</span><br>IC 95 % <span class="mono">[${num(x.sharpe_difference_ci_low,3)} ; ${num(x.sharpe_difference_ci_high,3)}]</span></p>`).join("");
$("#bucket-summary").innerHTML=D.buckets.map(x=>`<p><b>${x.bucket==="ranks_1_5"?"Rangs 1–5":"Rangs 6–10"}</b> · CAGR <span class="mono">${pct(x.cagr)}</span> · Sharpe <span class="mono">${num(x.sharpe)}</span> · DD <span class="mono">${pct(x.max_drawdown)}</span></p>`).join("");
$("#cost-summary").innerHTML=[10,25,50,100].map(c=>{const x=D.costs.find(r=>r.strategy==="alpha_top5_equal"&&r.cost_bps===c);return `<p><b>${c} pb × turnover</b> · CAGR <span class="mono">${pct(x?.cagr)}</span> · Sharpe <span class="mono">${num(x?.sharpe)}</span></p>`}).join("");
$("#promotion-summary").innerHTML=D.promotion.map(x=>`<p>${badge(x.pass)} ${x.gate.replaceAll("_"," ")}</p>`).join("");

const months=[...new Set(D.holdings.map(x=>x.holding_month))].sort();$("#holding-month").innerHTML=months.map(x=>`<option value="${x}">${x.slice(0,7)}</option>`).join("");$("#holding-month").value=months.at(-1);
const BT_SERIES=[
 {label:"Top 5 égal",key:"alpha_top5_return"},
 {label:"Legacy · snapshot 13/07",key:"legacy_return"},
 {label:"SPY",key:"spy_return"}
];
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const sampleStd=a=>{if(a.length<2)return null;const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/(a.length-1))};
const quantile=(a,q)=>{if(!a.length)return null;const s=[...a].sort((x,y)=>x-y),p=(s.length-1)*q,l=Math.floor(p),h=Math.ceil(p);return s[l]+(s[h]-s[l])*(p-l)};
const covariance=(a,b)=>{if(a.length<2||a.length!==b.length)return null;const am=mean(a),bm=mean(b);return a.reduce((s,x,i)=>s+(x-am)*(b[i]-bm),0)/(a.length-1)};
function drawdownEpisodes(rows,key){
 let wealth=1,peak=1,peakIndex=0,current=null;const episodes=[];
 rows.forEach((row,i)=>{
  wealth*=1+row[key];
  if(wealth>=peak){
   if(current){current.recovery=i;current.recoveryMonth=row.holding_month;current.duration=i-current.peakIndex;episodes.push(current);current=null}
   peak=wealth;peakIndex=i;return;
  }
  const dd=wealth/peak-1;
  if(!current)current={peakIndex,peakMonth:rows[peakIndex].holding_month,troughIndex:i,troughMonth:row.holding_month,depth:dd,recovery:null,recoveryMonth:null,duration:null};
  if(dd<current.depth){current.depth=dd;current.troughIndex=i;current.troughMonth=row.holding_month}
 });
 if(current){current.duration=rows.length-1-current.peakIndex;episodes.push(current)}
 return episodes.sort((a,b)=>a.depth-b.depth);
}
function periodStats(rows,spec){
 const r=rows.map(x=>x[spec.key]),b=rows.map(x=>x.spy_return),n=r.length,wealth=r.reduce((w,x)=>w*(1+x),1),years=n/12;
 const cagr=n&&wealth>0?Math.pow(wealth,1/years)-1:null,vol=(sampleStd(r)??0)*Math.sqrt(12),downside=Math.sqrt(mean(r.map(x=>Math.min(0,x)**2))??0)*Math.sqrt(12);
 const episodes=drawdownEpisodes(rows,spec.key),maxDD=episodes[0]?.depth??0,excess=r.map((x,i)=>x-b[i]),tracking=(sampleStd(excess)??0)*Math.sqrt(12),bstd=sampleStd(b),bvar=bstd==null?null:bstd*bstd,cov=covariance(r,b),beta=bvar?cov/bvar:null;
 const rfm=Math.pow(1.02,1/12)-1,alpha=beta==null?null:12*((mean(r)-rfm)-beta*(mean(b)-rfm)),rstd=sampleStd(r),corr=(rstd&&bstd)?cov/(rstd*bstd):null;
 const m=mean(r),m2=mean(r.map(x=>(x-m)**2)),skew=m2?mean(r.map(x=>(x-m)**3))/Math.pow(m2,1.5):null,kurt=m2?mean(r.map(x=>(x-m)**4))/(m2*m2)-3:null;
 const var95=quantile(r,.05),tail=r.filter(x=>x<=var95),gains=r.filter(x=>x>0).reduce((s,x)=>s+x,0),losses=Math.abs(r.filter(x=>x<0).reduce((s,x)=>s+x,0));
 const up=rows.filter(x=>x.spy_return>0),down=rows.filter(x=>x.spy_return<0),best=rows.reduce((a,x)=>x[spec.key]>a[spec.key]?x:a,rows[0]),worst=rows.reduce((a,x)=>x[spec.key]<a[spec.key]?x:a,rows[0]);
 return {label:spec.label,key:spec.key,n,total:wealth-1,cagr,volatility:vol,sharpe:vol?(cagr-.02)/vol:null,sortino:downside?(cagr-.02)/downside:null,calmar:maxDD?cagr/Math.abs(maxDD):null,maxDD,informationRatio:tracking?12*mean(excess)/tracking:null,beta,alpha,corr,hitSpy:mean(rows.map(x=>x[spec.key]>x.spy_return?1:0)),var95,cvar95:mean(tail),positive:mean(r.map(x=>x>0?1:0)),skew,kurt,omega:losses?gains/losses:null,upCapture:up.length?mean(up.map(x=>x[spec.key]))/mean(up.map(x=>x.spy_return)):null,downCapture:down.length?mean(down.map(x=>x[spec.key]))/mean(down.map(x=>x.spy_return)):null,best,worst,episodes};
}
function canonicalPeriodStats(rows){
 const key=rows[0].holding_month+"|"+rows.at(-1).holding_month,values=D.period_metrics[key];
 return BT_SERIES.map((spec,i)=>Object.assign({label:spec.label,key:spec.key,n:rows.length},Object.fromEntries(D.period_metric_fields.map((field,j)=>[field,values[i][j]]))));
}
function selectedBacktestRows(){
 const start=$("#bt-start").value,end=$("#bt-end").value;
 return D.monthly.filter(x=>x.holding_month>=start&&x.holding_month<=end);
}
function wealthFor(rows){
 const state=Object.fromEntries(BT_SERIES.map(s=>[s.label,{wealth:1,peak:1}]));
 return rows.map(row=>{const out={month:row.holding_month};BT_SERIES.forEach(s=>{const x=state[s.label];x.wealth*=1+row[s.key];x.peak=Math.max(x.peak,x.wealth);out[s.label]={wealth:x.wealth,drawdown:x.wealth/x.peak-1}});return out});
}
function drawBacktestLine(id,field,rows){
 const data=wealthFor(rows),el=$("#"+id);if(!data.length){el.innerHTML="<p class=\"fine\">Aucune donnée.</p>";return}
 const w=Math.max(700,el.clientWidth||900),h=310,p={l:54,r:18,t:16,b:34},series=BT_SERIES.map(x=>x.label),vals=data.flatMap(r=>series.map(s=>r[s][field]));
 let min=Math.min(...vals),max=Math.max(...vals);if(field==="wealth")min=Math.min(0,min);if(min===max){min-=.1;max+=.1}
 const x=i=>p.l+i*(w-p.l-p.r)/Math.max(1,data.length-1),y=v=>p.t+(max-v)*(h-p.t-p.b)/(max-min);
 let svg="<svg viewBox=\"0 0 "+w+" "+h+"\">";
 for(let j=0;j<5;j++){const v=min+(max-min)*j/4,yy=y(v);svg+="<line class=\"gridline\" x1=\""+p.l+"\" y1=\""+yy+"\" x2=\""+(w-p.r)+"\" y2=\""+yy+"\"/><text x=\"4\" y=\""+(yy+3)+"\">"+(field==="wealth"?v.toFixed(1):pct(v,0))+"</text>"}
 series.forEach(s=>{const pts=data.map((r,i)=>x(i)+","+y(r[s][field])).join(" ");svg+="<polyline points=\""+pts+"\" fill=\"none\" stroke=\""+color[s]+"\" stroke-width=\""+(s==="Top 5 égal"?2.5:1.7)+"\"/>"});
 [0,Math.floor(data.length/2),data.length-1].forEach(i=>svg+="<text x=\""+x(i)+"\" y=\""+(h-8)+"\" text-anchor=\""+(i===0?"start":i===data.length-1?"end":"middle")+"\">"+data[i].month.slice(0,7)+"</text>");el.innerHTML=svg+"</svg>";
}
function rollingValue(slice,spec,kind){
 const st=canonicalPeriodStats(slice).find(x=>x.key===spec.key);
 if(kind==="excess_spy")return 12*mean(slice.map(x=>x[spec.key]-x.spy_return));
 if(kind==="excess_legacy")return 12*mean(slice.map(x=>x[spec.key]-x.legacy_return));
 if(kind==="volatility")return st.annualized_volatility;
 if(kind==="drawdown")return st.max_drawdown;
 return st.sharpe;
}
function drawRolling(rows){
 const win=Number($("#rolling-window").value),kind=$("#rolling-metric").value,points=[];
 for(let i=win-1;i<rows.length;i++){const slice=rows.slice(i-win+1,i+1),values={};BT_SERIES.forEach(s=>values[s.label]=rollingValue(slice,s,kind));points.push({month:rows[i].holding_month,values})}
 const el=$("#rolling-chart");if(!points.length){el.innerHTML="<p class=\"range-warning\">La période est plus courte que la fenêtre glissante.</p>";return}
 const w=Math.max(700,el.clientWidth||900),h=310,p={l:58,r:18,t:16,b:34},series=BT_SERIES.map(x=>x.label),vals=points.flatMap(x=>series.map(s=>x.values[s])).filter(Number.isFinite);
 let min=Math.min(...vals),max=Math.max(...vals);if(min===max){min-=.1;max+=.1}const x=i=>p.l+i*(w-p.l-p.r)/Math.max(1,points.length-1),y=v=>p.t+(max-v)*(h-p.t-p.b)/(max-min);
 let svg="<svg viewBox=\"0 0 "+w+" "+h+"\">";for(let j=0;j<5;j++){const v=min+(max-min)*j/4,yy=y(v);svg+="<line class=\"gridline\" x1=\""+p.l+"\" y1=\""+yy+"\" x2=\""+(w-p.r)+"\" y2=\""+yy+"\"/><text x=\"3\" y=\""+(yy+3)+"\">"+(kind==="sharpe"?num(v,1):pct(v,0))+"</text>"}
 series.forEach(s=>{const pts=points.map((r,i)=>x(i)+","+y(r.values[s])).join(" ");svg+="<polyline points=\""+pts+"\" fill=\"none\" stroke=\""+color[s]+"\" stroke-width=\""+(s==="Top 5 égal"?2.5:1.5)+"\"/>"});[0,Math.floor(points.length/2),points.length-1].forEach(i=>svg+="<text x=\""+x(i)+"\" y=\""+(h-8)+"\" text-anchor=\""+(i===0?"start":i===points.length-1?"end":"middle")+"\">"+points[i].month.slice(0,7)+"</text>");el.innerHTML=svg+"</svg>";
}
const ATTR_METHODS=[
 {id:"alpha_top5_equal",label:"XGBoost Top 5 égal"},
 {id:"alpha_top10_equal",label:"XGBoost Top 10 égal"},
 {id:"Combined_Frequency",label:"Legacy · snapshot 13/07"},
 {id:"SPY total return",label:"SPY total return"}
];
let attrAxisMode="actions",attrTableRows=[];
$("#attr-strategy").innerHTML=ATTR_METHODS.map(x=>`<option value="${x.id}">${x.label}</option>`).join("");
function attributionBaseRows(){const start=$("#bt-start").value,end=$("#bt-end").value,strategy=$("#attr-strategy").value;return D.attribution.filter(x=>x.s===strategy&&x.m>=start&&x.m<=end)}
function updateAttributionScopes(){
 const rows=attributionBaseRows(),months=[...new Set(rows.map(x=>x.m))].sort(),years=[...new Set(months.map(x=>x.slice(0,4)))],previous=$("#attr-scope").value;
 $("#attr-scope").innerHTML='<option value="all">Toute la période sélectionnée</option><optgroup label="Années">'+years.map(y=>`<option value="y:${y}">${y}</option>`).join("")+'</optgroup><optgroup label="Mois">'+months.map(m=>`<option value="m:${m}">${m.slice(0,7)}</option>`).join("")+'</optgroup>';
 if([...$("#attr-scope").options].some(x=>x.value===previous))$("#attr-scope").value=previous;$("#attr-scope-wrap").style.display=attrAxisMode==="actions"?"inline":"none";
}
function groupedAttribution(){
 const base=attributionBaseRows(),allMonths=[...new Set(base.map(x=>x.m))].sort(),scope=$("#attr-scope").value;let rows=base,scale=12/Math.max(1,allMonths.length),terminalLabel="CAGR période",title="Contributions par action";
 if(attrAxisMode==="actions"){
  if(scope.startsWith("y:")){const y=scope.slice(2);rows=base.filter(x=>x.m.startsWith(y));scale=12/Math.max(1,new Set(rows.map(x=>x.m)).size);terminalLabel="CAGR "+y;title="Actions · "+y}
  else if(scope.startsWith("m:")){const m=scope.slice(2);rows=base.filter(x=>x.m===m);scale=1;terminalLabel="Rendement "+m.slice(0,7);title="Actions · "+m.slice(0,7)}
  const groups=new Map();rows.forEach(x=>{const q=groups.get(x.c)||{name:x.c,type:x.k,value:0,simple:0,months:new Set(),weights:[]};q.value+=(x.l||0)*scale;q.simple+=x.v||0;q.months.add(x.m);if(Number.isFinite(x.w))q.weights.push(x.w);groups.set(x.c,q)});return finalizeAttribution([...groups.values()],rows,scale,terminalLabel,title,false)
 }
 const field=attrAxisMode==="years"?x=>x.m.slice(0,4):x=>x.m.slice(0,7),groups=new Map();base.forEach(x=>{const name=field(x),q=groups.get(name)||{name,type:attrAxisMode==="years"?"year":"month",value:0,simple:0,months:new Set(),weights:[]};q.value+=(x.l||0)*scale;q.simple+=x.v||0;q.months.add(x.m);groups.set(name,q)});const items=[...groups.values()].sort((a,b)=>a.name.localeCompare(b.name));return finalizeAttribution(items,base,scale,"CAGR période",attrAxisMode==="years"?"Contribution de chaque année au CAGR":"Contribution de chaque mois au CAGR",true)
}
function finalizeAttribution(items,rows,scale,terminalLabel,title,chronological){
 const logTotal=items.reduce((s,x)=>s+x.value,0),terminal=Math.expm1(logTotal),monthly=new Map();rows.forEach(x=>monthly.set(x.m,(monthly.get(x.m)||0)+(x.v||0)));const periodReturn=[...monthly.values()].reduce((w,r)=>w*(1+r),1)-1;
 items.forEach(x=>{x.monthCount=x.months.size;x.averageWeight=x.weights.length?mean(x.weights):null;x.marginal=terminal-Math.expm1(logTotal-x.value)});if(!chronological)items.sort((a,b)=>b.value-a.value||a.name.localeCompare(b.name));return {items,rows,scale,logTotal,terminal,terminalLabel,title,periodReturn,monthCount:monthly.size,chronological}
}
function compactWaterfallItems(view){
 if(view.chronological||view.items.length<=32)return view.items;const positive=view.items.filter(x=>x.value>=0).sort((a,b)=>b.value-a.value).slice(0,15),negative=view.items.filter(x=>x.value<0).sort((a,b)=>a.value-b.value).slice(0,15),kept=new Set([...positive,...negative]),other=view.items.filter(x=>!kept.has(x));const items=[...positive];if(other.length)items.push({name:`Autres (${other.length})`,type:"aggregate",value:other.reduce((s,x)=>s+x.value,0),simple:other.reduce((s,x)=>s+x.simple,0),monthCount:new Set(other.flatMap(x=>[...x.months])).size,averageWeight:null,marginal:null});return [...items,...negative]
}
function drawAttributionWaterfall(view){
 const items=compactWaterfallItems(view),bridge=view.terminal-view.logTotal,steps=[...items,{name:"Effet composé",type:"bridge",value:bridge}],cumulative=[0];steps.forEach(x=>cumulative.push(cumulative.at(-1)+x.value));const values=[0,...cumulative,view.terminal],min=Math.min(...values),max=Math.max(...values),pad=Math.max(.01,(max-min)*.14),lo=min-pad,hi=max+pad,w=Math.max(900,(steps.length+1)*72),h=430,p={l:62,r:28,t:28,b:112},x=i=>p.l+i*(w-p.l-p.r)/(steps.length+1),y=v=>p.t+(hi-v)*(h-p.t-p.b)/(hi-lo||1),bar=38;
 let svg=`<svg viewBox="0 0 ${w} ${h}" style="width:${w}px">`;for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4,yy=y(v);svg+=`<line class="gridline" x1="${p.l}" y1="${yy}" x2="${w-p.r}" y2="${yy}"/><text x="4" y="${yy+3}">${pct(v,1)}</text>`}svg+=`<line x1="${p.l}" y1="${y(0)}" x2="${w-p.r}" y2="${y(0)}" stroke="var(--muted)"/>`;
 steps.forEach((q,i)=>{const before=cumulative[i],after=cumulative[i+1],top=y(Math.max(before,after)),bottom=y(Math.min(before,after)),fill=q.type==="bridge"?"#26387E":q.value>=0?"#16794B":"#B03A45",xx=x(i);svg+=`<line x1="${xx+bar/2}" y1="${y(after)}" x2="${x(i+1)-bar/2}" y2="${y(after)}" stroke="var(--muted)" stroke-dasharray="3 3"/><rect x="${xx-bar/2}" y="${top}" width="${bar}" height="${Math.max(2,bottom-top)}" fill="${fill}"><title>${q.name}: ${pct(q.value,3)}</title></rect>`;if(steps.length<45)svg+=`<text class="wf-value" x="${xx}" y="${top-6}" text-anchor="middle">${pct(q.value,1)}</text>`;svg+=`<text transform="translate(${xx},${h-p.b+18}) rotate(-48)" text-anchor="end">${q.name.replace('.US','').slice(0,20)}</text>`});const tx=x(steps.length),topy=y(Math.max(0,view.terminal)),bottomy=y(Math.min(0,view.terminal));svg+=`<rect x="${tx-bar/2}" y="${topy}" width="${bar}" height="${Math.max(2,bottomy-topy)}" fill="#9B8816"><title>${view.terminalLabel}: ${pct(view.terminal,4)}</title></rect><text class="wf-value" x="${tx}" y="${topy-7}" text-anchor="middle">${pct(view.terminal,1)}</text><text transform="translate(${tx},${h-p.b+18}) rotate(-48)" text-anchor="end">${view.terminalLabel}</text></svg>`;$("#attr-waterfall").innerHTML=svg;
}
function renderAttributionTable(){const q=$("#attr-search").value.toLowerCase(),rows=attrTableRows.filter(x=>!q||x.name.toLowerCase().includes(q));$("#attr-rows").innerHTML=rows.map(x=>`<tr><td>${x.name.replace('.US','')}</td><td>${x.type}</td>${td(x.monthCount)}${td(pct(x.value,3))}${td(pct(x.marginal,3))}${td(pct(x.simple,2))}${td(pct(x.averageWeight,2))}</tr>`).join("")}
function renderAttribution(){
 updateAttributionScopes();const view=groupedAttribution(),method=ATTR_METHODS.find(x=>x.id===$("#attr-strategy").value);attrTableRows=view.items;$("#attr-metrics").innerHTML=metric(view.terminalLabel,pct(view.terminal,2),`${method.label} · réconcilié`)+metric("Log annualisé additif",pct(view.logTotal,2),"somme exacte du waterfall")+metric("Rendement composé",pct(view.periodReturn,1),`${view.monthCount} mois dans le scope`)+metric("Composantes",view.items.length,attrAxisMode==="actions"?"actions + coûts":"périodes calendaires");$("#attr-table-title").textContent=view.title;$("#attr-table-note").textContent=`${view.items.length} composantes exhaustives ; graphique compacté seulement au-delà de 32 actions.`;drawAttributionWaterfall(view);renderAttributionTable();
}
$("#attr-strategy").onchange=()=>{updateAttributionScopes();renderAttribution()};$("#attr-scope").onchange=renderAttribution;$("#attr-search").oninput=renderAttributionTable;$$('#attr-axis button').forEach(b=>b.onclick=()=>{$$('#attr-axis button').forEach(x=>x.classList.remove('active'));b.classList.add('active');attrAxisMode=b.dataset.axis;updateAttributionScopes();renderAttribution()});
function renderBacktest(){
 const rows=selectedBacktestRows();if(!rows.length)return;const canonical=canonicalPeriodStats(rows),distribution=BT_SERIES.map(s=>periodStats(rows,s)),stats=canonical.map((x,i)=>({...distribution[i],...x})),top=stats[0],legacy=stats[1],spyStats=stats[2],fragile=rows.length<36;
 const currentRows=D.legacy_current_monthly.filter(x=>x.holding_month>=rows[0].holding_month&&x.holding_month<=rows.at(-1).holding_month),currentLegacy=periodStats(currentRows,{label:"Legacy · snapshot 27/07",key:"legacy_return"}),currentSpy=periodStats(currentRows,{label:"SPY total return",key:"spy_return"});
 $("#bt-period-status").innerHTML=rows[0].holding_month.slice(0,7)+" → "+rows.at(-1).holding_month.slice(0,7)+" · "+rows.length+" mois"+(fragile?" · <span class=\"range-warning\">fenêtre courte</span>":"");
 $("#snapshot-reconciliation").innerHTML="<strong>Deux snapshots Legacy, une convention SPY explicite</strong>Comparaison Boosting : Legacy snapshot 13/07 <b>"+pct(legacy.cagr,2)+"</b>. Dernier replay Legacy validé, snapshot 27/07 : <b>"+pct(currentLegacy.cagr,2)+"</b> sur les mêmes "+currentRows.length+" mois. SPY total return (<code>adjusted_close</code>) : <b>"+pct(currentSpy.cagr,2)+"</b>. Le SPY price return (<code>close</code>) n'est pas utilisé comme benchmark de performance.";
 $("#backtest-metrics").innerHTML=metric("Top 5 · CAGR",pct(top.cagr),"Legacy "+pct(legacy.cagr)+" · SPY "+pct(spyStats.cagr))+metric("Top 5 · Sharpe",num(top.sharpe),"Legacy "+num(legacy.sharpe)+" · "+(fragile?"fragile":"fenêtre exploitable"))+metric("Top 5 · Max DD",pct(top.maxDD),"Legacy "+pct(legacy.maxDD)+" · SPY "+pct(spyStats.maxDD))+metric("Information ratio",num(top.informationRatio),"Top 5 contre SPY");
 $("#advanced-rows").innerHTML=stats.map(x=>"<tr><td>"+x.label+"<span class=\"subtle\">"+x.n+" mois</span></td>"+td(pct(x.total_return))+td(pct(x.cagr))+td(pct(x.annualized_volatility))+td(num(x.sharpe))+td(num(x.sortino))+td(num(x.calmar))+td(pct(x.max_drawdown))+td(num(x.information_ratio))+td(num(x.beta))+td(pct(x.alpha))+td(num(x.correlation))+td(pct(x.benchmark_hit_rate))+td(pct(x.var_95))+td(pct(x.cvar_95))+"</tr>").join("");
 $("#distribution-rows").innerHTML=stats.map(x=>"<tr><td>"+x.label+"</td>"+td(x.best.holding_month.slice(0,7)+" · "+pct(x.best[x.key]))+td(x.worst.holding_month.slice(0,7)+" · "+pct(x.worst[x.key]))+td(pct(x.positive))+td(num(x.skew))+td(num(x.kurt))+td(num(x.omega))+td(pct(x.upCapture))+td(pct(x.downCapture))+"</tr>").join("");
 drawBacktestLine("wealth-chart","wealth",rows);drawBacktestLine("dd-chart","drawdown",rows);drawRolling(rows);
 renderAttribution();
 $("#drawdown-episodes").innerHTML=stats.map(x=>"<article class=\"panel\"><h3>"+x.label+"</h3><div class=\"episodes\">"+(x.episodes.length?x.episodes.slice(0,3).map(e=>"<div class=\"episode\"><span>"+e.peakMonth.slice(0,7)+" → "+e.troughMonth.slice(0,7)+"</span><b>"+pct(e.depth)+"</b><span>"+(e.recoveryMonth?"récup. "+e.recoveryMonth.slice(0,7):"non récupéré")+"</span><span>"+e.duration+" mois</span></div>").join(""):"<p class=\"fine\">Aucun drawdown.</p>")+"</div></article>").join("");
 const startYear=Number(rows[0].holding_month.slice(0,4)),endYear=Number(rows.at(-1).holding_month.slice(0,4)),annual=D.annual.filter(x=>x.months===12&&x.year>=startYear&&x.year<=endYear);
 $("#annual-table").innerHTML="<thead><tr><th>Année complète</th><th>Boosting Top 5</th><th>Legacy · snapshot 13/07</th><th>SPY total return</th></tr></thead><tbody>"+annual.map(x=>`<tr><td>${x.year}</td>${td(pct(x.alpha_top5_equal))}${td(pct(x.Legacy))}${td(pct(x['SPY total return']))}</tr>`).join("")+"</tbody>";
}
const btMonths=D.monthly.map(x=>x.holding_month),btOptions=btMonths.map(x=>"<option value=\""+x+"\">"+x.slice(0,7)+"</option>").join("");
$("#bt-start").innerHTML=btOptions;$("#bt-end").innerHTML=btOptions;$("#bt-start").value=btMonths[0];$("#bt-end").value=btMonths.at(-1);
function resetPreset(){$$(".preset").forEach(x=>x.classList.remove("active"))}
$("#bt-start").onchange=()=>{if($("#bt-start").value>$("#bt-end").value)$("#bt-end").value=$("#bt-start").value;resetPreset();renderBacktest()};
$("#bt-end").onchange=()=>{if($("#bt-end").value<$("#bt-start").value)$("#bt-start").value=$("#bt-end").value;resetPreset();renderBacktest()};
$$(".preset").forEach(b=>b.onclick=()=>{resetPreset();b.classList.add("active");const n=b.dataset.period;$("#bt-end").value=btMonths.at(-1);$("#bt-start").value=n==="full"?btMonths[0]:btMonths[Math.max(0,btMonths.length-Number(n))];renderBacktest()});
$("#rolling-window").onchange=renderBacktest;$("#rolling-metric").onchange=renderBacktest;
$("#wealth-legend").innerHTML=BT_SERIES.map(s=>"<span><i style=\"background:"+color[s.label]+"\"></i>"+s.label+"</span>").join("");
renderBacktest();

function portfolioCard(title,portfolio,month,cls=""){
 const current=D.holdings.filter(x=>x.holding_month===month&&x.portfolio===portfolio).sort((a,b)=>(a.rank??99)-(b.rank??99)),idx=months.indexOf(month),prev=idx>0?new Set(D.holdings.filter(x=>x.holding_month===months[idx-1]&&x.portfolio===portfolio).map(x=>x.ticker)):new Set(),now=new Set(current.map(x=>x.ticker)),exits=idx>0?D.holdings.filter(x=>x.holding_month===months[idx-1]&&x.portfolio===portfolio&&!now.has(x.ticker)).map(x=>x.ticker):[];
 return `<article class="panel portfolio-card ${cls}"><h2>${title}</h2>${current.map(x=>`<div class="ticker-row"><span class="ticker">${x.ticker.replace(".US","")}</span><span>${x.sector||"—"}<br><span class="fine">${x.rank?`rang ${x.rank} · `:""}${x.calibrated_probability==null?"":`p ${pct(x.calibrated_probability,0)} · `}réalisé ${pct(x.realized_return_1m)}</span></span><span><b>${pct(x.weight,0)}</b><br><span class="action ${prev.has(x.ticker)?"keep":"enter"}">${prev.has(x.ticker)?"KEEP":"ENTER"}</span></span></div>`).join("")}<p class="fine">Sorties : ${exits.length?exits.map(x=>x.replace(".US","")).join(", "):"aucune"}</p></article>`;
}
function renderPortfolios(){const m=$("#holding-month").value,r=D.monthly.find(x=>x.holding_month===m);$("#month-returns").textContent=`Top 5 ${pct(r?.alpha_top5_return)} · Top 10 ${pct(r?.alpha_top10_return)} · Legacy ${pct(r?.legacy_return)} · SPY ${pct(r?.spy_return)}`;$("#portfolio-cards").innerHTML=portfolioCard("Legacy","Legacy publié",m,"legacy")+portfolioCard("Alpha Top 5","Alpha Top 5 égal",m)+portfolioCard("Alpha Top 10","Alpha Top 10 égal",m)}
$("#holding-month").onchange=renderPortfolios;renderPortfolios();

const shapFolds=[...new Set(D.shap_month_counts.map(x=>x.fold))].sort((a,b)=>a-b);$("#shap-fold").innerHTML+=shapFolds.map(x=>`<option value="${x}">Fold ${x}</option>`).join("");
$("#shap-feature").innerHTML=D.features.map((f,i)=>`<option value="${i}">${i+1}. ${f}</option>`).join("");
const shapMonthCache=new Map();let shapRenderToken=0;
function updateShapMonths(){const f=$("#shap-fold").value,months=D.shap_month_counts.filter(x=>f==="all"||x.fold===Number(f));$("#shap-month").innerHTML='<option value="all">Tous les mois · échantillon global</option>'+months.map(x=>`<option value="${x.decision_month}">${x.decision_month.slice(0,7)} · ${x.rows} actions</option>`).join("")}
async function loadShapMonth(month){
 if(shapMonthCache.has(month))return shapMonthCache.get(month);
 const response=await fetch(new URL(`shap/${month.slice(0,7)}.json.gz`,location.href));if(!response.ok)throw new Error(`Fichier SHAP ${month.slice(0,7)} indisponible (${response.status})`);
 const buffer=await response.arrayBuffer(),bytes=new Uint8Array(buffer);let text;
 if(bytes[0]===31&&bytes[1]===139){if(typeof DecompressionStream==="undefined")throw new Error("Décompression gzip non prise en charge par ce navigateur");const stream=new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));text=await new Response(stream).text()}else{text=new TextDecoder().decode(bytes)}
 const rows=JSON.parse(text).rows;shapMonthCache.set(month,rows);return rows;
}
async function shapSubset(){const m=$("#shap-month").value,f=$("#shap-fold").value,rows=m==="all"?D.shap:await loadShapMonth(m);return rows.filter(x=>(f==="all"||x.f===Number(f))&&(m==="all"||x.m===m))}
function importance(rows){return D.features.map((f,i)=>({f,i,v:rows.reduce((a,r)=>a+(r.s[i]==null?0:Math.abs(r.s[i])),0)/Math.max(1,rows.filter(r=>r.s[i]!=null).length)})).sort((a,b)=>b.v-a.v)}
function canvasSetup(c){const rect=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=Math.max(600,rect.width*dpr);c.height=Math.max(300,rect.height*dpr);const ctx=c.getContext("2d");ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height}}
function renderBeeswarm(rows,imp){
 const c=$("#beeswarm"),{ctx,w,h}=canvasSetup(c),left=185,right=20,top=20,bottom=28,topf=imp.slice(0,15),all=topf.flatMap(x=>rows.map(r=>r.s[x.i]).filter(Number.isFinite)),lim=Math.max(.001,...all.map(Math.abs));
 const styles=getComputedStyle(document.documentElement),ink=styles.getPropertyValue("--muted"),line=styles.getPropertyValue("--line");ctx.clearRect(0,0,w,h);ctx.font='10px "IBM Plex Mono"';ctx.fillStyle=ink;ctx.strokeStyle=line;
 const x=v=>left+(v+lim)*(w-left-right)/(2*lim),y=j=>top+j*(h-top-bottom)/14;ctx.beginPath();ctx.moveTo(x(0),top-8);ctx.lineTo(x(0),h-bottom+8);ctx.stroke();
 topf.forEach((q,j)=>{const vals=rows.map(r=>r.v[q.i]).filter(Number.isFinite),mn=Math.min(...vals),mx=Math.max(...vals);ctx.fillStyle=ink;ctx.textAlign="right";ctx.fillText(q.f.slice(0,27),left-8,y(j)+3);rows.forEach((r,k)=>{const sv=r.s[q.i],fv=r.v[q.i];if(!Number.isFinite(sv))return;const z=Number.isFinite(fv)?(fv-mn)/(mx-mn||1):.5;ctx.fillStyle=`rgb(${Math.round(55+180*z)},${Math.round(98-30*z)},${Math.round(190-90*z)})`;ctx.globalAlpha=.66;ctx.beginPath();ctx.arc(x(sv),y(j)+((k*37)%13-6)*.65,2.5,0,Math.PI*2);ctx.fill()})});ctx.globalAlpha=1;ctx.fillStyle=ink;ctx.textAlign="center";ctx.fillText("- contribution",left+50,h-7);ctx.fillText("+ contribution",w-right-50,h-7);
}
function renderIndividual(rows,idx){
 const c=$("#individual"),{ctx,w,h}=canvasSetup(c),p={l:62,r:22,t:24,b:42},pts=rows.map(r=>({x:r.v[idx],y:r.s[idx],t:r.t,m:r.m})).filter(x=>Number.isFinite(x.x)&&Number.isFinite(x.y));if(!pts.length)return;
 let xmin=Math.min(...pts.map(x=>x.x)),xmax=Math.max(...pts.map(x=>x.x)),ymin=Math.min(...pts.map(x=>x.y)),ymax=Math.max(...pts.map(x=>x.y));const xx=x=>p.l+(x-xmin)*(w-p.l-p.r)/(xmax-xmin||1),yy=y=>p.t+(ymax-y)*(h-p.t-p.b)/(ymax-ymin||1),styles=getComputedStyle(document.documentElement),ink=styles.getPropertyValue("--muted"),line=styles.getPropertyValue("--line");ctx.clearRect(0,0,w,h);ctx.strokeStyle=line;ctx.fillStyle=ink;ctx.font='10px "IBM Plex Mono"';
 for(let j=0;j<5;j++){const y=p.t+j*(h-p.t-p.b)/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();const v=ymax-j*(ymax-ymin)/4;ctx.fillText(v.toFixed(3),5,y+3)}
 pts.forEach(q=>{ctx.fillStyle="#26387E";ctx.globalAlpha=.58;ctx.beginPath();ctx.arc(xx(q.x),yy(q.y),3,0,Math.PI*2);ctx.fill()});ctx.globalAlpha=1;ctx.fillStyle=ink;ctx.textAlign="center";ctx.fillText(`Valeur · ${xmin.toFixed(3)} → ${xmax.toFixed(3)}`,w/2,h-10);$("#individual-title").textContent=`Analyse individuelle · ${D.features[idx]}`;
}
async function renderShap(){
 const token=++shapRenderToken,m=$("#shap-month").value;if(m!=="all")$("#shap-metrics").innerHTML=metric("Chargement",m.slice(0,7),"détail exhaustif du mois");
 let rows;try{rows=await shapSubset()}catch(error){if(token===shapRenderToken)$("#shap-metrics").innerHTML='<div class="callout bad"><strong>Chargement SHAP impossible</strong>'+error.message+'</div>';return}if(token!==shapRenderToken)return;
 const imp=importance(rows),folds=[...new Set(rows.map(x=>x.f))],idx=Number($("#shap-feature").value||imp[0].i),foldMeta=folds.length===1?D.folds.find(x=>x.fold===folds[0]):null;if(m!=="all"&&!$("#shap-feature").dataset.manual){$("#shap-feature").value=imp[0].i}
 $("#shap-metrics").innerHTML=metric("Observations SHAP",rows.length,m==="all"?"échantillon global · 80 par fold":"toutes les actions scorées du mois")+metric("Test expliqué",foldMeta?`${foldMeta.test_start.slice(0,7)} → ${foldMeta.test_end.slice(0,7)}`:folds.join(", "),foldMeta?`train jusqu'au ${foldMeta.train_cutoff.slice(0,7)}`:"15 modèles OOS")+metric("Variables",D.features.length,"EMA uniquement")+metric("|SHAP| n°1",num(imp[0].v,4),imp[0].f);
 const mx=imp[0].v||1;$("#shap-bars").innerHTML=imp.slice(0,20).map((x,j)=>`<div class="shap-bar"><span>${j+1}. ${x.f.slice(0,28)}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${100*x.v/mx}%"></span></span><b>${num(x.v,4)}</b></div>`).join("");
 renderBeeswarm(rows,imp);renderIndividual(rows,Number($("#shap-feature").value));const fi=Number($("#shap-feature").value);$("#shap-detail-note").textContent=`${rows.length} observations ${m==="all"?"échantillonnées":"exhaustives"} ; variable ${D.features[fi]}`;$("#shap-detail").innerHTML=rows.slice().sort((a,b)=>Math.abs(b.s[fi]||0)-Math.abs(a.s[fi]||0)).map(r=>`<tr><td>${r.m.slice(0,7)}</td><td>${r.t}</td>${td(r.f)}${td(num(r.v[fi],5))}${td(num(r.s[fi],5))}</tr>`).join("");
}
$("#shap-fold").onchange=()=>{updateShapMonths();$("#shap-feature").dataset.manual="";renderShap()};$("#shap-month").onchange=()=>{$("#shap-feature").dataset.manual="";renderShap()};$("#shap-feature").onchange=()=>{$("#shap-feature").dataset.manual="1";renderShap()};updateShapMonths();
function renderLexicon(){const q=$("#lexicon-search").value.toLowerCase();$("#lexicon-rows").innerHTML=D.lexicon.filter(x=>!q||Object.values(x).join(" ").toLowerCase().includes(q)).map(x=>`<tr>${td(x.importance_rank)}<td><code>${x.feature}</code></td>${td(num(x.mean_abs_shap,5))}<td>${x.transformation}</td><td>${x.ema_numerator_span_days} / ${x.ema_denominator_span_days}</td><td>${x.unit}</td><td>${x.exact_definition}</td><td>${x.economic_interpretation}</td></tr>`).join("")}$("#lexicon-search").oninput=renderLexicon;renderLexicon();

$("#risk-model-rows").innerHTML=D.risk_models.map(r=>`<tr><td>${r.head}</td>${td(r.horizon+"m")}<td>${r.task_type}</td><td>${r.target}</td>${td(r.test_rows)}${td(num(r.monthly_spearman,3))}${td(num(r.rmse,3))}${td(num(r.normalized_rmse,3))}${td(num(r.mae,3))}${td(num(r.roc_auc,3))}${td(num(r.pr_auc_average_precision,3))}${td(num(r.pr_auc_lift_vs_prevalence,2))}</tr>`).join("");
$("#risk-perf-rows").innerHTML=D.risk_performance.map(r=>`<tr><td>${labels[r.series]||r.series.replaceAll("_"," ")}</td>${td(pct(r.cagr))}${td(pct(r.annualized_volatility))}${td(num(r.sharpe))}${td(pct(r.max_drawdown))}${td(`${r.worst_full_calendar_year} · ${pct(r.worst_full_calendar_year_return)}`)}</tr>`).join("");
$("#risk-gates").innerHTML=D.risk_gates.map(r=>`<article class="panel"><h3>${r.strategy.replaceAll("_"," ")}</h3>${Object.entries(r).filter(([k])=>k!=="strategy").map(([k,v])=>`<p>${badge(v)} ${k.replaceAll("_"," ")}</p>`).join("")}</article>`).join("");

const live=D.live.manifest,vm=live.validation_metrics;
$("#live-metrics").innerHTML=metric("ROC AUC validation",num(vm.roc_auc,3),"2025-07 → 2025-12")+metric("PR AUC",num(vm.pr_auc_average_precision,3),`prévalence ${pct(vm.positive_rate)}`)+metric("Lift PR",num(vm.pr_auc_lift_vs_prevalence,2),"contre prévalence")+metric("Univers live",live.live_universe_rows,`${live.holding_month_universe_filter.rows_before-live.holding_month_universe_filter.rows_after} retraits`);
function liveCard(title,rows,cls=""){return `<article class="panel portfolio-card ${cls}"><h2>${title}</h2>${rows.map(x=>{const price=x.last_close??x.price,weight=x.equal_weight??x.weight_normalized??x.weight;return `<div class="ticker-row"><span class="ticker">${String(x.ticker).replace(".US","")}</span><span>${x.company_name||x.sector||""}<br><span class="fine">${price==null?"":money(price)} ${x.calibrated_probability==null?"":`· p ${pct(x.calibrated_probability,0)}`}</span></span><b>${pct(weight,0)}</b></div>`}).join("")}</article>`}
$("#live-portfolios").innerHTML=liveCard("Legacy juillet",D.live.legacy,"legacy")+liveCard("Alpha Top 5",D.live.top5)+liveCard("Alpha Top 10",D.live.top10);
const lineage=[["Statut",live.status],["Décision",live.decision_month],["Détention",live.holding_month],["Train",`${live.train_start} → ${live.train_end} · ${live.train_rows} lignes`],["Validation",`${live.validation_start} → ${live.validation_end} · ${live.validation_rows} lignes`],["Maturité labels",live.label_maturity_cutoff],["Features",`${live.feature_count} · ${live.winner_pair_count} paires EMA`],["Tickers retirés",live.holding_month_universe_filter.removed_tickers.join(", ")]];
$("#live-lineage").innerHTML=lineage.map(x=>`<tr><th>${x[0]}</th><td class="mono">${x[1]}</td></tr>`).join("");
$("#sources").innerHTML=__SOURCES__.map(x=>`<p>${x.path}<br><span class="fine">${x.sha256}</span></p>`).join("");
</script>
</body></html>"""


def render(output_dir: Path) -> tuple[Path, Path]:
    payload, sources = build_payload()
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    shap_sidecars = _write_monthly_shap_sidecars(
        DIAGNOSTICS_DIR / "classification_h06/shap_samples.parquet",
        payload["features"],
        html_dir / "shap",
    )
    output_path = html_dir / "alpharank_research_center.html"
    manifest_path = output_dir / "manifest.json"
    source_manifest = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _hash(path),
            "bytes": path.stat().st_size,
        }
        for path in sources
    ]
    encoded = json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    page = HTML.replace("__PAYLOAD__", encoded).replace(
        "__SOURCES__", json.dumps(source_manifest, ensure_ascii=False)
    )
    output_path.write_text(page, encoding="utf-8")
    try:
        report_reference = str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        report_reference = str(output_path)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "report": report_reference,
                "status": "research_dashboard",
                "semantics": {
                    "historical_retraining": "once per outer fold",
                    "historical_legacy_snapshot": "20260713_201639",
                    "standalone_legacy_snapshot": "20260727_221253",
                    "performance_benchmark": "spy_total_return_adjusted_close",
                    "legacy_signal_benchmark": (
                        "spy_price_return_close; not used as the standard "
                        "performance benchmark"
                    ),
                    "monthly_shap_filter": (
                        "all test-month observations explained by that fold model"
                    ),
                    "live_retraining": "once per monthly execution",
                    "shap_unit": "raw XGBoost margin / log-odds",
                    "shap_sampling": (
                        "monthly views are exhaustive; all-month overview is "
                        "80 rows per fold"
                    ),
                    "cagr_attribution": (
                        "exact additive log-return allocation; "
                        "exp(sum(annualized log contributions)) - 1"
                    ),
                    "attribution_cost_treatment": (
                        "transaction costs are reported as a separate component"
                    ),
                },
                "counts": payload["meta"],
                "shap_sidecars": shap_sidecars,
                "data_lineage": payload["lineage"],
                "sources": source_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report, manifest = render(args.output_dir.resolve())
    print(report)
    print(manifest)


if __name__ == "__main__":
    main()
