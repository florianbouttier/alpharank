"""Calculate the data payload for the central research dashboard."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.portfolio.attribution import (
    portfolio_return_attribution,
    reference_return_attribution,
)
from alpharank.portfolio.lineage import (
    input_hashes_from_manifest,
    load_manifest,
    require_matching_data_contexts,
)
from alpharank.portfolio.performance import advanced_performance_statistics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOPN_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_ema_top5_vs_top10_quarantine_v7_20260726"
)
CHAMPION_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_ema_long_history_ticker_quarantine_v6_20260726"
)
DIAGNOSTICS_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_ema_fold_full_shap_20260810"
)
RISK_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726"
)
SCREENING_DIR = PROJECT_ROOT / ("outputs/multihorizon_boosting/screening_clean_20260725")
EMA_SCREENING_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_winners_pit_ema_only_20260725"
)
LIVE_DIR = PROJECT_ROOT / (
    "outputs/live_alpha/ema_classification_h6_202606_20260727_production_candidate_v3"
)
HISTORICAL_LEGACY_COMMON_DIR = PROJECT_ROOT / (
    "outputs/common_portfolio_replays/legacy_20260713_201639_spy_total_return"
)
CURRENT_LEGACY_COMMON_DIR = PROJECT_ROOT / (
    "outputs/common_portfolio_replays/legacy_20260727_221253_spy_total_return"
)
LATEST_OPEN_SOURCE_MANIFEST = PROJECT_ROOT / "data/open_source/official/manifests/latest_run.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/research_dashboard/legacy_ema_alpha_central_20260810_full_shap"
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
    comparison = monthly.select("holding_month", "legacy_return", "spy_return").join(
        common,
        on="holding_month",
        how="inner",
    )
    legacy_maximum_error = float(
        (comparison["legacy_return"] - comparison["source_legacy_return"]).abs().max()
    )
    spy_maximum_error = float(
        (comparison["spy_return"] - comparison["source_spy_return"]).abs().max()
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
            pl.read_parquet(CHAMPION_DIR / "classification_h06/predictions.parquet")[
                "decision_month"
            ].max()
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
            pl.col("strategy").is_in(["Combined_Frequency", "Combined_Equal", "SPY total return"])
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
        .filter(pl.col("strategy").is_in(["alpha_top5_equal", "alpha_top10_equal"]))
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
        .filter(pl.col("strategy").is_in(["alpha_top5_equal", "alpha_top10_equal"]))
        .with_columns(
            (pl.col("net_return") - pl.col("benchmark_return")).alias("active_return"),
            ((1.0 + pl.col("net_return")) / (1.0 + pl.col("benchmark_return")) - 1.0).alias(
                "relative_return"
            ),
        )
    )
    alpha_attribution = portfolio_return_attribution(
        alpha_holdings,
        alpha_monthly,
    )

    legacy_holdings = pl.read_parquet(
        HISTORICAL_LEGACY_COMMON_DIR / "legacy_common_total_return_holdings.parquet"
    ).filter(pl.col("strategy") == "Combined_Frequency")
    legacy_monthly = pl.read_csv(
        HISTORICAL_LEGACY_COMMON_DIR / "legacy_common_total_return_monthly.csv",
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
    ).filter(pl.col("holding_month").is_between(date(2011, 8, 1), date(2025, 11, 1)))
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
    samples = pl.read_parquet(samples_path).sort(["decision_month", "ticker"])
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
        samples.group_by("decision_month", "fold").len(name="rows").sort("decision_month")
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
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
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
        frame = monthly.filter(pl.col("holding_month").is_between(start, end))
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
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            volatility = math.sqrt(variance * 12)
            output.append(
                {
                    "period": label,
                    "series": series,
                    "months": len(values),
                    "cagr": cagr,
                    "volatility": volatility,
                    "sharpe": (cagr - 0.02) / volatility if volatility else None,
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
        "legacy": _records(LIVE_DIR / "legacy_portfolio_same_holding_month.csv"),
    }


def build_payload() -> tuple[dict[str, Any], list[Path]]:
    champion_manifest = load_manifest(CHAMPION_DIR / "manifest.json")
    champion_data_manifest = (
        _project_path(champion_manifest["config"]["data_dir"]).parent / "data_input_manifest.json"
    )
    live_manifest = load_manifest(LIVE_DIR / "manifest.json")
    live_data_manifest = (
        _project_path(live_manifest["config"]["data_dir"]).parent / "data_input_manifest.json"
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
        raise FileNotFoundError("Missing dashboard sources:\n" + "\n".join(map(str, missing)))
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
        "performance": _records(TOPN_DIR / "performance_legacy_convention.csv"),
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
        "ema_screening": _records(EMA_SCREENING_DIR / "model_horizon_summary.csv"),
        "champion": _records(CHAMPION_DIR / "model_horizon_summary.csv"),
        "folds": _fold_diagnostics(),
        "risk_models": _records(RISK_DIR / "risk_model_metrics.csv"),
        "risk_performance": _records(RISK_DIR / "allocation_performance_legacy_convention.csv"),
        "risk_gates": _records(RISK_DIR / "allocation_acceptance_gates.csv"),
        "shap": shap,
        "shap_month_counts": shap_month_counts,
        "lexicon": lexicon,
        "features": features,
        "live": _live_payload(),
    }
    return payload, source_files
