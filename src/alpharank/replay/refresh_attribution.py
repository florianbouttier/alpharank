"""Causal attribution tables for a refresh replay report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.replay.refresh_compare import TableSpec, compare_frames

MATERIALITY_TOLERANCE = 1e-12
PREDICTION_KEYS = ("method", "horizon", "decision_month", "ticker")
LEGACY_KEYS = ("strategy", "decision_month", "holding_month", "ticker")
PRICE_TABLES = frozenset({"final_price", "sp500_price"})
SEC_TABLES = frozenset({"income_statement", "balance_sheet", "cash_flow", "earnings"})


@dataclass(frozen=True, slots=True)
class ScenarioArtifacts:
    """Replay artifacts for one controlled data-family scenario."""

    name: str
    label: str
    legacy_run: Path
    boosting_run: Path
    common_status: str
    common_run: Path | None = None


@dataclass(frozen=True, slots=True)
class RefreshAttributionInputs:
    """Four-scenario inputs required to separate price and SEC effects."""

    audit_report: Path
    scenarios: tuple[ScenarioArtifacts, ...]
    focus_ticker: str = "CVC.US"
    focus_month: date = date(2016, 6, 1)


def build_refresh_attribution(inputs: RefreshAttributionInputs) -> dict[str, object]:
    """Build display-ready evidence without changing any replay artifact."""

    scenarios = _scenario_map(inputs.scenarios)
    audit = _read_object(inputs.audit_report)
    cutoff = date.fromisoformat(str(audit["historical_cutoff"]))
    legacy_frames = {
        name: _read_legacy(scenario.legacy_run, cutoff) for name, scenario in scenarios.items()
    }
    prediction_frames = {
        name: _read_predictions(scenario.boosting_run, cutoff)
        for name, scenario in scenarios.items()
    }
    baseline_legacy = legacy_frames["baseline"]
    baseline_predictions = prediction_frames["baseline"]
    legacy_comparisons = [
        _compare_legacy(baseline_legacy, legacy_frames[name], name)
        for name in ("price_only", "sec_only", "full")
    ]
    prediction_comparisons = [
        _compare_predictions(baseline_predictions, prediction_frames[name], name)
        for name in ("price_only", "sec_only", "full")
    ]
    legacy_sec_to_full = _compare_legacy(
        legacy_frames["sec_only"], legacy_frames["full"], "sec_only_to_full"
    )
    boosting_sec_to_full = _compare_predictions(
        prediction_frames["sec_only"], prediction_frames["full"], "sec_only_to_full"
    )
    return {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_cutoff": cutoff.isoformat(),
        "status": audit["status"],
        "promotion_allowed": audit["promotion_allowed_by_this_gate"],
        "gate_failure": audit.get("common_replay_failure"),
        "headline": _headline(
            legacy_comparisons,
            prediction_comparisons,
            legacy_sec_to_full,
            boosting_sec_to_full,
        ),
        "snapshot_tables": _snapshot_tables(audit),
        "legacy_comparisons": legacy_comparisons,
        "legacy_sec_to_full": legacy_sec_to_full,
        "legacy_sec_to_full_events": _legacy_events(
            legacy_frames["sec_only"], legacy_frames["full"]
        )
        .sort(LEGACY_KEYS)
        .to_dicts(),
        "legacy_timeline": _legacy_timeline(baseline_legacy, legacy_frames["full"]),
        "legacy_top_tickers": _legacy_top_tickers(baseline_legacy, legacy_frames["full"]),
        "prediction_comparisons": prediction_comparisons,
        "boosting_sec_to_full": boosting_sec_to_full,
        "score_histogram": _score_histogram(baseline_predictions, prediction_frames["full"]),
        "top_score_movers": _top_score_movers(baseline_predictions, prediction_frames["full"]),
        "top10_timeline": _top_n_timeline(baseline_predictions, prediction_frames["full"], 10),
        "common_portfolios": _common_portfolio_comparisons(scenarios, cutoff),
        "focus": _focus_evidence(
            inputs,
            audit,
            scenarios,
            prediction_frames,
        ),
        "feature_fold": _feature_fold_evidence(scenarios, prediction_frames, inputs),
        "scenario_statuses": [
            {
                "scenario": scenario.name,
                "label": scenario.label,
                "common_status": scenario.common_status,
            }
            for scenario in inputs.scenarios
        ],
        "provenance": _provenance(audit, inputs, scenarios),
    }


def _scenario_map(
    scenarios: Sequence[ScenarioArtifacts],
) -> dict[str, ScenarioArtifacts]:
    mapped = {scenario.name: scenario for scenario in scenarios}
    expected = {"baseline", "price_only", "sec_only", "full"}
    if set(mapped) != expected or len(mapped) != len(scenarios):
        raise ValueError(
            f"Refresh attribution requires exactly these scenarios: {sorted(expected)}"
        )
    return mapped


def _read_legacy(run_dir: Path, cutoff: date) -> pl.DataFrame:
    path = run_dir / "legacy_common_holdings.parquet"
    frame = pl.read_parquet(path).filter(pl.col("decision_month") <= pl.lit(cutoff))
    _require_columns(frame, (*LEGACY_KEYS, "target_weight"), path)
    return frame.select(*LEGACY_KEYS, "target_weight").sort(LEGACY_KEYS)


def _read_predictions(run_dir: Path, cutoff: date) -> pl.DataFrame:
    path = run_dir / "classification_h06" / "predictions.parquet"
    frame = pl.read_parquet(path).filter(pl.col("decision_month") <= pl.lit(cutoff))
    _require_columns(
        frame,
        (*PREDICTION_KEYS, "score", "calibrated_probability"),
        path,
    )
    return frame


def _compare_legacy(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    scenario: str,
) -> dict[str, object]:
    spec = TableSpec("legacy_positions", "", LEGACY_KEYS, "decision_month", "ticker")
    diff = compare_frames(
        baseline,
        candidate,
        spec=spec,
        historical_cutoff=None,
        materiality_tolerance=MATERIALITY_TOLERANCE,
    )
    summary = diff.summary
    return {
        "scenario": scenario,
        "baseline_rows": summary["baseline_rows"],
        "candidate_rows": summary["candidate_rows"],
        "added_rows": summary["added_rows"],
        "removed_rows": summary["removed_rows"],
        "changed_common_rows": summary["changed_common_rows"],
        "total_position_events": (
            _as_int(summary["added_rows"])
            + _as_int(summary["removed_rows"])
            + _as_int(summary["changed_common_rows"])
        ),
    }


def _compare_predictions(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    scenario: str,
) -> dict[str, object]:
    spec = TableSpec(
        "boosting_predictions",
        "",
        PREDICTION_KEYS,
        "decision_month",
        "ticker",
    )
    diff = compare_frames(
        baseline,
        candidate,
        spec=spec,
        historical_cutoff=None,
        materiality_tolerance=MATERIALITY_TOLERANCE,
    )
    paired = baseline.join(candidate, on=list(PREDICTION_KEYS), how="inner", suffix="__candidate")
    score_stats = _numeric_change_stats(paired, "score")
    probability_stats = _numeric_change_stats(paired, "calibrated_probability")
    top_5 = _top_n_summary(baseline, candidate, 5)
    top_10 = _top_n_summary(baseline, candidate, 10)
    return {
        "scenario": scenario,
        "baseline_rows": diff.summary["baseline_rows"],
        "candidate_rows": diff.summary["candidate_rows"],
        "common_rows": paired.height,
        "added_rows": diff.summary["added_rows"],
        "removed_rows": diff.summary["removed_rows"],
        "any_changed_rows": diff.summary["changed_common_rows"],
        "score": score_stats,
        "calibrated_probability": probability_stats,
        "raw_signal_top_5": top_5,
        "raw_signal_top_10": top_10,
    }


def _numeric_change_stats(paired: pl.DataFrame, column: str) -> dict[str, object]:
    difference = (
        pl.col(column).cast(pl.Float64) - pl.col(f"{column}__candidate").cast(pl.Float64)
    ).abs()
    observed = paired.select(difference.alias("difference"))["difference"].drop_nulls()
    material = observed.filter(observed > MATERIALITY_TOLERANCE)
    return {
        "changed_rows": material.len(),
        "changed_share": material.len() / paired.height if paired.height else 0.0,
        "median_absolute_difference": _quantile(material, 0.5),
        "p95_absolute_difference": _quantile(material, 0.95),
        "maximum_absolute_difference": _as_float(material.max()) if material.len() else 0.0,
    }


def _quantile(series: pl.Series, probability: float) -> float:
    value = series.quantile(probability, interpolation="nearest")
    return float(value) if value is not None else 0.0


def _top_n_summary(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    top_n: int,
) -> dict[str, int]:
    left = _raw_signal_keys(baseline, top_n)
    right = _raw_signal_keys(candidate, top_n)
    keys = ["decision_month", "ticker"]
    return {
        "entries": right.join(left, on=keys, how="anti").height,
        "exits": left.join(right, on=keys, how="anti").height,
    }


def _raw_signal_keys(predictions: pl.DataFrame, top_n: int) -> pl.DataFrame:
    return boosting_predictions_to_holdings(
        predictions,
        strategy=f"Diagnostic Top {top_n}",
        top_n=top_n,
    ).select("decision_month", "ticker")


def _legacy_events(baseline: pl.DataFrame, candidate: pl.DataFrame) -> pl.DataFrame:
    keys = list(LEGACY_KEYS)
    added = (
        candidate.join(baseline, on=keys, how="anti")
        .select(keys)
        .with_columns(pl.lit("added").alias("change_type"))
    )
    removed = (
        baseline.join(candidate, on=keys, how="anti")
        .select(keys)
        .with_columns(pl.lit("removed").alias("change_type"))
    )
    paired = baseline.join(candidate, on=keys, how="inner", suffix="__candidate")
    changed = (
        paired.filter(
            (pl.col("target_weight") - pl.col("target_weight__candidate")).abs()
            > MATERIALITY_TOLERANCE
        )
        .select(keys)
        .with_columns(pl.lit("weight_changed").alias("change_type"))
    )
    return pl.concat([added, removed, changed], how="vertical")


def _legacy_timeline(baseline: pl.DataFrame, candidate: pl.DataFrame) -> list[dict[str, object]]:
    events = _legacy_events(baseline, candidate)
    return (
        events.with_columns(pl.col("decision_month").dt.year().alias("year"))
        .group_by("year", "change_type")
        .len(name="rows")
        .sort(["year", "change_type"])
        .to_dicts()
    )


def _legacy_top_tickers(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
) -> list[dict[str, object]]:
    return (
        _legacy_events(baseline, candidate)
        .group_by("ticker")
        .agg(
            pl.len().alias("events"),
            (pl.col("change_type") == "added").sum().alias("added"),
            (pl.col("change_type") == "removed").sum().alias("removed"),
            (pl.col("change_type") == "weight_changed").sum().alias("weight_changed"),
        )
        .sort("events", descending=True)
        .head(25)
        .to_dicts()
    )


def _ranked_scores(frame: pl.DataFrame, label: str) -> pl.DataFrame:
    return (
        frame.select("decision_month", "ticker", "score")
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over("decision_month")
            .alias(f"rank_{label}"),
            pl.col("score").alias(f"score_{label}"),
        )
        .drop("score")
    )


def _top_score_movers(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
) -> list[dict[str, object]]:
    paired = _ranked_scores(baseline, "baseline").join(
        _ranked_scores(candidate, "candidate"),
        on=["decision_month", "ticker"],
        how="inner",
    )
    return (
        paired.with_columns(
            (pl.col("score_candidate") - pl.col("score_baseline")).alias("score_change"),
            (pl.col("rank_candidate") - pl.col("rank_baseline")).alias("rank_change"),
        )
        .with_columns(pl.col("score_change").abs().alias("absolute_score_change"))
        .sort("absolute_score_change", descending=True)
        .head(40)
        .to_dicts()
    )


def _score_histogram(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
) -> list[dict[str, object]]:
    paired = baseline.select(*PREDICTION_KEYS, "score").join(
        candidate.select(*PREDICTION_KEYS, "score"),
        on=list(PREDICTION_KEYS),
        how="inner",
        suffix="__candidate",
    )
    differences = paired.select(
        (pl.col("score") - pl.col("score__candidate")).abs().alias("difference")
    )["difference"]
    boundaries = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1.0)
    rows: list[dict[str, object]] = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        rows.append(
            {
                "label": f"{lower:g}–{upper:g}",
                "rows": differences.filter((differences >= lower) & (differences < upper)).len(),
            }
        )
    rows.append({"label": ">=1", "rows": differences.filter(differences >= 1.0).len()})
    return rows


def _top_n_timeline(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    top_n: int,
) -> list[dict[str, object]]:
    left = _raw_signal_keys(baseline, top_n)
    right = _raw_signal_keys(candidate, top_n)
    keys = ["decision_month", "ticker"]
    added = right.join(left, on=keys, how="anti").with_columns(pl.lit("entry").alias("type"))
    removed = left.join(right, on=keys, how="anti").with_columns(pl.lit("exit").alias("type"))
    return (
        pl.concat([added, removed], how="vertical")
        .group_by("decision_month", "type")
        .len(name="rows")
        .sort(["decision_month", "type"])
        .to_dicts()
    )


def _common_portfolio_comparisons(
    scenarios: Mapping[str, ScenarioArtifacts],
    cutoff: date,
) -> list[dict[str, object]]:
    baseline_run = scenarios["baseline"].common_run
    if baseline_run is None:
        return []
    baseline = _read_common_holdings(baseline_run, cutoff)
    rows: list[dict[str, object]] = []
    for name in ("price_only", "sec_only", "full"):
        run = scenarios[name].common_run
        if run is None:
            rows.append({"scenario": name, "status": scenarios[name].common_status})
            continue
        candidate = _read_common_holdings(run, cutoff)
        spec = TableSpec("common_positions", "", LEGACY_KEYS, "decision_month", "ticker")
        diff = compare_frames(
            baseline,
            candidate,
            spec=spec,
            historical_cutoff=None,
            materiality_tolerance=MATERIALITY_TOLERANCE,
        )
        row: dict[str, object] = {
            "scenario": name,
            "status": scenarios[name].common_status,
        }
        row.update(diff.summary)
        rows.append(row)
    return rows


def _read_common_holdings(run_dir: Path, cutoff: date) -> pl.DataFrame:
    path = run_dir / "comparison_common_holdings.parquet"
    frame = pl.read_parquet(path).filter(pl.col("decision_month") <= pl.lit(cutoff))
    return frame.select(*LEGACY_KEYS, "target_weight").sort(LEGACY_KEYS)


def _focus_evidence(
    inputs: RefreshAttributionInputs,
    audit: Mapping[str, object],
    scenarios: Mapping[str, ScenarioArtifacts],
    predictions: Mapping[str, pl.DataFrame],
) -> dict[str, object]:
    score_rows = []
    for name in ("baseline", "price_only", "sec_only", "full"):
        ranked = _ranked_scores(predictions[name], name)
        row = ranked.filter(
            (pl.col("ticker") == inputs.focus_ticker)
            & (pl.col("decision_month") == pl.lit(inputs.focus_month))
        )
        if row.height != 1:
            raise ValueError(f"Expected one focus prediction for {name}, got {row.height}")
        values = row.to_dicts()[0]
        score_rows.append(
            {
                "scenario": name,
                "score": values[f"score_{name}"],
                "rank": values[f"rank_{name}"],
                "common_status": scenarios[name].common_status,
            }
        )
    audit_inputs = _object_mapping(audit, "inputs")
    baseline_snapshot = Path(str(audit_inputs["baseline_snapshot"]))
    candidate_snapshot = Path(str(audit_inputs["candidate_snapshot"]))
    return {
        "ticker": inputs.focus_ticker,
        "decision_month": inputs.focus_month.isoformat(),
        "scores": score_rows,
        "price": _focus_price_evidence(
            baseline_snapshot,
            candidate_snapshot,
            inputs.focus_ticker,
        ),
        "sec": _focus_sec_evidence(
            baseline_snapshot,
            candidate_snapshot,
            inputs.focus_ticker,
        ),
    }


def _focus_price_evidence(
    baseline_snapshot: Path,
    candidate_snapshot: Path,
    ticker: str,
) -> dict[str, object]:
    spec = TableSpec("focus_price", "US_Finalprice.parquet", ("ticker", "date"), "date")
    baseline = pl.read_parquet(baseline_snapshot / spec.relative_path).filter(
        pl.col("ticker") == ticker
    )
    candidate = pl.read_parquet(candidate_snapshot / spec.relative_path).filter(
        pl.col("ticker") == ticker
    )
    diff = compare_frames(
        baseline,
        candidate,
        spec=spec,
        historical_cutoff=None,
        materiality_tolerance=MATERIALITY_TOLERANCE,
    )
    return {
        **diff.summary,
        "first_date": str(baseline["date"].min()),
        "last_date": str(baseline["date"].max()),
        "strictly_identical": not diff.has_historical_drift,
    }


def _focus_sec_evidence(
    baseline_snapshot: Path,
    candidate_snapshot: Path,
    ticker: str,
) -> dict[str, object]:
    tables = (
        ("income_statement", "US_Income_statement.parquet", "filing_date"),
        ("balance_sheet", "US_Balance_sheet.parquet", "filing_date"),
        ("cash_flow", "US_Cash_flow.parquet", "filing_date"),
        ("earnings", "US_Earnings.parquet", "reportDate"),
    )
    rows = []
    latest_dates: list[date] = []
    for name, filename, filing_column in tables:
        baseline = pl.read_parquet(baseline_snapshot / filename).filter(pl.col("ticker") == ticker)
        candidate = pl.read_parquet(candidate_snapshot / filename).filter(
            pl.col("ticker") == ticker
        )
        filing_dates = candidate.select(
            pl.col(filing_column).cast(pl.String).str.to_date(strict=False).alias("filing_date")
        )["filing_date"].drop_nulls()
        if filing_dates.len():
            latest = filing_dates.max()
            if not isinstance(latest, date):
                raise ValueError(f"Invalid filing date for {ticker} in {filename}")
            latest_dates.append(latest)
        rows.append(
            {"table": name, "baseline_rows": baseline.height, "candidate_rows": candidate.height}
        )
    return {
        "tables": rows,
        "baseline_rows": sum(_as_int(row["baseline_rows"]) for row in rows),
        "candidate_rows": sum(_as_int(row["candidate_rows"]) for row in rows),
        "latest_filing_date": max(latest_dates).isoformat() if latest_dates else None,
    }


def _feature_fold_evidence(
    scenarios: Mapping[str, ScenarioArtifacts],
    predictions: Mapping[str, pl.DataFrame],
    inputs: RefreshAttributionInputs,
) -> dict[str, object]:
    focus = predictions["baseline"].filter(
        (pl.col("ticker") == inputs.focus_ticker)
        & (pl.col("decision_month") == pl.lit(inputs.focus_month))
    )
    fold = _as_int(focus["fold"].item())
    rows = {}
    for name, scenario in scenarios.items():
        manifest = pl.read_csv(
            scenario.boosting_run / "classification_h06" / "fold_feature_manifest.csv"
        ).filter(pl.col("fold") == fold)
        rows[name] = manifest.to_dicts()[0]
    baseline_pairs = {tuple(pair) for pair in json.loads(str(rows["baseline"]["winner_pairs"]))}
    candidate_pairs = {tuple(pair) for pair in json.loads(str(rows["full"]["winner_pairs"]))}
    return {
        "fold": fold,
        "baseline_pair_count": len(baseline_pairs),
        "candidate_pair_count": len(candidate_pairs),
        "retained_pair_count": len(baseline_pairs & candidate_pairs),
        "baseline_rows": _fold_row_counts(rows["baseline"]),
        "candidate_rows": _fold_row_counts(rows["full"]),
    }


def _fold_row_counts(row: Mapping[str, object]) -> dict[str, int]:
    return {
        "train": _as_int(row["train_rows"]),
        "validation": _as_int(row["validation_rows"]),
        "test": _as_int(row["test_rows"]),
    }


def _snapshot_tables(audit: Mapping[str, object]) -> list[dict[str, object]]:
    values = audit.get("snapshot_comparison")
    if not isinstance(values, list):
        raise ValueError("Audit report lacks snapshot_comparison")
    rows = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Invalid snapshot comparison row")
        name = str(value["table"])
        family = "prices" if name in PRICE_TABLES else "SEC" if name in SEC_TABLES else "universe"
        rows.append({**value, "family": family})
    return rows


def _headline(
    legacy: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    legacy_sec_to_full: Mapping[str, object],
    boosting_sec_to_full: Mapping[str, object],
) -> dict[str, object]:
    legacy_by_name = {str(row["scenario"]): row for row in legacy}
    predictions_by_name = {str(row["scenario"]): row for row in predictions}
    return {
        "legacy_full_changed_common": legacy_by_name["full"]["changed_common_rows"],
        "legacy_price_only_events": legacy_by_name["price_only"]["total_position_events"],
        "legacy_sec_only_events": legacy_by_name["sec_only"]["total_position_events"],
        "legacy_sec_to_full_events": legacy_sec_to_full["total_position_events"],
        "boosting_full_changed_common": predictions_by_name["full"]["any_changed_rows"],
        "boosting_full_score_changed": _object_mapping(predictions_by_name["full"], "score")[
            "changed_rows"
        ],
        "boosting_price_score_changed": _object_mapping(predictions_by_name["price_only"], "score")[
            "changed_rows"
        ],
        "boosting_sec_score_changed": _object_mapping(predictions_by_name["sec_only"], "score")[
            "changed_rows"
        ],
        "boosting_sec_to_full_score_changed": _object_mapping(boosting_sec_to_full, "score")[
            "changed_rows"
        ],
    }


def _provenance(
    audit: Mapping[str, object],
    inputs: RefreshAttributionInputs,
    scenarios: Mapping[str, ScenarioArtifacts],
) -> dict[str, object]:
    comparison = _object_mapping(audit, "provenance_comparison")
    reporting_module = Path(__file__).parents[1] / "reporting" / "refresh_replay_html.py"
    return {
        "audit_report_sha256": _sha256(inputs.audit_report),
        "report_builder_sha256": {
            "refresh_attribution.py": _sha256(Path(__file__)),
            "refresh_replay_html.py": _sha256(reporting_module),
        },
        "all_code_identical": comparison["all_code_identical"],
        "all_config_identical": comparison["all_config_identical"],
        "all_runtime_identical": comparison["all_runtime_identical"],
        "audit_created_at_utc": audit["created_at_utc"],
        "scenario_artifacts": _scenario_artifact_hashes(scenarios),
    }


def _scenario_artifact_hashes(
    scenarios: Mapping[str, ScenarioArtifacts],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ("baseline", "price_only", "sec_only", "full"):
        scenario = scenarios[name]
        common_manifest = (
            scenario.common_run / "manifest.json" if scenario.common_run is not None else None
        )
        rows.append(
            {
                "scenario": name,
                "legacy_manifest_sha256": _sha256(scenario.legacy_run / "data_input_manifest.json"),
                "boosting_manifest_sha256": _sha256(scenario.boosting_run / "manifest.json"),
                "common_manifest_sha256": (
                    _sha256(common_manifest) if common_manifest is not None else None
                ),
            }
        )
    return rows


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _object_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object at {key}")
    return nested


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks report columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected integer-compatible value, got {type(value).__name__}")
    return int(value)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected float-compatible value, got {type(value).__name__}")
    return float(value)
