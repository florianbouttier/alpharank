"""Evidence model for a ticker-transition price replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

_PREDICTION_KEYS = ("decision_month", "ticker", "fold", "method", "horizon")
_PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "adjusted_close")
_PERFORMANCE_COLUMNS = ("total_return", "cagr", "sharpe", "max_drawdown")
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class TickerTransitionReplayInputs:
    """Immutable paths and focus identifiers for one baseline/candidate replay."""

    baseline_legacy_run: Path
    candidate_legacy_run: Path
    baseline_boosting_run: Path
    candidate_boosting_run: Path
    baseline_trend_run: Path
    candidate_common_run: Path
    candidate_trend_run: Path
    target_ticker: str
    provider_ticker: str
    focus_decision_month: date
    expected_causal_rank: int
    generated_at_utc: datetime


def build_ticker_transition_replay_report(
    inputs: TickerTransitionReplayInputs,
) -> dict[str, object]:
    """Build one machine-readable attribution from existing replay artifacts."""

    prices = _price_evidence(inputs)
    predictions = _prediction_evidence(inputs)
    legacy = _legacy_evidence(inputs)
    portfolios = _portfolio_evidence(inputs)
    manifests = _manifest_evidence(inputs)
    checks = _checks(
        prices,
        predictions,
        legacy,
        portfolios,
        manifests,
        expected_causal_rank=inputs.expected_causal_rank,
    )
    snapshot = _read_json(inputs.candidate_legacy_run / "input_snapshot/snapshot_manifest.json")
    report: dict[str, object] = {
        "status": "passed" if all(checks.values()) else "failed",
        "publication_eligible": False,
        "publication_note": (
            "Research replay only; the canonical model-input pointer was not moved."
        ),
        "generated_at_utc": inputs.generated_at_utc.isoformat(),
        "transition": {
            "target_ticker": inputs.target_ticker,
            "provider_ticker": inputs.provider_ticker,
            "focus_decision_month": inputs.focus_decision_month.isoformat(),
            "focus_holding_month": _next_month(inputs.focus_decision_month).isoformat(),
            "expected_causal_rank": inputs.expected_causal_rank,
        },
        "snapshot": {
            "composition_id": snapshot.get("composition_id"),
            "generated_at": snapshot.get("generated_at"),
            "price_sha256": _nested(snapshot, "output_sha256", "US_Finalprice.parquet"),
            "data_freshness": snapshot.get("data_freshness", {}),
        },
        "prices": prices,
        "predictions": predictions,
        "legacy": legacy,
        "portfolios": portfolios,
        "manifests": manifests,
        "checks": checks,
        "artifacts": _artifact_records(inputs),
    }
    return report


def write_report_bundle(
    report: Mapping[str, object],
    *,
    output_json: Path,
    output_html: Path,
    html_writer: Callable[[Mapping[str, object], Path], None],
) -> Path:
    """Write JSON and HTML, then bind both immutable payloads in a manifest."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    )
    output_json.write_text(serialized, encoding="utf-8")
    html_writer(report, output_html)
    manifest_path = output_json.with_name("ticker_transition_replay_report_manifest.json")
    manifest = {
        "report_json": _source_record(output_json),
        "report_html": _source_record(output_html),
        "input_artifacts": report.get("artifacts", []),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _price_evidence(inputs: TickerTransitionReplayInputs) -> dict[str, object]:
    baseline = _ticker_prices(inputs.baseline_legacy_run, inputs.target_ticker)
    candidate = _ticker_prices(inputs.candidate_legacy_run, inputs.target_ticker)
    common = baseline.join(candidate, on="date", how="inner", suffix="_candidate")
    changed = _changed_rows(common, _PRICE_COLUMNS, suffix="_candidate")
    added = candidate.join(baseline.select("date"), on="date", how="anti").sort("date")
    removed = baseline.join(candidate.select("date"), on="date", how="anti")
    audit_path = (
        inputs.candidate_legacy_run
        / "input_snapshot/lineage/prices/price_ticker_transition_audit.parquet"
    )
    audit = _read_parquet(audit_path).filter(
        (pl.col("target_ticker") == inputs.target_ticker)
        & (pl.col("provider_ticker") == inputs.provider_ticker)
    )
    policy = _read_json(audit_path.with_name("price_ticker_transition_policy.json"))
    required_lineage = (
        "provider_daily_return",
        "provider_adjusted_close",
        "selected_adjusted_close",
        "return_source_vintage_id",
    )
    unlineaged = audit.filter(
        pl.any_horizontal(pl.col(column).is_null() for column in required_lineage)
    )
    return {
        "baseline_rows": baseline.height,
        "candidate_rows": candidate.height,
        "added_rows": added.height,
        "removed_rows": removed.height,
        "changed_common_rows": changed,
        "baseline_last_date": baseline["date"].max(),
        "candidate_last_date": candidate["date"].max(),
        "added_first_date": added["date"].min(),
        "added_last_date": added["date"].max(),
        "audit_rows": audit.height,
        "audit_first_date": audit["date"].min(),
        "audit_last_date": audit["date"].max(),
        "policy_id": policy.get("policy_id"),
        "derivation_rule": "selected[t] = selected[t-1] * (1 + provider_daily_return[t])",
        "unlineaged_overlay_rows": unlineaged.height,
        "return_source_vintages": audit["return_source_vintage_id"].unique().sort().to_list(),
        "evidence_urls": audit["evidence_url"].unique().sort().to_list(),
    }


def _prediction_evidence(inputs: TickerTransitionReplayInputs) -> dict[str, object]:
    baseline = _predictions(inputs.baseline_boosting_run)
    candidate = _predictions(inputs.candidate_boosting_run)
    common = baseline.join(candidate, on=list(_PREDICTION_KEYS), how="inner", suffix="_candidate")
    only_candidate = candidate.join(
        baseline.select(_PREDICTION_KEYS), on=list(_PREDICTION_KEYS), how="anti"
    )
    only_baseline = baseline.join(
        candidate.select(_PREDICTION_KEYS), on=list(_PREDICTION_KEYS), how="anti"
    )
    baseline_focus = _focus_prediction(baseline, inputs)
    candidate_focus = _focus_prediction(candidate, inputs)
    baseline_rank = _focus_rank(
        baseline,
        inputs.baseline_trend_run / "causal_trend_eligibility.parquet",
        inputs,
    )
    candidate_rank = _focus_rank(
        candidate,
        inputs.candidate_trend_run / "causal_trend_eligibility.parquet",
        inputs,
    )
    return {
        "baseline_rows": baseline.height,
        "candidate_rows": candidate.height,
        "only_baseline_rows": only_baseline.height,
        "only_candidate_rows": only_candidate.height,
        "only_candidate": _records(only_candidate.select(_PREDICTION_KEYS, "target_status_1m")),
        "changed_common_scores": _changed_rows(common, ("score",), suffix="_candidate"),
        "changed_common_probabilities": _changed_rows(
            common, ("calibrated_probability",), suffix="_candidate"
        ),
        "baseline_focus": baseline_focus,
        "candidate_focus": candidate_focus,
        "baseline_causal_rank": baseline_rank,
        "candidate_causal_rank": candidate_rank,
    }


def _legacy_evidence(inputs: TickerTransitionReplayInputs) -> dict[str, object]:
    filename = "legacy_common_holdings.parquet"
    baseline = _read_parquet(inputs.baseline_legacy_run / filename)
    candidate = _read_parquet(inputs.candidate_legacy_run / filename)
    keys = ("holding_month", "ticker", "strategy")
    only_candidate = candidate.join(baseline.select(keys), on=list(keys), how="anti")
    only_baseline = baseline.join(candidate.select(keys), on=list(keys), how="anti")
    common = baseline.join(candidate, on=list(keys), how="inner", suffix="_candidate")
    performance = _performance_comparison(
        inputs.baseline_legacy_run / "legacy_common_performance.csv",
        inputs.candidate_legacy_run / "legacy_common_performance.csv",
    )
    return {
        "baseline_holdings": baseline.height,
        "candidate_holdings": candidate.height,
        "only_baseline_holdings": only_baseline.height,
        "only_candidate_holdings": only_candidate.height,
        "changed_common_returns": _changed_rows(
            common, ("realized_return", "target_weight"), suffix="_candidate"
        ),
        "target_ticker_baseline_holdings": baseline.filter(
            pl.col("ticker") == inputs.target_ticker
        ).height,
        "target_ticker_candidate_holdings": candidate.filter(
            pl.col("ticker") == inputs.target_ticker
        ).height,
        "performance": performance,
    }


def _portfolio_evidence(inputs: TickerTransitionReplayInputs) -> dict[str, object]:
    native = _read_csv(inputs.candidate_common_run / "comparison_common_performance.csv")
    trend = _read_csv(inputs.candidate_trend_run / "comparison_common_performance.csv")
    baseline_trend = _read_csv(inputs.baseline_trend_run / "comparison_common_performance.csv")
    monthly = _read_csv(inputs.candidate_trend_run / "comparison_common_monthly.csv")
    focus_month = inputs.focus_decision_month.isoformat()
    focus_strategies = (
        "Boosting Top 15 | Causal trend",
        "Boosting Top 20 | Causal trend",
    )
    focus = monthly.filter(
        (pl.col("decision_month") == focus_month) & pl.col("strategy").is_in(focus_strategies)
    ).select(
        "strategy",
        "decision_month",
        "holding_month",
        "gross_return",
        "turnover",
        "transaction_cost",
        "net_return",
        "benchmark_return",
        "active_return",
        "n_positions",
    )
    target_return = float(
        str(
            _focus_prediction(_predictions(inputs.candidate_boosting_run), inputs)[
                "future_return_1m"
            ]
        )
    )
    return {
        "native_performance": _performance_records(native),
        "causal_trend_performance": _performance_records(trend),
        "baseline_shared_performance": _performance_comparison_frames(baseline_trend, trend),
        "focus_month": _records(focus.sort("strategy")),
        "target_ticker_gross_contribution_top15": target_return / 15.0,
        "target_ticker_gross_contribution_top20": target_return / 20.0,
    }


def _manifest_evidence(inputs: TickerTransitionReplayInputs) -> dict[str, object]:
    native = _read_json(inputs.candidate_common_run / "manifest.json")
    trend = _read_json(inputs.candidate_trend_run / "manifest.json")
    runtime = trend.get("runtime_provenance", {})
    git = runtime.get("git", {}) if isinstance(runtime, dict) else {}
    return {
        "native_comparison_eligible": native.get("comparison_eligible"),
        "native_publication_eligible": native.get("publication_eligible"),
        "native_lineage_passed": _nested(native, "lineage_check", "passed"),
        "trend_comparison_eligible": trend.get("comparison_eligible"),
        "trend_publication_eligible": trend.get("publication_eligible"),
        "trend_lineage_passed": _nested(trend, "lineage_check", "passed"),
        "trend_methodology_status": trend.get("methodology_status"),
        "git_head": git.get("head") if isinstance(git, dict) else None,
        "git_dirty": git.get("dirty") if isinstance(git, dict) else None,
    }


def _checks(
    prices: Mapping[str, object],
    predictions: Mapping[str, object],
    legacy: Mapping[str, object],
    portfolios: Mapping[str, object],
    manifests: Mapping[str, object],
    *,
    expected_causal_rank: int,
) -> dict[str, bool]:
    trend_strategies = {
        str(row["strategy"]) for row in _mapping_rows(portfolios, "causal_trend_performance")
    }
    return {
        "overlay_added_rows_match_audit": (
            prices["added_rows"] == prices["audit_rows"] and int(str(prices["added_rows"])) > 0
        ),
        "prior_target_prices_unchanged": prices["changed_common_rows"] == 0,
        "all_overlay_rows_are_return_lineaged": prices["unlineaged_overlay_rows"] == 0,
        "boosting_scores_unchanged": predictions["changed_common_scores"] == 0,
        "legacy_holdings_unchanged": (
            legacy["only_baseline_holdings"] == 0
            and legacy["only_candidate_holdings"] == 0
            and legacy["changed_common_returns"] == 0
        ),
        "target_return_is_evaluable": (
            _nested(predictions, "candidate_focus", "target_status_1m") == "evaluable"
        ),
        "target_causal_rank_matches_expected": (
            _nested(predictions, "candidate_causal_rank", "rank") == expected_causal_rank
        ),
        "causal_top15_and_top20_completed": {
            "Boosting Top 15 | Causal trend",
            "Boosting Top 20 | Causal trend",
        }.issubset(trend_strategies),
        "common_lineage_passed": bool(manifests["native_lineage_passed"])
        and bool(manifests["trend_lineage_passed"]),
        "runtime_was_clean": manifests["git_dirty"] is False,
    }


def _ticker_prices(run_dir: Path, ticker: str) -> pl.DataFrame:
    frame = _read_parquet(run_dir / "input_snapshot/US_Finalprice.parquet")
    return frame.filter(pl.col("ticker") == ticker).sort("date")


def _predictions(run_dir: Path) -> pl.DataFrame:
    frame = _read_parquet(run_dir / "classification_h06/predictions.parquet")
    duplicates = frame.group_by(_PREDICTION_KEYS).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise ValueError(f"Prediction keys are not unique in {run_dir}")
    return frame.with_row_index("_source_order")


def _focus_prediction(
    predictions: pl.DataFrame,
    inputs: TickerTransitionReplayInputs,
) -> dict[str, object]:
    focus = predictions.filter(
        (pl.col("ticker") == inputs.target_ticker)
        & (pl.col("decision_month") == pl.lit(inputs.focus_decision_month))
    )
    if focus.height != 1:
        raise ValueError(f"Expected one focus prediction, found {focus.height}")
    return focus.select(
        "decision_month",
        "ticker",
        "score",
        "calibrated_probability",
        "future_return_1m",
        "benchmark_future_return_1m",
        "future_excess_return_1m",
        "target_status_1m",
    ).row(0, named=True)


def _focus_rank(
    predictions: pl.DataFrame,
    registry_path: Path,
    inputs: TickerTransitionReplayInputs,
) -> dict[str, object]:
    registry = _read_parquet(registry_path).select(
        "decision_month", "ticker", "fold", "trend_eligible", "trend_positive_pair_fraction"
    )
    ranked = (
        predictions.filter(pl.col("decision_month") == pl.lit(inputs.focus_decision_month))
        .join(registry, on=["decision_month", "ticker", "fold"], how="left", validate="1:1")
        .filter(pl.col("trend_eligible"))
        .sort(["score", "_source_order"], descending=[True, False])
        .with_row_index("rank", offset=1)
    )
    focus = ranked.filter(pl.col("ticker") == inputs.target_ticker)
    if focus.height != 1:
        raise ValueError(f"Expected one ranked focus prediction, found {focus.height}")
    row = focus.select(
        "rank", "score", "future_return_1m", "target_status_1m", "trend_positive_pair_fraction"
    ).row(0, named=True)
    return {"eligible_rows": ranked.height, **row}


def _performance_comparison(baseline_path: Path, candidate_path: Path) -> list[dict[str, object]]:
    return _performance_comparison_frames(_read_csv(baseline_path), _read_csv(candidate_path))


def _performance_comparison_frames(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
) -> list[dict[str, object]]:
    common = baseline.select("strategy", *_PERFORMANCE_COLUMNS).join(
        candidate.select("strategy", *_PERFORMANCE_COLUMNS),
        on="strategy",
        how="inner",
        suffix="_candidate",
        validate="1:1",
    )
    expressions = [
        (pl.col(f"{column}_candidate") - pl.col(column)).alias(f"{column}_delta")
        for column in _PERFORMANCE_COLUMNS
    ]
    return _records(common.with_columns(*expressions).sort("strategy"))


def _performance_records(frame: pl.DataFrame) -> list[dict[str, object]]:
    columns = (
        "strategy",
        "start_holding_month",
        "end_holding_month",
        "months",
        *_PERFORMANCE_COLUMNS,
    )
    return _records(frame.select(*columns).sort("strategy"))


def _changed_rows(frame: pl.DataFrame, columns: Sequence[str], *, suffix: str) -> int:
    changes = [
        (pl.col(column).is_null() != pl.col(f"{column}{suffix}").is_null())
        | (
            (pl.col(column) - pl.col(f"{column}{suffix}"))
            .abs()
            .gt(_FLOAT_TOLERANCE)
            .fill_null(False)
        )
        for column in columns
    ]
    return frame.filter(pl.any_horizontal(changes)).height


def _artifact_records(inputs: TickerTransitionReplayInputs) -> list[dict[str, object]]:
    paths = (
        inputs.candidate_legacy_run / "input_snapshot/snapshot_manifest.json",
        inputs.candidate_legacy_run / "legacy_common_holdings.parquet",
        inputs.candidate_boosting_run / "classification_h06/predictions.parquet",
        inputs.candidate_common_run / "manifest.json",
        inputs.candidate_trend_run / "manifest.json",
        inputs.candidate_trend_run / "comparison_common_performance.csv",
    )
    return [_source_record(path) for path in paths]


def _source_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_csv(path)


def _records(frame: pl.DataFrame) -> list[dict[str, object]]:
    return [dict(row) for row in frame.iter_rows(named=True)]


def _mapping_rows(mapping: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = mapping[key]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Expected list of objects at {key}")
    return value


def _nested(mapping: Mapping[str, object], first: str, second: str) -> object:
    child = mapping.get(first)
    return child.get(second) if isinstance(child, dict) else None


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)
