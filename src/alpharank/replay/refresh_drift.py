"""End-to-end refresh replay audit and fail-closed classification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from alpharank.replay.refresh_compare import (
    SNAPSHOT_TABLES,
    FrameDiff,
    TableSpec,
    changed_key_events,
    compare_frames,
    read_table,
    write_frame_diff,
)
from alpharank.replay.refresh_provenance import compare_provenance_pairs


@dataclass(frozen=True, slots=True)
class ReplayAuditInputs:
    """Baseline and candidate inputs required by a complete refresh audit."""

    baseline_snapshot: Path
    candidate_snapshot: Path
    baseline_legacy: Path
    candidate_legacy: Path
    baseline_boosting: Path
    candidate_boosting: Path
    baseline_common: Path
    candidate_common: Path
    historical_cutoff: date


@dataclass(frozen=True, slots=True)
class ReplayTable:
    """One replay artifact and the columns whose equality is contractual."""

    stage: str
    root_name: str
    spec: TableSpec
    columns: tuple[str, ...]


REPLAY_TABLES = (
    ReplayTable(
        "legacy_portfolio",
        "legacy",
        TableSpec(
            "legacy_positions",
            "legacy_common_holdings.parquet",
            ("strategy", "decision_month", "holding_month", "ticker"),
            "decision_month",
            "ticker",
        ),
        ("strategy", "decision_month", "holding_month", "ticker", "target_weight"),
    ),
    ReplayTable(
        "legacy_simulation",
        "legacy",
        TableSpec(
            "legacy_monthly",
            "legacy_common_monthly.parquet",
            ("strategy", "decision_month", "holding_month"),
            "decision_month",
        ),
        (),
    ),
    ReplayTable(
        "boosting_signal",
        "boosting",
        TableSpec(
            "boosting_predictions",
            "classification_h06/predictions.parquet",
            ("method", "horizon", "decision_month", "ticker"),
            "decision_month",
            "ticker",
        ),
        (),
    ),
    ReplayTable(
        "common_portfolio",
        "common",
        TableSpec(
            "common_positions",
            "comparison_common_holdings.parquet",
            ("strategy", "decision_month", "holding_month", "ticker"),
            "decision_month",
            "ticker",
        ),
        ("strategy", "decision_month", "holding_month", "ticker", "target_weight"),
    ),
    ReplayTable(
        "common_simulation",
        "common",
        TableSpec(
            "common_monthly",
            "comparison_common_monthly.parquet",
            ("strategy", "decision_month", "holding_month"),
            "decision_month",
        ),
        (),
    ),
)


def audit_refresh_replay(
    inputs: ReplayAuditInputs,
    output_dir: Path,
    *,
    materiality_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare one complete candidate with the published baseline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_diffs = _compare_snapshot_tables(inputs, output_dir, materiality_tolerance)
    replay_diffs = _compare_replay_tables(inputs, output_dir, materiality_tolerance)
    provenance = _compare_provenance(inputs)
    attribution = _build_attribution(snapshot_diffs, replay_diffs, output_dir)
    status = _classify(snapshot_diffs, replay_diffs, provenance, attribution)
    report = {
        "contract_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "promotion_allowed_by_this_gate": status == "identical_historical_portfolios",
        "historical_cutoff": inputs.historical_cutoff.isoformat(),
        "inputs": _input_paths(inputs),
        "snapshot_comparison": [diff.summary for diff in snapshot_diffs.values()],
        "replay_comparison": [
            item[1].summary | {"stage": item[0]} for item in replay_diffs.values()
        ],
        "provenance_comparison": provenance,
        "portfolio_attribution": attribution,
        "fail_closed_reason": _fail_closed_reason(status),
    }
    _write_json(output_dir / "refresh_replay_report.json", report)
    return report


def audit_blocked_refresh(
    failed_refresh_run: Path,
    baseline_snapshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Turn a failed data gate into an explicit no-model replay report."""

    gate_path = failed_refresh_run / "price_revision_guard.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Missing failed refresh gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    evidence = _hash_refresh_evidence(failed_refresh_run)
    run_id = failed_refresh_run.name
    report = {
        "contract_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_before_replay",
        "promotion_allowed_by_this_gate": False,
        "refresh_run_id": run_id,
        "baseline_snapshot": str(baseline_snapshot.resolve()),
        "failed_gate": {
            "name": "price_revision_guard",
            "passed": gate.get("passed", False),
            "reasons": gate.get("blocking_reasons", gate.get("failure_reasons", [])),
            "policy": gate.get("policy", {}),
            "candidate_rows": gate.get("candidate_rows"),
            "candidate_tickers": gate.get("candidate_tickers"),
            "historical_daily_return_revisions_over_threshold": gate.get(
                "historical_daily_return_revisions_over_threshold"
            ),
            "historical_return_revision_tickers": gate.get("historical_return_revision_tickers"),
            "historical_return_revision_examples": gate.get(
                "historical_return_revision_examples", []
            ),
            "historical_return_availability_changes": gate.get(
                "historical_return_availability_changes"
            ),
            "transition_factor_findings": gate.get("transition_factor_findings"),
        },
        "evidence": evidence,
        "model_execution": {
            "legacy_candidate_executed": False,
            "boosting_candidate_executed": False,
            "reason": "Candidate data failed before snapshot publication; models must not consume it.",
        },
        "fail_closed_reason": (
            "The candidate was quarantined before Legacy and Boosting. "
            "The published snapshot pointer remains the only eligible model input."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "refresh_replay_report.json", report)
    return report


def _compare_snapshot_tables(
    inputs: ReplayAuditInputs,
    output_dir: Path,
    tolerance: float,
) -> dict[str, FrameDiff]:
    diffs = {}
    for spec in SNAPSHOT_TABLES:
        diff = compare_frames(
            read_table(inputs.baseline_snapshot, spec),
            read_table(inputs.candidate_snapshot, spec),
            spec=spec,
            historical_cutoff=inputs.historical_cutoff,
            materiality_tolerance=tolerance,
        )
        diffs[spec.name] = diff
        write_frame_diff(output_dir / "data_diffs", spec.name, diff)
    return diffs


def _compare_replay_tables(
    inputs: ReplayAuditInputs,
    output_dir: Path,
    tolerance: float,
) -> dict[str, tuple[str, FrameDiff]]:
    roots = _replay_roots(inputs)
    diffs = {}
    for table in REPLAY_TABLES:
        baseline = read_table(roots[f"baseline_{table.root_name}"], table.spec)
        candidate = read_table(roots[f"candidate_{table.root_name}"], table.spec)
        if table.columns:
            baseline = baseline.select(table.columns)
            candidate = candidate.select(table.columns)
        diff = compare_frames(
            baseline,
            candidate,
            spec=table.spec,
            historical_cutoff=inputs.historical_cutoff,
            materiality_tolerance=tolerance,
        )
        diffs[table.spec.name] = (table.stage, diff)
        write_frame_diff(output_dir / "replay_diffs", table.spec.name, diff)
    return diffs


def _compare_provenance(inputs: ReplayAuditInputs) -> dict[str, Any]:
    pairs = {
        "legacy": (
            inputs.baseline_legacy / "data_input_manifest.json",
            inputs.candidate_legacy / "data_input_manifest.json",
        ),
        "boosting": (
            inputs.baseline_boosting / "manifest.json",
            inputs.candidate_boosting / "manifest.json",
        ),
        "common": (
            inputs.baseline_common / "manifest.json",
            inputs.candidate_common / "manifest.json",
        ),
    }
    return compare_provenance_pairs(pairs)


def _build_attribution(
    snapshot_diffs: dict[str, FrameDiff],
    replay_diffs: dict[str, tuple[str, FrameDiff]],
    output_dir: Path,
) -> dict[str, Any]:
    first_stage = next(
        (
            stage
            for stage, has_drift in _ordered_drift_stages(snapshot_diffs, replay_diffs)
            if has_drift
        ),
        None,
    )
    portfolio_events = changed_key_events(replay_diffs["common_positions"][1])
    if not portfolio_events.is_empty():
        portfolio_events.write_parquet(output_dir / "portfolio_drift_keys.parquet")
    data_datasets = [name for name, diff in snapshot_diffs.items() if diff.has_historical_drift]
    portfolio_rows = portfolio_events.height
    return {
        "first_divergent_stage": first_stage,
        "data_datasets_with_historical_drift": data_datasets,
        "portfolio_drift_rows": portfolio_rows,
        "portfolio_drift_keys_path": (
            str((output_dir / "portfolio_drift_keys.parquet").resolve()) if portfolio_rows else None
        ),
        "exhaustively_attributed": portfolio_rows == 0,
        "review_status": (
            "not_required_identical"
            if portfolio_rows == 0
            else "blocked_pending_causal_key_attribution"
        ),
        "explanation": (
            "No portfolio position or weight changed through the common cutoff."
            if portfolio_rows == 0
            else "Exact portfolio and upstream drift keys are retained, but correlation is not "
            "accepted as causal attribution; human review is required."
        ),
    }


def _ordered_drift_stages(
    snapshot_diffs: dict[str, FrameDiff],
    replay_diffs: dict[str, tuple[str, FrameDiff]],
) -> list[tuple[str, bool]]:
    return [
        ("snapshot", any(diff.has_historical_drift for diff in snapshot_diffs.values())),
        *[(stage, diff.has_historical_drift) for stage, diff in replay_diffs.values()],
    ]


def _classify(
    snapshot_diffs: dict[str, FrameDiff],
    replay_diffs: dict[str, tuple[str, FrameDiff]],
    provenance: dict[str, Any],
    attribution: dict[str, Any],
) -> str:
    del snapshot_diffs
    common_drift = replay_diffs["common_positions"][1].has_historical_drift
    legacy_drift = replay_diffs["legacy_positions"][1].has_historical_drift
    if not common_drift and not legacy_drift:
        return "identical_historical_portfolios"
    provenance_same = all(
        provenance[name]
        for name in ("all_code_identical", "all_config_identical", "all_runtime_identical")
    )
    if not provenance_same:
        return "code_config_runtime_drift"
    if attribution["exhaustively_attributed"]:
        return "explained_data_drift"
    return "unexplained_portfolio_drift"


def _hash_refresh_evidence(run_dir: Path) -> list[dict[str, Any]]:
    evidence = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        evidence.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_roots(inputs: ReplayAuditInputs) -> dict[str, Path]:
    return {
        "baseline_legacy": inputs.baseline_legacy,
        "candidate_legacy": inputs.candidate_legacy,
        "baseline_boosting": inputs.baseline_boosting,
        "candidate_boosting": inputs.candidate_boosting,
        "baseline_common": inputs.baseline_common,
        "candidate_common": inputs.candidate_common,
    }


def _input_paths(inputs: ReplayAuditInputs) -> dict[str, str]:
    return {
        field: str(getattr(inputs, field).resolve())
        for field in (
            "baseline_snapshot",
            "candidate_snapshot",
            "baseline_legacy",
            "candidate_legacy",
            "baseline_boosting",
            "candidate_boosting",
            "baseline_common",
            "candidate_common",
        )
    }


def _fail_closed_reason(status: str) -> str | None:
    if status == "identical_historical_portfolios":
        return None
    return f"Refresh replay status {status} requires review and blocks snapshot promotion."


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
