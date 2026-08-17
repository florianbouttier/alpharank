"""Locked Legacy Optuna protocol and complete experiment audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


LEGACY_SEARCH_PROTOCOL_ID = "legacy-optuna-search-v1"
LEGACY_SEARCH_SPACE = {
    "n_long": {"kind": "int", "low": 50, "high": 400},
    "n_short": {"kind": "int", "low": 1, "high": 100},
    "n_asset": {"kind": "int", "low": 5, "high": 30},
    "n_max_per_sector": {"kind": "int", "low": 1, "high": 2},
}
LOCKED_LEGACY_ANCHORS = (
    {"n_long": 260, "n_short": 71, "n_asset": 5, "n_max_per_sector": 2},
    {"n_long": 201, "n_short": 87, "n_asset": 5, "n_max_per_sector": 2},
    {"n_long": 224, "n_short": 83, "n_asset": 7, "n_max_per_sector": 2},
    {"n_long": 181, "n_short": 96, "n_asset": 24, "n_max_per_sector": 2},
    {"n_long": 138, "n_short": 5, "n_asset": 22, "n_max_per_sector": 1},
)
LOCKED_LEGACY_SEEDS = (42, 41)
LOCKED_LEGACY_N_TRIALS = 30
LOCKED_CALIBRATION_START = "2010-01"


def legacy_search_protocol_manifest() -> dict[str, Any]:
    return {
        "protocol_id": LEGACY_SEARCH_PROTOCOL_ID,
        "search_space": LEGACY_SEARCH_SPACE,
        "seeds": list(LOCKED_LEGACY_SEEDS),
        "n_trials_per_split": LOCKED_LEGACY_N_TRIALS,
        "calibration_start": LOCKED_CALIBRATION_START,
        "split_rule": "each January expanding-window calibration",
        "sampler": "Optuna TPESampler with declared seed",
        "top_optuna_candidates_retained": 10,
        "stable_anchor_candidates": list(LOCKED_LEGACY_ANCHORS),
        "selection_rule": (
            "maximum calibration score, then n_asset, n_max_per_sector, "
            "n_long, n_short ascending"
        ),
        "uses_final_confirmation_for_selection": False,
    }


def write_legacy_search_audit(
    *,
    output_path: Path,
    experiments: Mapping[str, Mapping[str, Any]],
    n_trials: int,
    first_date: str,
    n_jobs: int,
) -> dict[str, Any]:
    """Persist every Optuna trial and refined candidate with rejection reasons."""

    audit_experiments: list[dict[str, Any]] = []
    for experiment_id, output in sorted(experiments.items()):
        for split, study in sorted(
            (output.get("studies") or {}).items(), key=lambda item: str(item[0])
        ):
            trials = sorted(study.trials, key=lambda trial: int(trial.number))
            raw_trials = [
                {
                    "number": int(trial.number),
                    "params": {key: int(value) for key, value in sorted(trial.params.items())},
                    "value": float(trial.value) if trial.value is not None else None,
                    "state": str(trial.state.name),
                }
                for trial in trials
            ]
            winner = {
                key: int(value)
                for key, value in sorted(
                    (study.user_attrs.get("refined_best_params") or {}).items()
                )
            }
            refined = []
            for candidate in study.user_attrs.get("refined_candidates") or []:
                params = {
                    key: int(candidate[key]) for key in LEGACY_SEARCH_SPACE
                }
                selected = params == winner
                refined.append(
                    {
                        **params,
                        "score": float(candidate["score"]),
                        "selected": selected,
                        "rejection_reason": (
                            None if selected else "lower_score_or_locked_tiebreak"
                        ),
                    }
                )
            audit_experiments.append(
                {
                    "experiment_id": str(experiment_id),
                    "split": str(split),
                    "raw_trials": raw_trials,
                    "refined_candidates": refined,
                    "winner": winner,
                }
            )
    protocol = legacy_search_protocol_manifest()
    promotion_eligible = (
        int(n_trials) == LOCKED_LEGACY_N_TRIALS
        and str(first_date) == LOCKED_CALIBRATION_START
        and int(n_jobs) == 1
    )
    payload = {
        "protocol": protocol,
        "runtime": {
            "n_trials_per_split": int(n_trials),
            "calibration_start": str(first_date),
            "n_jobs": int(n_jobs),
        },
        "promotion_eligible": promotion_eligible,
        "promotion_blockers": (
            []
            if promotion_eligible
            else ["runtime_does_not_match_locked_search_protocol"]
        ),
        "experiments": audit_experiments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
