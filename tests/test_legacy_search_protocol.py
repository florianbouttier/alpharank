from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from alpharank.strategy.legacy import StrategyLearner
from alpharank.strategy.search_protocol import (
    LEGACY_SEARCH_SPACE,
    LOCKED_CALIBRATION_START,
    LOCKED_LEGACY_N_TRIALS,
    write_legacy_search_audit,
)


class _RecordingTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.calls.append((name, low, high))
        return low


def _study() -> SimpleNamespace:
    state = SimpleNamespace(name="COMPLETE")
    trials = [
        SimpleNamespace(
            number=1,
            params={
                "n_long": 200,
                "n_short": 20,
                "n_asset": 10,
                "n_max_per_sector": 2,
            },
            value=1.1,
            state=state,
        ),
        SimpleNamespace(
            number=0,
            params={
                "n_long": 100,
                "n_short": 10,
                "n_asset": 5,
                "n_max_per_sector": 1,
            },
            value=1.0,
            state=state,
        ),
    ]
    candidates = [
        {**trials[0].params, "score": 1.1},
        {**trials[1].params, "score": 1.0},
    ]
    return SimpleNamespace(
        trials=trials,
        user_attrs={
            "refined_best_params": trials[0].params,
            "refined_candidates": candidates,
        },
    )


def test_legacy_search_protocol_is_locked(tmp_path: Path) -> None:
    trial = _RecordingTrial()
    sampled = StrategyLearner.sample_space(trial)
    assert trial.calls == [
        (name, int(spec["low"]), int(spec["high"]))
        for name, spec in LEGACY_SEARCH_SPACE.items()
    ]
    assert sampled == {name: int(spec["low"]) for name, spec in LEGACY_SEARCH_SPACE.items()}

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = write_legacy_search_audit(
        output_path=first_path,
        experiments={"11": {"studies": {"2020-01": _study()}}},
        n_trials=LOCKED_LEGACY_N_TRIALS,
        first_date=LOCKED_CALIBRATION_START,
        n_jobs=1,
    )
    second = write_legacy_search_audit(
        output_path=second_path,
        experiments={"11": {"studies": {"2020-01": _study()}}},
        n_trials=LOCKED_LEGACY_N_TRIALS,
        first_date=LOCKED_CALIBRATION_START,
        n_jobs=1,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["promotion_eligible"] is True
    assert first["protocol"]["uses_final_confirmation_for_selection"] is False
    experiment = first["experiments"][0]
    assert experiment["winner"]["n_long"] == 200
    assert [trial["number"] for trial in experiment["raw_trials"]] == [0, 1]
    rejected = [
        candidate
        for candidate in experiment["refined_candidates"]
        if not candidate["selected"]
    ]
    assert rejected[0]["rejection_reason"] == "lower_score_or_locked_tiebreak"

    smoke = write_legacy_search_audit(
        output_path=tmp_path / "smoke.json",
        experiments={},
        n_trials=1,
        first_date=LOCKED_CALIBRATION_START,
        n_jobs=1,
    )
    assert smoke["promotion_eligible"] is False
    assert smoke["promotion_blockers"] == [
        "runtime_does_not_match_locked_search_protocol"
    ]
