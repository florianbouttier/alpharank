from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.replay import (
    ReplayValidationError,
    create_recomputable_replay_package,
    validate_and_recompute_replay_package,
)


def _holdings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": ["sealed", "sealed"],
            "decision_month": [date(2025, 1, 1), date(2025, 1, 1)],
            "holding_month": [date(2025, 2, 1), date(2025, 2, 1)],
            "ticker": ["AAA", "BBB"],
            "target_weight": [0.5, 0.5],
            "realized_return": [0.10, -0.02],
            "benchmark_return": [0.03, 0.03],
            "feature_max_asof_at": [datetime(2025, 1, 30, tzinfo=timezone.utc)] * 2,
            "signal_cutoff_at": [datetime(2025, 1, 31, tzinfo=timezone.utc)] * 2,
            "execution_at": [datetime(2025, 2, 3, 14, 30, tzinfo=timezone.utc)] * 2,
            "first_return_observation_at": [datetime(2025, 2, 3, 14, 31, tzinfo=timezone.utc)] * 2,
            "holding_return_end_at": [datetime(2025, 2, 28, 21, 0, tzinfo=timezone.utc)] * 2,
        }
    )


def test_replay_recomputes_outputs_from_sealed_inputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = tmp_path / "model.json"
    model.write_text('{"model":"fixture","version":1}\n', encoding="utf-8")
    config = {
        "transaction_cost_bps": 10.0,
        "missing_return_policy": "raise",
        "causal_timing_policy": "require_explicit",
    }

    for mutated_role in (None, "code", "config", "input", "model"):
        package = tmp_path / f"package-{mutated_role or 'clean'}"
        create_recomputable_replay_package(
            package,
            holdings=_holdings(),
            config=config,
            model_path=model,
            project_root=project_root,
        )
        if mutated_role is None:
            report = validate_and_recompute_replay_package(package)
            assert report["passed"] is True
            assert report["code_file_count"] >= 10
            continue

        artifact = next(
            row
            for row in __import__("json").loads(
                (package / "replay_manifest.json").read_text(encoding="utf-8")
            )["artifacts"]
            if row["role"] == mutated_role
        )
        artifact_path = package / artifact["path"]
        artifact_path.write_bytes(artifact_path.read_bytes() + b"mutation")
        with pytest.raises(ReplayValidationError, match=mutated_role):
            validate_and_recompute_replay_package(package)
