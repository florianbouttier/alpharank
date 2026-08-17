from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from alpharank.governance import (
    SealedConfirmationError,
    create_sealed_confirmation_protocol,
    open_sealed_confirmation,
    register_confirmation_experiment,
    validate_confirmation_for_promotion,
)


NOW = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)


def _create_protocol(tmp_path: Path, name: str) -> Path:
    dataset = tmp_path / name / "sealed_data"
    dataset.mkdir(parents=True)
    (dataset / "returns.parquet").write_bytes(b"unseen final period")
    registry = tmp_path / name / "confirmation.json"
    create_sealed_confirmation_protocol(
        registry_path=registry,
        dataset_dir=dataset,
        period_id="2025-01_2026-06",
        period_start="2025-01-01",
        period_end="2026-06-30",
        expected_experiment_ids=("legacy", "boosting-v2"),
        approved_by="methodology-owner",
        sealed_at=NOW,
    )
    return registry


def _register(registry: Path, experiment_id: str) -> None:
    register_confirmation_experiment(
        registry_path=registry,
        experiment_id=experiment_id,
        hypothesis=f"locked hypothesis for {experiment_id}",
        command=f"run --experiment {experiment_id}",
        config_sha256=f"config-{experiment_id}",
        result_manifest_sha256=f"result-{experiment_id}",
        registered_at=NOW,
    )


def test_sealed_period_is_single_use(tmp_path: Path) -> None:
    premature = _create_protocol(tmp_path, "premature")
    _register(premature, "legacy")
    with pytest.raises(SealedConfirmationError, match="registry was complete"):
        open_sealed_confirmation(
            registry_path=premature,
            opened_by="methodology-owner",
            reason="premature inspection",
            opened_at=NOW,
        )
    premature_state = json.loads(premature.read_text(encoding="utf-8"))
    assert premature_state["status"] == "invalidated"
    assert "missing_experiments:boosting-v2" in premature_state["invalidations"][0]["reason"]

    registry = _create_protocol(tmp_path, "complete")
    _register(registry, "legacy")
    _register(registry, "boosting-v2")
    opened = open_sealed_confirmation(
        registry_path=registry,
        opened_by="methodology-owner",
        reason="all variants locked",
        opened_at=NOW,
    )
    assert opened["status"] == "opened"
    assert {item["experiment_id"] for item in opened["experiments"]} == {
        "legacy",
        "boosting-v2",
    }
    assert validate_confirmation_for_promotion(registry)["period_id"] == "2025-01_2026-06"

    with pytest.raises(SealedConfirmationError, match="cannot be registered"):
        _register(registry, "legacy")
    invalidated = json.loads(registry.read_text(encoding="utf-8"))
    assert invalidated["status"] == "invalidated"
    with pytest.raises(SealedConfirmationError, match="not promotion-eligible"):
        validate_confirmation_for_promotion(registry)
