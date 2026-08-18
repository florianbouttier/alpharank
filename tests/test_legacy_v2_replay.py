from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.legacy_v2 import validate_legacy_v2_replay
from alpharank.portfolio.costs import TransactionCostModel
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def test_legacy_v2_run_is_replayable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    holdings = _holdings()
    models = (
        TransactionCostModel("zero"),
        TransactionCostModel("standard_10bps", commission_bps=10.0),
        TransactionCostModel("stress_30bps", commission_bps=30.0),
    )
    monthly = pl.concat(
        [
            simulate_weighted_portfolio(
                holdings,
                transaction_cost_model=model,
                causal_timing_policy="require_explicit",
            )
            for model in models
        ]
    )
    holdings_path = run_dir / "legacy_v2_holdings.parquet"
    monthly_path = run_dir / "legacy_v2_monthly.parquet"
    holdings.write_parquet(holdings_path)
    monthly.write_parquet(monthly_path)
    composition_id = "a" * 64
    (run_dir / "data_input_manifest.json").write_text(
        json.dumps(
            {
                "run_config": {
                    "n_trials": 30,
                    "n_jobs": 1,
                    "price_eligibility_policy_id": "monthly_price_eligibility_v1",
                    "methodology_identity": {
                        "methodology_version": "v2-causal",
                        "composition_id": composition_id,
                    },
                },
                "runtime_provenance": {"git": {"dirty": False}},
            }
        ),
        encoding="utf-8",
    )
    replay_manifest = {
        "scope": "alpharank_legacy_v2_replay",
        "execution_policy": {"identifier": "next_session_open_v1"},
        "missing_return_policy": "raise",
        "canonical_cost_scenario_id": "standard_10bps",
        "cost_scenarios": [asdict(model) for model in models],
        "artifacts": {
            "holdings": {"path": str(holdings_path), "sha256": _sha256(holdings_path)},
            "monthly": {"path": str(monthly_path), "sha256": _sha256(monthly_path)},
        },
    }
    (run_dir / "legacy_v2_replay_manifest.json").write_text(
        json.dumps(replay_manifest), encoding="utf-8"
    )

    report = validate_legacy_v2_replay(
        run_dir, expected_composition_id=composition_id
    )

    assert report["passed"] is True
    assert report["maximum_absolute_replay_error"] == 0.0
    assert report["scenario_count"] == 3
    monthly.with_columns((pl.col("net_return") + 0.01).alias("net_return")).write_parquet(
        monthly_path
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_legacy_v2_replay(run_dir, expected_composition_id=composition_id)


def _holdings() -> pl.DataFrame:
    signal = datetime(2024, 1, 31, 21, tzinfo=timezone.utc)
    execution = datetime(2024, 2, 1, 14, 30, tzinfo=timezone.utc)
    first = datetime(2024, 2, 1, 14, 30, 0, 1, tzinfo=timezone.utc)
    ending = datetime(2024, 2, 29, 21, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "strategy": ["Combined_Frequency"],
            "decision_month": [date(2024, 1, 1)],
            "holding_month": [date(2024, 2, 1)],
            "ticker": ["A.US"],
            "target_weight": [1.0],
            "realized_return": [0.1],
            "benchmark_return": [0.05],
            "feature_max_asof_at": [signal],
            "signal_cutoff_at": [signal],
            "execution_at": [execution],
            "first_return_observation_at": [first],
            "holding_return_end_at": [ending],
            "execution_policy_id": ["next_session_open_v1"],
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
