from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.replay.common import (
    gate_boosting_predictions_for_execution_open,
    gate_boosting_predictions_for_holding_membership,
    gate_boosting_predictions_for_pre_execution_blocks,
    standard_v2_cost_model,
    validate_common_v2_replay,
)


def test_common_v2_replay_is_comparison_eligible(tmp_path: Path) -> None:
    holdings = _holdings()
    model = standard_v2_cost_model()
    investable = pl.concat(
        [
            simulate_weighted_portfolio(
                frame,
                transaction_cost_model=model,
                causal_timing_policy="require_explicit",
            )
            for frame in holdings.partition_by("strategy", maintain_order=True)
        ]
    )
    spy = reference_monthly_series(
        investable.filter(pl.col("strategy") == "Legacy"),
        strategy="SPY total return",
        return_column="benchmark_return",
    )
    monthly = pl.concat([investable, spy], how="diagonal_relaxed")
    artifacts = write_common_portfolio_artifacts(
        output_dir=tmp_path,
        holdings=holdings,
        monthly_returns=monthly,
        prefix="common_v2",
    )
    composition_id = "c" * 64
    manifest = {
        "scope": "alpharank_common_v2_replay",
        "comparison_eligible": True,
        "composition_id": composition_id,
        "execution_policy_id": "next_session_open_v1",
        "missing_return_policy": "raise",
        "transaction_cost_model": asdict(model),
        "source_validation": {
            "snapshot": {"passed": True},
            "legacy": {"passed": True},
            "boosting": {"passed": True},
        },
        "artifacts": {
            label: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for label, path in artifacts.items()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_common_v2_replay(
        tmp_path, expected_composition_id=composition_id
    )

    assert report["passed"] is True
    assert report["comparison_eligible"] is True
    assert report["maximum_absolute_reconciliation_error"] == 0.0
    assert report["strategy_count"] == 3


def test_boosting_common_replay_gates_holding_membership_before_ranking() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2018, 12, 1)] * 2,
            "ticker": ["SCG.US", "CI.US"],
            "score": [2.0, 1.0],
        }
    )
    membership = pl.DataFrame(
        {
            "year_month": [date(2018, 12, 1), date(2019, 1, 1)],
            "ticker": ["SCG.US", "CI.US"],
        }
    )

    gated = gate_boosting_predictions_for_holding_membership(
        predictions,
        membership,
    )

    assert gated["ticker"].to_list() == ["CI.US"]


def test_boosting_common_replay_requires_first_session_execution_open() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2018, 12, 1)] * 2,
            "ticker": ["SCG.US", "CI.US"],
            "score": [2.0, 1.0],
        }
    )
    prices = pl.DataFrame(
        {
            "ticker": ["SCG.US", "CI.US", "CI.US"],
            "date": [date(2018, 12, 31), date(2019, 1, 2), date(2019, 1, 31)],
            "open": [47.0, 190.0, 195.0],
        }
    )

    gated = gate_boosting_predictions_for_execution_open(predictions, prices)

    assert gated["ticker"].to_list() == ["CI.US"]


def test_boosting_common_replay_rejects_known_pre_open_suspension() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2023, 4, 1)] * 2,
            "ticker": ["FRC.US", "JPM.US"],
            "score": [2.0, 1.0],
        }
    )
    blocks = pl.DataFrame(
        {
            "terminal_event_id": ["FRC-2023-05-01-FDIC"],
            "ticker": ["FRC.US"],
            "effective_date": [date(2023, 5, 1)],
            "known_at": [datetime(2023, 5, 1, 7, 26, tzinfo=timezone.utc)],
            "entry_allowed": [False],
        }
    )

    gated = gate_boosting_predictions_for_pre_execution_blocks(
        predictions,
        blocks,
    )

    assert gated["ticker"].to_list() == ["JPM.US"]


def _holdings() -> pl.DataFrame:
    rows = []
    for strategy, ticker, realized in (
        ("Legacy", "A", 0.10),
        ("Boosting Top 5", "B", 0.12),
    ):
        rows.append(
            {
                "strategy": strategy,
                "decision_month": date(2024, 1, 1),
                "holding_month": date(2024, 2, 1),
                "ticker": ticker,
                "target_weight": 1.0,
                "realized_return": realized,
                "benchmark_return": 0.05,
                "feature_max_asof_at": datetime(
                    2024, 1, 31, 21, tzinfo=timezone.utc
                ),
                "signal_cutoff_at": datetime(
                    2024, 1, 31, 21, tzinfo=timezone.utc
                ),
                "execution_at": datetime(
                    2024, 2, 1, 14, 30, tzinfo=timezone.utc
                ),
                "first_return_observation_at": datetime(
                    2024, 2, 1, 14, 30, 0, 1, tzinfo=timezone.utc
                ),
                "holding_return_end_at": datetime(
                    2024, 2, 29, 21, tzinfo=timezone.utc
                ),
                "execution_policy_id": "next_session_open_v1",
            }
        )
    return pl.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
