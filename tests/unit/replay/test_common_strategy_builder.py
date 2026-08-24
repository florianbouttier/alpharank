from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alpharank.replay import CommonStrategyReplayConfig as PublicReplayConfig
from alpharank.replay import build_common_strategy_replay as public_replay_builder
from alpharank.replay.common_strategy import (
    CommonStrategyReplayConfig,
    build_common_strategy_replay,
    build_native_boosting_holdings,
)


def test_common_strategy_builder_is_exposed_by_replay_package() -> None:
    assert PublicReplayConfig is CommonStrategyReplayConfig
    assert public_replay_builder is build_common_strategy_replay


def test_native_holdings_preserve_historical_top_5_and_top_10_outputs() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2025, 1, 1)] * 12,
            "ticker": [f"T{index:02d}" for index in range(12)],
            "score": [float(12 - index) for index in range(12)],
            "future_return_1m": [0.01] * 12,
            "benchmark_future_return_1m": [0.005] * 12,
        }
    )

    holdings = build_native_boosting_holdings(predictions)

    counts = holdings.group_by("strategy").len().sort("strategy")
    assert counts.to_dicts() == [
        {"strategy": "Boosting Top 10", "len": 10},
        {"strategy": "Boosting Top 5", "len": 5},
    ]
    assert holdings.filter(pl.col("strategy") == "Boosting Top 5")["ticker"].to_list() == [
        "T00",
        "T01",
        "T02",
        "T03",
        "T04",
    ]


def test_public_builder_fails_closed_when_required_inputs_are_missing(tmp_path: Path) -> None:
    config = CommonStrategyReplayConfig(
        legacy_run_dir=tmp_path / "legacy",
        boosting_run_dir=tmp_path / "boosting",
        output_dir=tmp_path / "output",
        project_root=tmp_path,
        command_argv=("python", "scripts/build_common_legacy_boosting_replay.py"),
    )

    with pytest.raises(FileNotFoundError, match="data_input_manifest.json"):
        build_common_strategy_replay(config)
