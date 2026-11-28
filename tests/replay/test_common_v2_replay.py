from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.opening_orders import evaluate_opening_intents
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.terminal_price_registry import TerminalPriceRegistry
from alpharank.replay.common import (
    build_common_v2_comparison,
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


@pytest.mark.parametrize("missing_open", [True, False])
def test_common_v2_replay_preserves_the_selected_opening_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_open: bool
) -> None:
    _write_opening_replay_inputs(tmp_path, monkeypatch, missing_open=missing_open)
    output_dir = tmp_path / "replay"
    replay_inputs = {
        "legacy_run_dir": tmp_path,
        "boosting_run_dir": tmp_path,
        "causal_snapshot_dir": tmp_path,
        "output_dir": output_dir,
    }
    if missing_open:
        with pytest.raises(RuntimeError, match="Opening execution blocked"):
            build_common_v2_comparison(**replay_inputs, top_n_values=(1,))
    else:
        assert build_common_v2_comparison(**replay_inputs, top_n_values=(1,))["passed"]

    journal_path = output_dir / "opening_execution_journal.parquet"
    journal = pl.read_parquet(journal_path)
    selected = journal.filter(pl.col("strategy") == "Boosting Top 1").row(0, named=True)
    assert (selected["ticker"], selected["selection_rank"], selected["target_weight"]) == (
        "B.US",
        1,
        1.0,
    )
    assert selected["execution_status"] == (
        "opening_quote_unavailable" if missing_open else "opening_quote_observed"
    )
    assert set(journal["strategy"]) == {"Legacy", "Boosting Top 1"}
    receipt = json.loads((output_dir / "opening_execution_receipt.json").read_text())
    assert receipt["status"] == (
        "blocked_unavailable_opening" if missing_open else "all_openings_observed"
    )
    assert receipt["journal_sha256"] == _sha256(journal_path)
    assert (output_dir / "common_v2_monthly.parquet").exists() is not missing_open


def _write_opening_replay_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, missing_open: bool
) -> None:
    for check in (
        "validate_causal_v2_snapshot",
        "validate_legacy_v2_replay",
        "validate_boosting_v2_replay",
    ):
        monkeypatch.setattr(
            f"alpharank.replay.common.{check}",
            lambda *args, **kwargs: {
                "composition_id": "c" * 64,
                "snapshot_id": "test-snapshot",
                "passed": True,
            },
        )
    monkeypatch.setattr(
        TerminalPriceRegistry, "successor_prices", lambda *args, **kwargs: pl.DataFrame()
    )
    snapshot = tmp_path / "input_snapshot"
    snapshot.mkdir()
    legacy = (
        _holdings()
        .filter(pl.col("strategy") == "Legacy")
        .with_columns(pl.lit("Combined_Frequency").alias("strategy"))
    )
    legacy.write_parquet(tmp_path / "legacy.parquet")
    (tmp_path / "legacy_v2_replay_manifest.json").write_text(
        json.dumps({"artifacts": {"holdings": {"path": str(tmp_path / "legacy.parquet")}}}),
        encoding="utf-8",
    )
    prediction_dir = tmp_path / "classification_h06"
    prediction_dir.mkdir()
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    pl.DataFrame(
        {
            "ticker": ["B.US", "C.US"],
            "decision_month": [date(2024, 1, 1)] * 2,
            "score": [2.0, 1.0],
            "future_return_1m": [None, 0.20],
            "benchmark_future_return_1m": [None, 0.05],
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    (snapshot / "SP500_Constituents.csv").write_text(
        "ticker,date\nB.US,2024-02-01\nC.US,2024-02-01\n", encoding="utf-8"
    )
    prices = pl.DataFrame(
        [
            (ticker, day, 10.0, 10.0, 10.0)
            for ticker in ("A", "B.US", "C.US", "SPY.US")
            for day in (
                "2024-01-31",
                "2024-02-02" if missing_open and ticker == "B.US" else "2024-02-01",
                "2024-02-29",
            )
        ],
        schema=["ticker", "date", "open", "close", "adjusted_close"],
        orient="row",
    )
    prices.filter(pl.col("ticker") != "SPY.US").write_parquet(snapshot / "US_Finalprice.parquet")
    prices.filter(pl.col("ticker") == "SPY.US").write_parquet(snapshot / "SP500Price.parquet")


def test_opening_suspension_is_evaluated_at_execution_without_changing_intents() -> None:
    holdings = _holdings()
    prices = pl.DataFrame(
        {
            "ticker": ["A", "B", "SPY.US"],
            "date": ["2024-02-01"] * 3,
            "open": [10.0] * 3,
            "close": [10.0] * 3,
            "adjusted_close": [10.0] * 3,
        }
    )
    stocks = prices.filter(pl.col("ticker") != "SPY.US")
    benchmark = prices.filter(pl.col("ticker") == "SPY.US")
    suspension = pl.DataFrame(
        {
            "terminal_event_id": ["B-suspension"],
            "ticker": ["B"],
            "effective_date": [date(2024, 2, 1)],
            "known_at": [datetime(2024, 2, 1, 14, 29, tzinfo=timezone.utc)],
            "entry_allowed": [False],
        }
    )
    before_open = evaluate_opening_intents(holdings, stocks, benchmark, suspension)
    after_open = evaluate_opening_intents(
        holdings, stocks, benchmark, suspension.with_columns(pl.col("known_at").dt.offset_by("2m"))
    )

    intended = ["strategy", "ticker", "decision_month", "target_weight", "selection_rank"]
    assert before_open.select(intended).equals(after_open.select(intended))
    selected = before_open.filter(pl.col("ticker") == "B").row(0, named=True)
    assert selected["execution_status"] == "known_pre_open_suspension"
    assert selected["terminal_event_id"] == "B-suspension"
    assert selected["terminal_known_at"] < selected["scheduled_open_at"]
    assert set(after_open["execution_status"]) == {"opening_quote_observed"}
    assert after_open["terminal_event_id"].null_count() == 2


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
