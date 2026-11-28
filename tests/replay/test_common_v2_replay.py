from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.portfolio.allocation import drifted_weights_after_returns
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.opening_orders import (
    evaluate_opening_intents,
    evaluate_rebalance_exits,
    record_rebalance_exits,
    value_rebalance_holdings,
)
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


@pytest.mark.parametrize("reviewed_successor_openings", [False, True])
def test_common_v2_open_to_open_uses_the_next_shared_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reviewed_successor_openings: bool
) -> None:
    _write_opening_replay_inputs(tmp_path, monkeypatch, missing_open=False)
    snapshot = tmp_path / "input_snapshot"
    for name, quotes in (
        ("US_Finalprice.parquet", [("A", 12.0), ("B.US", 9.0)]),
        ("SP500Price.parquet", [("SPY.US", 11.0)]),
    ):
        path = snapshot / name
        initial = pl.read_parquet(path)
        exits = pl.DataFrame(
            [(ticker, "2024-03-01", opening, 10.0, 10.0) for ticker, opening in quotes],
            schema=initial.schema,
            orient="row",
        )
        pl.concat([initial, exits]).write_parquet(path)
    output_dir = tmp_path / "open_to_open"

    report = build_common_v2_comparison(
        legacy_run_dir=tmp_path,
        boosting_run_dir=tmp_path,
        causal_snapshot_dir=tmp_path,
        output_dir=output_dir,
        top_n_values=(1,),
        holding_return_policy="rebalance_open_to_rebalance_open_v1",
        rank_buffer=1,
        reviewed_successor_openings=reviewed_successor_openings,
    )

    assert report["passed"] is True
    monthly = pl.read_parquet(output_dir / "common_v2_monthly.parquet")
    assert report["status"] == "candidate_open_to_open_replay"
    assert report["promotion_eligible"] is False
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["boosting_rank_buffer"] == 1
    assert ("terminal_successor_opening_registry" in manifest["sources"]) is reviewed_successor_openings
    metrics = {row["strategy"]: row for row in monthly.to_dicts()}
    assert metrics["Legacy"]["gross_return"] == pytest.approx(0.20)
    assert metrics["Legacy"]["net_return"] == pytest.approx(1.20 * 0.999 - 1.0)
    assert metrics["Boosting Top 1"]["gross_return"] == pytest.approx(-0.10)
    assert metrics["Boosting Top 1 | Rank buffer +1"]["gross_return"] == pytest.approx(-0.10)
    assert metrics["SPY total return"]["net_return"] == pytest.approx(0.10)
    assert monthly["benchmark_return"].to_list() == pytest.approx([0.10] * 4)


def test_open_to_open_cash_settlement_and_missing_successor(tmp_path: Path) -> None:
    intents = _holdings().filter(pl.col("ticker") == "A").select(
        "strategy", "decision_month", "holding_month", "ticker", "target_weight",
        "realized_return", "benchmark_return",
    )
    prices = pl.DataFrame({"ticker": ["A"], "date": ["2024-02-01"], "open": [10.0]})
    prices = prices.with_columns(pl.col("open").alias("close"), pl.col("open").alias("adjusted_close"))
    benchmark = pl.concat([
        prices.with_columns(pl.lit("SPY.US").alias("ticker"), pl.lit(day).alias("date"))
        for day in ("2024-01-31", "2024-02-01", "2024-03-01")
    ])
    openings = evaluate_opening_intents(intents, prices, benchmark, pl.DataFrame())
    event = pl.DataFrame({
        "ticker": ["A"],
        "terminal_event_id": ["A-transaction"],
        "event_type": ["cash_merger"],
        "effective_date": [date(2024, 2, 12)],
        "known_at": [datetime(2024, 2, 12, 20, tzinfo=timezone.utc)],
        "source": ["reviewed source"],
        "source_url": ["https://example.com"],
        "price_vintage_id": ["reviewed vintage"],
        "cash_per_share": [12.0],
        "recovery_per_share": [None],
        "distribution_per_share": [0.0],
    })

    cash = evaluate_rebalance_exits(openings, prices, benchmark, event)
    assert cash["valuation_status"].item() == "resolved_terminal_event"
    valuation = value_rebalance_holdings(intents, cash).row(0, named=True)
    assert valuation["realized_return"] == pytest.approx(0.20)
    assert valuation["ending_ticker"] == "__CASH__"

    stock = evaluate_rebalance_exits(
        openings, prices, benchmark, event.with_columns(pl.lit("stock_merger").alias("event_type"))
    )
    with pytest.raises(RuntimeError, match="Rebalance valuation blocked"):
        record_rebalance_exits(stock, tmp_path)
    persisted = pl.read_parquet(tmp_path / "rebalance_valuation_journal.parquet")
    assert persisted["valuation_status"].item() == "unsupported_terminal_exit"
    assert persisted["exit_event_id"].item() == "A-transaction"

    pending = evaluate_rebalance_exits(openings, prices, benchmark.head(2), event)
    assert pending["valuation_status"].item() == "pending_next_market_open"


def test_successor_and_cash_merge_into_drifted_rebalance_weights() -> None:
    weights = drifted_weights_after_returns(
        {"A": 0.5, "B": 0.5},
        {"A": 0.0, "B": -0.20},
        ending_tickers={"A": "B"},
        ending_cash_fractions={"A": 0.20},
    )
    assert weights == pytest.approx({"B": 0.80 / 0.90, "__CASH__": 0.10 / 0.90})
    holdings = pl.DataFrame({
        "strategy": ["Legacy"] * 3,
        "decision_month": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 2, 1)],
        "holding_month": [date(2024, 2, 1), date(2024, 2, 1), date(2024, 3, 1)],
        "ticker": ["A", "B", "B"],
        "target_weight": [0.5, 0.5, 1.0],
        "realized_return": [0.0, -0.20, 0.0],
        "benchmark_return": [0.0, 0.0, 0.0],
        "ending_ticker": ["B", "B", "B"],
        "ending_cash_fraction": [0.20, 0.0, 0.0],
    })
    monthly = simulate_weighted_portfolio(
        holdings, transaction_cost_bps=10.0, causal_timing_policy="legacy_month_only"
    )
    assert monthly["turnover"].to_list() == pytest.approx([1.0, 0.10 / 0.90])
    assert monthly["transaction_cost"].to_list() == pytest.approx([0.001, 0.001 * 0.10 / 0.90])


def test_reviewed_stock_settlement_uses_exact_raw_open(tmp_path: Path) -> None:
    intents = _holdings().filter((pl.col("strategy") == "Legacy") & (pl.col("ticker") == "A")).head(1)
    prices = pl.DataFrame({
        "ticker": ["A", "B"], "date": ["2024-02-01", "2024-03-01"],
        "open": [10.0, 16.0], "close": [10.0, 16.0], "adjusted_close": [10.0, 8.0],
        "volume": [400, 500],
    })
    benchmark = pl.concat([
        prices.head(1).with_columns(pl.lit("SPY.US").alias("ticker"), pl.lit(day).alias("date"))
        for day in ("2024-01-31", "2024-02-01", "2024-03-01")
    ])
    openings = evaluate_opening_intents(intents, prices, benchmark, pl.DataFrame())
    event = pl.DataFrame({
        "ticker": ["A"], "terminal_event_id": ["exchange"], "event_type": ["stock_merger"],
        "effective_date": [date(2024, 2, 12)],
        "known_at": [datetime(2024, 2, 12, 20, tzinfo=timezone.utc)],
        "source": ["reviewed"], "source_url": ["https://example.com"], "price_vintage_id": ["sealed"],
        "cash_per_share": [11.0], "recovery_per_share": [None], "distribution_per_share": [1.0],
    })
    review = pl.DataFrame({
        "exit_event_id": ["exchange"], "exit_date": [date(2024, 3, 1)], "successor_ticker": ["B"],
        "successor_ratio": [0.5], "reviewed_successor_open": [16.0], "reviewed_successor_volume": [500.0],
    })
    exits = evaluate_rebalance_exits(openings, prices, benchmark, event, reviewed_successor_openings=review)
    valuation = value_rebalance_holdings(intents, exits).row(0, named=True)
    assert valuation["realized_return"] == pytest.approx(1.0)
    assert valuation["terminal_value_per_share"] == pytest.approx(20.0)
    assert valuation["terminal_successor_end_price"] == pytest.approx(16.0)
    assert valuation["ending_ticker"] == "B"
    assert valuation["ending_cash_fraction"] == pytest.approx(0.60)

    wrong_mark = prices.with_columns(
        pl.when(pl.col("ticker") == "B").then(8.0).otherwise(pl.col("open")).alias("open")
    )
    wrong_volume = prices.with_columns(pl.lit(400).alias("volume"))
    for refused_quotes in (prices.head(1), wrong_mark, wrong_volume):
        refused = evaluate_rebalance_exits(
            openings, refused_quotes, benchmark, event, reviewed_successor_openings=review
        )
        with pytest.raises(RuntimeError, match="Rebalance valuation blocked"):
            record_rebalance_exits(refused, tmp_path)
        persisted = pl.read_parquet(tmp_path / "rebalance_valuation_journal.parquet")
        assert persisted["valuation_status"].item() == "terminal_successor_open_unavailable"


def test_reviewed_successor_openings_require_open_to_open(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Reviewed successor openings require the open-to-open policy"):
        build_common_v2_comparison(
            legacy_run_dir=tmp_path, boosting_run_dir=tmp_path, causal_snapshot_dir=tmp_path,
            output_dir=tmp_path / "unused", reviewed_successor_openings=True,
        )
    assert not (tmp_path / "unused").exists()


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
