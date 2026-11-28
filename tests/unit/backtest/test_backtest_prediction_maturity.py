from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alpharank.backtest import pipeline
from alpharank.backtest.application import ApplicationBacktestConfig, run_application_backtest
from alpharank.backtest.config import BacktestConfig
from alpharank.backtest.datasets import build_model_frame
from alpharank.backtest.features import compute_monthly_index_returns, compute_monthly_stock_prices
from alpharank.backtest.portfolio import select_top_n


def _learn(
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    remove_future_quote: bool,
) -> tuple[BacktestConfig, pipeline.LearningArtifacts]:
    sessions = [
        date(2020, 1, 31),
        date(2020, 2, 28),
        date(2020, 3, 31),
        date(2020, 4, 30),
        date(2020, 5, 29),
        date(2020, 6, 12),
    ]
    stock_prices = pl.DataFrame(
        {
            "ticker": ["A.US"] * 6 + ["B.US"] * 6,
            "date": sessions * 2,
            "adjusted_close": [100.0] * 10 + [110.0, 150.0],
        }
    )
    if remove_future_quote:
        stock_prices = stock_prices.filter(
            (pl.col("ticker") != "A.US") | (pl.col("date") != sessions[4])
        )
    monthly_prices = compute_monthly_stock_prices(stock_prices)
    index_monthly = compute_monthly_index_returns(
        pl.DataFrame({"date": sessions, "adjusted_close": [100.0] * 5 + [120.0]})
    )
    keys = monthly_prices.select("ticker", "year_month")
    constituents = pl.DataFrame({"Ticker": ["A", "B"] * 6, "Date": sorted(sessions * 2)})
    modeling = build_model_frame(
        monthly_prices=monthly_prices,
        technical_features=keys.with_columns(pl.lit(1.0).alias("feature")),
        fundamental_features=keys,
        index_monthly=index_monthly,
        constituents=constituents,
        start_month="2020-01",
        missing_feature_threshold=0.35,
    )

    monkeypatch.setattr(pipeline, "_prepare_modeling_frame", lambda *args, **kwargs: modeling)
    config = BacktestConfig(
        data_dir=output_dir,
        output_dir=output_dir,
        min_train_months=2,
        walk_forward_val_months=1,
        top_n=1,
        outperformance_threshold=0.05,
        fold_min_train_rows=1,
        fold_min_val_rows=1,
        fold_min_test_rows=1,
        shap_sample_size=0,
        save_optuna_all_plots=False,
        verbose=False,
    )
    return config, pipeline.run_learning_phase(config)


def test_future_quote_cannot_change_an_earlier_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, observed = _learn(tmp_path / "observed", monkeypatch, remove_future_quote=False)
    _, missing = _learn(tmp_path / "missing", monkeypatch, remove_future_quote=True)
    april = date(2020, 4, 1)
    observed_scores = observed.predictions.filter(pl.col("decision_month") == april)
    missing_scores = missing.predictions.filter(pl.col("decision_month") == april)

    assert observed_scores.select("ticker", "prediction").equals(
        missing_scores.select("ticker", "prediction")
    )
    assert select_top_n(missing_scores, top_n=1)["ticker"].to_list() == ["A.US"]
    assert missing_scores["target_label"].to_list() == [None, 1]
    assert missing.fold_metrics.filter(pl.col("test_month_start") == str(april))[
        "test_evaluable_rows"
    ].item() == 1
    with pytest.raises(ValueError, match="Missing realized return"):
        pipeline.run_backtest_phase(config, missing)
    selections = pl.read_parquet(missing.run_dir / "selections.parquet")
    assert selections.filter(pl.col("decision_month") == april)["ticker"].to_list() == ["A.US"]


def test_pending_last_decision_is_scored_without_a_kpi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, learning = _learn(tmp_path, monkeypatch, remove_future_quote=False)

    may = date(2020, 5, 1)
    pending = learning.predictions.filter(pl.col("decision_month") == may)
    assert pending["ticker"].to_list() == ["A.US", "B.US"]
    assert pending["holding_month"].to_list() == [date(2020, 6, 1)] * 2
    assert pending["target_label"].to_list() == [None, None]
    last_fold = learning.fold_metrics.filter(pl.col("test_month_start") == str(may))
    assert last_fold.select("test_evaluable_rows", "test_auc").row(0) == (0, None)

    backtest = pipeline.run_backtest_from_learning(config, learning)
    assert backtest.selections["decision_month"].to_list() == [date(2020, 4, 1), may]
    assert backtest.monthly_returns["holding_month"].to_list() == [may]
    assert backtest.split_kpis.filter(pl.col("split") == "test")["n_folds"].item() == 1
    assert backtest.fold_metrics.filter(pl.col("test_month_start") == str(may))[
        "test_auc"
    ].item() is None

    application = run_application_backtest(
        learning.predictions, ApplicationBacktestConfig(name="Test", top_n=1)
    )
    assert application.selections["holding_month"].to_list() == [may, date(2020, 6, 1)]
    assert application.monthly_returns["holding_month"].to_list() == [may]

    score_only = run_application_backtest(pending, application.config)
    assert score_only.selections.height == 1
    assert score_only.monthly_returns.is_empty()
    assert score_only.kpis.is_empty()
