from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.multihorizon.data import _add_multihorizon_targets
from alpharank.multihorizon.data import _append_legacy_labels
from alpharank.multihorizon.data import _point_in_time_price_eligibility
from alpharank.multihorizon.legacy_ema import (
    add_active_legacy_oracle_features,
    legacy_winning_pairs,
    point_in_time_fold_features,
)
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.metrics import (
    build_prediction_portfolios,
    score_predictions,
)
from alpharank.multihorizon.pipeline import _prediction_frame, _score_only_panel
from alpharank.multihorizon.splits import (
    PurgedCombinatorialMonthSplit,
    horizon_walk_forward_windows,
)
from alpharank.multihorizon.risk import (
    add_daily_forward_risk_targets,
    build_risk_weighted_backtest,
    capped_inverse_risk_weights,
)
from alpharank.multihorizon.trading import build_monthly_top_n_returns
from alpharank.multihorizon.config import (
    LATEST_COMMON_COMPARISON_PROFILE,
    validate_latest_common_comparison_profile,
)


def test_latest_common_comparison_profile_fails_closed_on_config_drift() -> None:
    config = dict(LATEST_COMMON_COMPARISON_PROFILE)
    config["score_only_end_month"] = "2026-06"
    assert validate_latest_common_comparison_profile(config)["passed"] is True

    config["minimum_monthly_median_dollar_volume"] = 0.0
    validation = validate_latest_common_comparison_profile(config)
    assert validation["passed"] is False
    assert validation["mismatches"] == {
        "minimum_monthly_median_dollar_volume": {
            "expected": 1_000_000.0,
            "observed": 0.0,
        }
    }


def test_future_target_requires_an_exact_calendar_gap() -> None:
    stock = pl.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 4, 1)],
            "last_close": [100.0, 110.0, 130.0],
            "monthly_return": [0.0, 0.1, 130 / 110 - 1],
        }
    )
    index = pl.DataFrame(
        {
            "year_month": [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
            ],
            "index_close": [100.0, 101.0, 102.0, 103.0],
            "index_monthly_return": [0.0, 0.01, 0.01, 0.01],
        }
    )
    result = _add_multihorizon_targets(stock, index, [1])
    assert result["future_return_1m"].to_list()[0] == pytest.approx(0.1)
    assert result["future_return_1m"].to_list()[1:] == [None, None]


def test_price_eligibility_is_month_local_and_rejects_bad_ohlc() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["A.US"] * 4,
            "date": [
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 2, 3),
                date(2020, 2, 4),
            ],
            "adjusted_close": [10.0, 10.1, 10.2, 10.3],
            "close": [10.0, 10.1, 10.2, 10.3],
            "open": [9.9, 10.0, 10.1, 20.0],
            "high": [10.1, 10.2, 10.3, 10.4],
            "low": [9.8, 9.9, 10.0, 10.1],
            "volume": [200_000.0] * 4,
        }
    )

    eligibility = _point_in_time_price_eligibility(
        prices,
        minimum_observations=2,
        minimum_median_dollar_volume=1_000_000.0,
        maximum_ohlc_violation_rate=0.25,
    ).sort("decision_month")

    assert eligibility["_price_eligible"].to_list() == [True, False]


def test_preprocessor_uses_train_only_median_after_monthly_fill() -> None:
    train = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "x": [1.0, 3.0],
            "too_sparse": [None, 1.0],
        }
    )
    future = pl.DataFrame(
        {"decision_month": [date(2020, 3, 1)], "x": [None], "too_sparse": [999.0]}
    )
    preprocessor = fit_fold_preprocessor(train, ["x", "too_sparse"], max_missing_ratio=0.4)
    assert preprocessor.features == ("x",)
    _, matrix = preprocessor.transform(future)
    np.testing.assert_allclose(matrix, [[2.0]])


def test_outer_window_respects_label_maturity_and_purge() -> None:
    windows = horizon_walk_forward_windows(
        list(range(240)),
        horizon=36,
        min_train_months=120,
        validation_months=24,
        test_months=12,
        step_months=12,
    )
    first = windows[0]
    assert len(first.train_months) == 120
    assert first.validation_months[0] - first.train_months[-1] == 36
    assert first.test_months[0] - first.validation_months[-1] == 36


def test_outer_window_can_keep_a_final_partial_test_block() -> None:
    windows = horizon_walk_forward_windows(
        list(range(31)),
        horizon=2,
        min_train_months=12,
        validation_months=6,
        test_months=6,
        step_months=6,
        include_partial_test_window=True,
    )
    assert [len(window.test_months) for window in windows] == [6, 5]
    assert windows[-1].test_months == tuple(range(26, 31))


def test_inner_cpcv_removes_overlapping_label_intervals() -> None:
    months = [month for month in range(24) for _ in range(2)]
    splitter = PurgedCombinatorialMonthSplit(months, horizon=6, n_groups=4)
    for train_idx, test_idx in splitter.split():
        train_months = np.asarray(months)[train_idx]
        test_months = np.asarray(months)[test_idx]
        for train_month in train_months:
            assert not np.any(
                (train_month <= test_months + 6) & (train_month + 6 >= test_months)
            )


def test_legacy_teacher_join_does_not_leave_raw_weight_as_a_feature(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["A.US"],
            "decision_month": [date(2020, 1, 1)],
        }
    )
    legacy_path = tmp_path / "legacy.parquet"
    pl.DataFrame(
        {
            "portfolio_model": ["Combined_Frequency"],
            "year_month": [date(2020, 2, 1)],
            "ticker": ["A.US"],
            "n_models": [3],
            "weight_normalized": [0.25],
        }
    ).write_parquet(legacy_path)

    result = _append_legacy_labels(frame, legacy_path)

    assert "weight_normalized" not in result.columns
    assert result["legacy_weight_normalized"].to_list() == [0.25]


def test_regression_report_contains_prediction_error_metrics() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 4,
            "ticker": ["A", "B", "C", "D"],
            "score": [0.3, 0.2, 0.1, 0.0],
            "future_excess_return_1m": [0.4, 0.1, -0.1, 0.0],
            "future_excess_rank_1m": [1.0, 0.75, 0.25, 0.5],
            "future_return_1m": [0.4, 0.1, -0.1, 0.0],
            "legacy_selected": [1, 0, 0, 0],
        }
    )
    metrics, _ = score_predictions(
        predictions,
        method="regression",
        horizon=1,
        top_n_values=(2,),
    )
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(np.mean((np.array([0.4, 0.1, -0.1, 0.0]) - np.array([0.3, 0.2, 0.1, 0.0])) ** 2))
    )
    assert {"mae", "r2", "ndcg_at_10", "spearman_ic"} <= metrics.keys()


def test_classification_ranking_score_is_separate_from_calibrated_probability() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 4,
            "ticker": ["A", "B", "C", "D"],
            "score": [0.9, 0.8, 0.2, 0.1],
            "calibrated_probability": [0.5, 0.5, 0.5, 0.5],
            "future_excess_return_1m": [0.4, 0.3, -0.1, -0.2],
            "future_excess_rank_1m": [1.0, 0.9, 0.2, 0.1],
            "future_return_1m": [0.4, 0.3, -0.1, -0.2],
            "legacy_selected": [1, 0, 0, 0],
        }
    )

    metrics, _ = score_predictions(
        predictions,
        method="classification",
        horizon=1,
        top_n_values=(2,),
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.25)


def test_score_only_tail_builds_portfolio_without_a_mature_h6_target() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2026, 5, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "score": [3.0, 2.0, 1.0],
            "legacy_selected": [1, 0, 0],
            "future_excess_return_6m": [None, None, None],
            "future_excess_return_1m": [0.10, 0.02, -0.04],
        }
    )

    portfolio = build_prediction_portfolios(
        predictions,
        horizon=6,
        top_n_values=(2,),
    )

    assert portfolio["future_excess_return"][0] is None
    assert portfolio["realized_one_month_excess"][0] == pytest.approx(0.06)


def test_score_only_panel_stops_at_explicit_complete_decision_month() -> None:
    frame = pl.DataFrame(
        {
            "decision_month": [
                date(2026, 5, 1),
                date(2026, 6, 1),
                date(2026, 7, 1),
            ],
            "ticker": ["A", "A", "A"],
            "future_excess_return_1m": [0.01, 0.02, 0.03],
            "legacy_label_available": [1, 1, 1],
        }
    )

    panel = _score_only_panel(
        frame,
        method="classification",
        feature_mode="legacy_winners_pit_ema_only",
        end_month="2026-06",
    )

    assert panel is not None
    assert panel["decision_month"].to_list() == [
        date(2026, 5, 1),
        date(2026, 6, 1),
    ]


def test_prediction_status_separates_ticker_gaps_from_horizon_maturity() -> None:
    class Fitted:
        @staticmethod
        def predict_raw_score(matrix):
            return np.arange(len(matrix), dtype=float)

        @staticmethod
        def predict(matrix):
            return np.full(len(matrix), 0.5)

    source = pl.DataFrame(
        {
            "decision_month": [date(2025, 12, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "legacy_selected": [0, 0, 0],
            "future_excess_return_6m": [0.1, None, None],
            "benchmark_future_return_6m": [0.05, 0.05, None],
        }
    )

    predictions = _prediction_frame(
        source=source,
        matrix=np.zeros((3, 1)),
        fitted=Fitted(),
        fold=1,
        method="classification",
        horizon=6,
    )

    assert predictions["target_status"].to_list() == [
        "evaluable",
        "ticker_target_unavailable",
        "horizon_pending",
    ]


def test_monthly_trading_backtest_applies_turnover_cost() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 2, 1),
                date(2020, 2, 1),
            ],
            "ticker": ["A", "B", "C", "A", "B", "C"],
            "score": [3.0, 2.0, 1.0, 3.0, 1.0, 2.0],
            "future_return_1m": [0.10, 0.00, -0.10, 0.02, 0.00, 0.04],
            "benchmark_future_return_1m": [0.01] * 6,
        }
    )
    monthly = build_monthly_top_n_returns(
        predictions,
        top_n=2,
        transaction_cost_bps=10.0,
    )
    assert monthly["turnover"].to_list() == pytest.approx([1.0, 0.5])
    assert monthly["net_return"].to_list() == pytest.approx([0.049, 0.0295])


def test_daily_risk_target_uses_only_strictly_future_months() -> None:
    dates = [
        date(2020, 1, 30),
        date(2020, 1, 31),
        date(2020, 2, 3),
        date(2020, 2, 4),
        date(2020, 3, 2),
        date(2020, 3, 3),
        date(2020, 3, 4),
    ]
    prices = pl.DataFrame(
        {
            "ticker": ["A"] * len(dates),
            "date": dates,
            "adjusted_close": [
                100.0,
                110.0,
                121.0,
                108.9,
                119.79,
                107.811,
                118.5921,
            ],
        }
    )
    frame = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1)],
        }
    )
    result = add_daily_forward_risk_targets(
        frame,
        final_price=prices,
        horizons=(1,),
        minimum_daily_observations_per_month=2,
    )
    february_daily_returns = np.asarray([0.1, -0.1])
    expected = np.std(february_daily_returns, ddof=1) * np.sqrt(252.0)
    assert result["future_realized_volatility_1m"][0] == pytest.approx(expected)
    assert result["future_realized_volatility_1m"][1] == pytest.approx(expected)


def test_inverse_risk_weights_respect_cap_and_prefer_lower_risk() -> None:
    weights = capped_inverse_risk_weights(
        [0.10, 0.20, 0.30, 0.40, 0.50],
        maximum_weight=0.30,
    )
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.30 + 1e-12
    assert weights[0] > weights[-1]


def test_sector_diversification_preserves_alpha_order_within_constraints() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 6,
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "predicted_risk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "future_return_1m": [0.01] * 6,
            "benchmark_future_return_1m": [0.005] * 6,
        }
    )
    general = pl.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "GicSector": ["Tech", "Tech", "Tech", "Finance", "Health", "Energy"],
            "Sector": ["Tech", "Tech", "Tech", "Finance", "Health", "Energy"],
        }
    )
    monthly, holdings = build_risk_weighted_backtest(
        predictions,
        general=general,
        strategy="risk_sector",
        top_n=5,
        risk_column="predicted_risk",
        maximum_names_per_sector=2,
        maximum_sector_weight=0.40,
    )
    assert holdings["ticker"].to_list() == ["A", "B", "D", "E", "F"]
    assert monthly["maximum_sector_weight"][0] <= 0.40 + 1e-12


def test_top_ten_equal_allocation_selects_ten_highest_scores() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 12,
            "ticker": [f"T{index:02d}" for index in range(12)],
            "score": [float(12 - index) for index in range(12)],
            "future_return_1m": [0.01] * 12,
            "benchmark_future_return_1m": [0.005] * 12,
        }
    )
    general = pl.DataFrame(
        {
            "ticker": [f"T{index:02d}" for index in range(12)],
            "GicSector": ["Test"] * 12,
            "Sector": ["Test"] * 12,
        }
    )

    monthly, holdings = build_risk_weighted_backtest(
        predictions,
        general=general,
        strategy="alpha_top10_equal",
        top_n=10,
    )

    assert monthly["n_positions"][0] == 10
    assert holdings["ticker"].to_list() == [
        f"T{index:02d}" for index in range(10)
    ]
    assert holdings["portfolio_weight"].to_list() == pytest.approx([0.1] * 10)


def _write_legacy_winner_fixture(path) -> None:
    pl.DataFrame(
        {
            "portfolio_model": [
                "Legacy_Optuna_11",
                "Legacy_Optuna_11",
                "Legacy_Optuna_12",
                "Combined_Frequency",
            ],
            "year_month": [
                date(2020, 2, 1),
                date(2021, 2, 1),
                date(2020, 2, 1),
                date(2020, 2, 1),
            ],
            "n_short": [5, 7, 12, None],
            "n_long": [257, 333, 150, None],
        }
    ).write_parquet(path)


def test_point_in_time_winner_features_exclude_future_pairs(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.parquet"
    _write_legacy_winner_fixture(legacy_path)
    all_features = (
        "relative_ema_ratio_5_257",
        "relative_ema_ratio_5_257_rank_month",
        "relative_ema_ratio_7_333",
        "relative_ema_ratio_12_150",
        "volatility_12m",
    )

    ema_only, pairs = point_in_time_fold_features(
        all_features=all_features,
        legacy_path=legacy_path,
        train_decision_cutoff=date(2020, 12, 1),
        include_non_relative_features=False,
    )
    ema_plus, _ = point_in_time_fold_features(
        all_features=all_features,
        legacy_path=legacy_path,
        train_decision_cutoff=date(2020, 12, 1),
        include_non_relative_features=True,
    )

    assert pairs == ((5, 257), (12, 150))
    assert ema_only == (
        "relative_ema_ratio_5_257",
        "relative_ema_ratio_5_257_rank_month",
        "relative_ema_ratio_12_150",
    )
    assert "relative_ema_ratio_7_333" not in ema_plus
    assert "volatility_12m" in ema_plus


def test_active_legacy_oracle_uses_each_paths_current_pair(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.parquet"
    pl.DataFrame(
        {
            "portfolio_model": [
                "Legacy_Optuna_11",
                "Legacy_Optuna_12",
                "Legacy_Optuna_21",
                "Legacy_Optuna_22",
            ],
            "year_month": [date(2020, 2, 1)] * 4,
            "n_short": [5, 12, 5, 12],
            "n_long": [257, 150, 257, 150],
        }
    ).write_parquet(legacy_path)
    frame = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)],
            "relative_ema_ratio_5_257": [1.25],
            "relative_ema_ratio_12_150": [0.75],
            "relative_ema_ratio_5_257_rank_month": [0.9],
            "relative_ema_ratio_12_150_rank_month": [0.2],
            "relative_ema_ratio_5_257_z_month": [1.0],
            "relative_ema_ratio_12_150_z_month": [-1.0],
            "relative_ema_ratio_5_257_top_quartile": [1],
            "relative_ema_ratio_12_150_top_quartile": [0],
            "relative_ema_ratio_5_257_bottom_quartile": [0],
            "relative_ema_ratio_12_150_bottom_quartile": [1],
        }
    )

    result, feature_columns = add_active_legacy_oracle_features(
        frame,
        legacy_path=legacy_path,
        available_pairs=((5, 257), (12, 150)),
    )

    assert len(feature_columns) == 20
    assert result["legacy_active_11_raw"].to_list() == [1.25]
    assert result["legacy_active_12_raw"].to_list() == [0.75]
    assert result["legacy_active_21_rank_month"].to_list() == [0.9]
    assert result["legacy_active_22_bottom_quartile"].to_list() == [1]


def test_legacy_winning_pairs_ignores_combined_basket(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.parquet"
    _write_legacy_winner_fixture(legacy_path)
    assert legacy_winning_pairs(legacy_path) == (
        (5, 257),
        (7, 333),
        (12, 150),
    )
