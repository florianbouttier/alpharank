from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.maturity import split_completed_portfolio_months
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_month": [
                date(2026, 6, 1),
                date(2026, 6, 1),
                date(2026, 7, 1),
                date(2026, 7, 1),
            ],
            "ticker": ["A.US", "B.US", "C.US", "D.US"],
            "score": [0.9, 0.8, 0.7, 0.6],
            "future_return_1m": [0.10, 0.05, None, None],
            "benchmark_future_return_1m": [0.02, 0.02, None, None],
        }
    )


def test_score_only_month_is_retained_but_excluded_from_economic_replay() -> None:
    split = split_completed_portfolio_months(_predictions())

    assert split.completed_predictions["decision_month"].unique().to_list() == [
        date(2026, 6, 1)
    ]
    assert split.score_only_predictions["decision_month"].unique().to_list() == [
        date(2026, 7, 1)
    ]
    assert split.manifest["last_completed_decision_month"] == "2026-06-01"
    assert split.manifest["score_only_months"] == ["2026-07-01"]


@pytest.mark.parametrize(
    "benchmark_returns, expected_message",
    [
        ([0.02, None], "Partially observed"),
        ([0.02, 0.03], "Inconsistent"),
    ],
)
def test_ambiguous_benchmark_month_fails_closed(
    benchmark_returns: list[float | None], expected_message: str
) -> None:
    predictions = _predictions().filter(pl.col("decision_month") == date(2026, 6, 1))
    predictions = predictions.with_columns(
        pl.Series("benchmark_future_return_1m", benchmark_returns)
    )

    with pytest.raises(ValueError, match=expected_message):
        split_completed_portfolio_months(predictions)


def test_missing_selected_stock_return_is_not_hidden_by_maturity_split() -> None:
    predictions = _predictions().filter(pl.col("decision_month") == date(2026, 6, 1))
    predictions = predictions.with_columns(
        pl.when(pl.col("ticker") == "A.US")
        .then(None)
        .otherwise(pl.col("future_return_1m"))
        .alias("future_return_1m")
    )
    split = split_completed_portfolio_months(predictions)
    holdings = boosting_predictions_to_holdings(
        split.completed_predictions,
        strategy="Boosting Top 2",
        top_n=2,
    )

    with pytest.raises(ValueError, match="Missing realized return"):
        simulate_weighted_portfolio(
            holdings,
            causal_timing_policy="legacy_month_only",
        )
