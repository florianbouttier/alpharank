from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from alpharank.data.processing import IndexDataManager
from alpharank.strategy.legacy import StrategyLearner


def test_return_from_training_exports_selected_optuna_caps() -> None:
    fit = pd.DataFrame(
        {
            "year_month": pd.period_range("2025-01", periods=4, freq="M").tolist() * 2,
            "ticker": ["AAA.US"] * 4 + ["BBB.US"] * 4,
            "date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"] * 2),
            "n_long": [100] * 8,
            "n_short": [20] * 8,
            "n_asset": [30] * 8,
            "dr": [1.02, 1.01, 1.03, 1.01, 1.01, 1.02, 0.99, 1.04],
            "quantile_mtr": [0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8],
            "mtr": [1.1] * 8,
        }
    )
    stocks_filter = pd.DataFrame(
        {
            "year_month": pd.period_range("2024-12", periods=4, freq="M").tolist() * 2,
            "ticker": ["AAA.US"] * 4 + ["BBB.US"] * 4,
        }
    )
    sector = pd.DataFrame({"ticker": ["AAA.US", "BBB.US"], "Sector": ["Technology", "Technology"]})
    index_prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]),
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    components = pd.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US"] * 4,
            "year_month": pd.period_range("2025-01", periods=4, freq="M").tolist() * 2,
        }
    )
    monthly_returns = pd.DataFrame(
        {
            "year_month": pd.period_range("2025-01", periods=4, freq="M").tolist(),
            "monthly_return": [0.0, 0.01, 0.0, 0.02],
        }
    )
    index = IndexDataManager(index_prices, components, monthly_returns_df=monthly_returns, backend="polars")

    output = StrategyLearner.return_from_training(
        df_fiting=fit,
        stocks_filter=stocks_filter,
        sector=sector,
        index=index,
        alpha=2,
        temp=12,
        mode="mean",
        params={"n_asset": 5, "n_max_per_sector": 2},
        backend="polars",
    )

    aggregated = output["aggregated"]
    detailed = output["detailed"]

    assert {"selected_model", "selected_n_asset", "selected_n_max_per_sector"}.issubset(aggregated.columns)
    assert {"selected_model", "selected_n_asset", "selected_n_max_per_sector"}.issubset(detailed.columns)
    assert aggregated["selected_n_asset"].unique().tolist() == [5]
    assert aggregated["selected_n_max_per_sector"].unique().tolist() == [2]
    assert detailed["selected_n_asset"].unique().tolist() == [5]
    assert detailed["selected_n_max_per_sector"].unique().tolist() == [2]
    assert aggregated["selected_model"].str.contains(r"\|asset=5\|sector=2").all()
    assert not detailed["sector_constraint_enabled"].any()
    assert set(detailed["sector_constraint_reason"]) == {
        "disabled_no_point_in_time_sector_history"
    }


def test_legacy_sector_cap_uses_pit_sector() -> None:
    months = pd.period_range("2025-01", periods=4, freq="M")
    tickers = ["AAA.US", "BBB.US", "CCC.US"]
    fit_rows: list[dict[str, object]] = []
    filter_rows: list[dict[str, object]] = []
    scores = {"AAA.US": 0.9, "BBB.US": 0.8, "CCC.US": 0.7}
    for ticker in tickers:
        for month in months:
            fit_rows.append(
                {
                    "year_month": month,
                    "ticker": ticker,
                    "date": month.end_time,
                    "n_long": 100,
                    "n_short": 20,
                    "n_asset": 30,
                    "dr": 1.01,
                    "quantile_mtr": scores[ticker],
                    "mtr": 1.1,
                }
            )
            filter_rows.append(
                {
                    "year_month": month - 1,
                    "ticker": ticker,
                }
            )
    index = IndexDataManager(
        pd.DataFrame(
            {
                "date": [month.end_time for month in months],
                "close": [100.0, 101.0, 102.0, 103.0],
            }
        ),
        pd.DataFrame(
            {
                "ticker": tickers * len(months),
                "year_month": [month for month in months for _ in tickers],
            }
        ),
        monthly_returns_df=pd.DataFrame(
            {"year_month": months, "monthly_return": [0.0] * len(months)}
        ),
        backend="polars",
    )
    known_at = datetime(2020, 1, 2, tzinfo=timezone.utc)
    history = pd.DataFrame(
        {
            "ticker": tickers,
            "Sector": ["Technology", "Technology", "Health Care"],
            "classification_id": ["aaa-tech", "bbb-tech", "ccc-health"],
            "source_url": ["https://example.test/sectors"] * 3,
            "confidence": ["official"] * 3,
            "observed_at": [known_at] * 3,
            "effective_at": [known_at] * 3,
        }
    )

    def run(sectors: pd.DataFrame) -> pd.DataFrame:
        return StrategyLearner.return_from_training(
            df_fiting=pd.DataFrame(fit_rows),
            stocks_filter=pd.DataFrame(filter_rows),
            sector=sectors,
            index=index,
            alpha=2,
            temp=12,
            mode="mean",
            params={"n_asset": 2, "n_max_per_sector": 1},
            backend="polars",
        )["detailed"]

    selected = run(history)
    future = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "ticker": ["BBB.US"],
                    "Sector": ["Health Care"],
                    "classification_id": ["bbb-health-future"],
                    "source_url": ["https://example.test/future"],
                    "confidence": ["official"],
                    "observed_at": [datetime(2026, 1, 2, tzinfo=timezone.utc)],
                    "effective_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                }
            ),
        ],
        ignore_index=True,
    )
    mutated = run(future)

    assert set(selected["ticker"]) == {"AAA.US", "CCC.US"}
    assert selected["sector_constraint_enabled"].all()
    assert (
        selected["sector_known_at_selected"] <= selected["decision_at"]
    ).all()
    columns = ["year_month", "ticker", "Sector", "classification_id"]
    pd.testing.assert_frame_equal(
        selected[columns].reset_index(drop=True),
        mutated[columns].reset_index(drop=True),
    )
