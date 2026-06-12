from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.backtest.portfolio import select_top_n


def test_select_top_n_can_rank_on_custom_score_column() -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "year_month": [date(2024, 1, 1)] * 3,
            "prediction": [0.9, 0.8, 0.1],
            "selection_score": [0.1, 0.8, 0.7],
        }
    )

    selected = select_top_n(frame, top_n=1, score_col="selection_score")

    assert selected.get_column("ticker").to_list() == ["B"]
