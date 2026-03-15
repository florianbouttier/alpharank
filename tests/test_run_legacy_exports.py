from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl
import pytest

from scripts.run_legacy import _get_detailed_output, _indexed_frame_to_polars, _named_frames_to_long


def test_named_frames_to_long_adds_model_label_and_sorts() -> None:
    frame_a = pl.DataFrame(
        {
            "year_month": [date(2020, 2, 1), date(2020, 1, 1)],
            "monthly_return": [0.2, 0.1],
            "ticker": ["BBB.US", "AAA.US"],
        }
    )
    frame_b = pd.DataFrame(
        {
            "year_month": [pd.Period("2020-01", freq="M")],
            "monthly_return": [0.3],
            "ticker": ["CCC.US"],
        }
    )

    out = _named_frames_to_long({"model_b": frame_b, "model_a": frame_a})

    assert out.columns == ["model", "year_month", "monthly_return", "ticker"]
    assert out.get_column("model").to_list() == ["model_a", "model_a", "model_b"]


def test_indexed_frame_to_polars_turns_series_index_into_year_month_column() -> None:
    series = pd.Series(
        [0.1, -0.2],
        index=pd.period_range("2020-01", periods=2, freq="M"),
        name="monthly_return",
    )

    out = _indexed_frame_to_polars(series)

    assert out.columns == ["year_month", "monthly_return"]
    assert out.height == 2


def test_get_detailed_output_accepts_both_legacy_spellings() -> None:
    detailed = pl.DataFrame({"ticker": ["AAA.US"], "year_month": [date(2020, 1, 1)]})

    assert _get_detailed_output({"detailed": detailed}).equals(detailed)
    assert _get_detailed_output({"detailled": detailed}).equals(detailed)


def test_get_detailed_output_raises_if_missing() -> None:
    with pytest.raises(KeyError, match="Missing `detailed`/`detailled` portfolio output"):
        _get_detailed_output({"aggregated": pl.DataFrame()}, label="Legacy_Optuna_11")
