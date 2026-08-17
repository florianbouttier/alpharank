from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl


HOLDINGS_REQUIRED_COLUMNS = (
    "strategy",
    "decision_month",
    "holding_month",
    "ticker",
    "target_weight",
    "realized_return",
    "benchmark_return",
)

CAUSAL_TIMING_REQUIRED_COLUMNS = (
    "feature_max_asof_at",
    "signal_cutoff_at",
    "execution_at",
    "first_return_observation_at",
    "holding_return_end_at",
)

MONTHLY_REQUIRED_COLUMNS = (
    "strategy",
    "decision_month",
    "holding_month",
    "gross_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "benchmark_return",
    "active_return",
    "relative_return",
    "n_positions",
)


def _missing(columns: Iterable[str], available: Iterable[str]) -> list[str]:
    existing = set(available)
    return [column for column in columns if column not in existing]


def validate_holdings(frame: pl.DataFrame, *, weight_tolerance: float = 1e-9) -> None:
    """Validate the neutral holdings contract without changing the frame."""

    missing = _missing(HOLDINGS_REQUIRED_COLUMNS, frame.columns)
    if missing:
        raise ValueError(f"Missing holdings contract columns: {', '.join(missing)}")
    if frame.is_empty():
        return
    if frame.select(pl.col("ticker").is_null().any()).item():
        raise ValueError("Holdings contain a null ticker.")
    if frame.select((pl.col("target_weight") < 0.0).any()).item():
        raise ValueError("Long-only holdings cannot contain negative target weights.")
    if frame.select(~pl.col("target_weight").is_finite().all()).item():
        raise ValueError("Holdings contain non-finite target weights.")
    duplicate_count = (
        frame.group_by(["strategy", "decision_month", "holding_month", "ticker"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError("Holdings contain duplicate strategy/month/ticker rows.")
    invalid_months = (
        frame.group_by(["strategy", "decision_month", "holding_month"])
        .agg(pl.col("target_weight").sum().alias("weight_sum"))
        .filter((pl.col("weight_sum") - 1.0).abs() > weight_tolerance)
    )
    if invalid_months.height:
        example = invalid_months.row(0, named=True)
        raise ValueError(
            "Target weights must sum to one for every portfolio month; "
            f"first invalid row={example}."
        )
    invalid_timing = frame.filter(
        pl.col("holding_month") != pl.col("decision_month").dt.offset_by("1mo")
    )
    if invalid_timing.height:
        raise ValueError("Every holding_month must equal decision_month + one month.")


def validate_causal_timing(frame: pl.DataFrame) -> None:
    """Enforce feature -> signal -> trade -> realized-return chronology."""

    missing = _missing(CAUSAL_TIMING_REQUIRED_COLUMNS, frame.columns)
    if missing:
        raise ValueError(
            "Missing causal timing columns: " + ", ".join(missing)
        )
    if frame.is_empty():
        return
    timing = frame.with_columns(
        [
            pl.col(column)
            .cast(pl.Datetime(time_zone="UTC"), strict=False)
            .alias(column)
            for column in CAUSAL_TIMING_REQUIRED_COLUMNS
        ]
    )
    null_timing = timing.filter(
        pl.any_horizontal(
            [pl.col(column).is_null() for column in CAUSAL_TIMING_REQUIRED_COLUMNS]
        )
    )
    if not null_timing.is_empty():
        raise ValueError("Causal timing columns cannot contain null values.")
    invalid_order = timing.filter(
        ~(
            (pl.col("feature_max_asof_at") <= pl.col("signal_cutoff_at"))
            & (pl.col("signal_cutoff_at") < pl.col("execution_at"))
            & (pl.col("execution_at") < pl.col("first_return_observation_at"))
            & (
                pl.col("first_return_observation_at")
                <= pl.col("holding_return_end_at")
            )
        )
    )
    if not invalid_order.is_empty():
        raise ValueError(
            "Causal timing must satisfy feature <= signal < execution < "
            "first return observation <= holding end."
        )
    wrong_signal_month = timing.filter(
        pl.col("signal_cutoff_at").dt.date().dt.truncate("1mo")
        != pl.col("decision_month")
    )
    if not wrong_signal_month.is_empty():
        raise ValueError("Signal cutoff must occur inside decision_month.")
    wrong_holding_month = timing.filter(
        (pl.col("execution_at").dt.date().dt.truncate("1mo") != pl.col("holding_month"))
        | (
            pl.col("holding_return_end_at").dt.date().dt.truncate("1mo")
            != pl.col("holding_month")
        )
    )
    if not wrong_holding_month.is_empty():
        raise ValueError(
            "Execution and holding-return end must occur inside holding_month."
        )

    if "return_resolution" in timing.columns:
        terminal = timing.filter(
            pl.col("return_resolution") == "resolved_terminal_event"
        )
        if not terminal.is_empty():
            terminal_required = (
                "terminal_event_id",
                "terminal_effective_date",
                "terminal_event_known_at",
            )
            missing_terminal = _missing(terminal_required, timing.columns)
            if missing_terminal:
                raise ValueError(
                    "Resolved terminal returns lack timing lineage: "
                    + ", ".join(missing_terminal)
                )
            terminal = terminal.with_columns(
                pl.col("terminal_effective_date").cast(pl.Date, strict=False),
                pl.col("terminal_event_known_at").cast(
                    pl.Datetime(time_zone="UTC"), strict=False
                ),
            )
            invalid_terminal = terminal.filter(
                pl.col("terminal_event_id").is_null()
                | pl.col("terminal_effective_date").is_null()
                | pl.col("terminal_event_known_at").is_null()
                | (
                    pl.col("terminal_effective_date")
                    < pl.col("execution_at").dt.date()
                )
                | (
                    pl.col("terminal_effective_date")
                    > pl.col("holding_return_end_at").dt.date()
                )
            )
            if not invalid_terminal.is_empty():
                raise ValueError(
                    "Resolved terminal event must be sourced and effective during "
                    "the holding period."
                )


def validate_monthly_returns(frame: pl.DataFrame) -> None:
    missing = _missing(MONTHLY_REQUIRED_COLUMNS, frame.columns)
    if missing:
        raise ValueError(f"Missing monthly return columns: {', '.join(missing)}")
    if frame.is_empty():
        return
    duplicates = (
        frame.group_by(["strategy", "decision_month", "holding_month"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicates.height:
        raise ValueError("Monthly returns contain duplicate strategy/month rows.")
    invalid_timing = frame.filter(
        pl.col("holding_month") != pl.col("decision_month").dt.offset_by("1mo")
    )
    if invalid_timing.height:
        raise ValueError("Monthly return timing violates decision t -> holding t+1.")
    for column in (
        "gross_return",
        "turnover",
        "transaction_cost",
        "net_return",
        "benchmark_return",
        "active_return",
        "relative_return",
    ):
        values = frame[column].to_numpy().astype(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Monthly return column {column!r} contains non-finite values.")


def empty_monthly_returns() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "strategy": pl.Utf8,
            "decision_month": pl.Date,
            "holding_month": pl.Date,
            "gross_return": pl.Float64,
            "turnover": pl.Float64,
            "transaction_cost": pl.Float64,
            "cost_scenario_id": pl.Utf8,
            "spread_cost": pl.Float64,
            "slippage_cost": pl.Float64,
            "impact_cost": pl.Float64,
            "commission_cost": pl.Float64,
            "fx_cost": pl.Float64,
            "net_return": pl.Float64,
            "benchmark_return": pl.Float64,
            "active_return": pl.Float64,
            "relative_return": pl.Float64,
            "n_positions": pl.Int64,
            "maximum_position_weight": pl.Float64,
            "maximum_sector_weight": pl.Float64,
            "sector_count": pl.Int64,
        }
    )
