from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Sequence

import polars as pl

from alpharank.portfolio.performance import advanced_performance_statistics


def align_return_series(
    series: Mapping[str, pl.DataFrame],
    *,
    month_column: str = "holding_month",
    return_column: str = "net_return",
    how: str = "inner",
) -> pl.DataFrame:
    """Align named monthly return series on one explicit calendar."""

    if how not in {"inner", "full"}:
        raise ValueError("how must be 'inner' or 'full'.")
    aligned: pl.DataFrame | None = None
    for name, frame in series.items():
        missing = [
            column for column in (month_column, return_column) if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Series {name!r} is missing columns: {', '.join(missing)}")
        current = (
            frame.select(
                pl.col(month_column).cast(pl.Date),
                pl.col(return_column).cast(pl.Float64).alias(name),
            )
            .unique(month_column)
            .sort(month_column)
        )
        aligned = current if aligned is None else aligned.join(current, on=month_column, how=how)
    if aligned is None:
        return pl.DataFrame(schema={month_column: pl.Date})
    return aligned.sort(month_column)


def reference_monthly_series(
    source: pl.DataFrame,
    *,
    strategy: str,
    return_column: str,
    benchmark_column: str = "benchmark_return",
) -> pl.DataFrame:
    """Represent Legacy or SPY as a canonical monthly comparison series."""

    required = {"decision_month", "holding_month", return_column, benchmark_column}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Reference source is missing: {', '.join(missing)}")
    return (
        source.select(
            "decision_month",
            "holding_month",
            pl.col(return_column).cast(pl.Float64).alias("net_return"),
            pl.col(benchmark_column).cast(pl.Float64).alias("benchmark_return"),
        )
        .unique("holding_month")
        .with_columns(
            pl.lit(strategy).alias("strategy"),
            pl.col("net_return").alias("gross_return"),
            pl.lit(0.0).alias("turnover"),
            pl.lit(0.0).alias("transaction_cost"),
            (pl.col("net_return") - pl.col("benchmark_return")).alias("active_return"),
            ((1.0 + pl.col("net_return")) / (1.0 + pl.col("benchmark_return")) - 1.0).alias(
                "relative_return"
            ),
            pl.lit(0).cast(pl.Int64).alias("n_positions"),
            pl.lit(0.0).alias("maximum_position_weight"),
            pl.lit(0.0).alias("maximum_sector_weight"),
            pl.lit(0).cast(pl.Int64).alias("sector_count"),
        )
        .select(
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
            "maximum_position_weight",
            "maximum_sector_weight",
            "sector_count",
        )
        .sort("holding_month")
    )


def subperiod_metric_grid(
    monthly_wide: pl.DataFrame,
    *,
    strategy_columns: Sequence[str],
    benchmark_column: str,
    metric_fields: Sequence[str],
    month_column: str = "holding_month",
) -> dict[str, list[list[float | None]]]:
    """Calculate one canonical metric grid for every inclusive month range."""

    required = {
        month_column,
        benchmark_column,
        *strategy_columns,
    }
    missing = sorted(required - set(monthly_wide.columns))
    if missing:
        raise ValueError("Comparison grid is missing columns: " + ", ".join(missing))
    comparison_columns = list(dict.fromkeys([benchmark_column, *strategy_columns]))
    frame = monthly_wide.select(month_column, *comparison_columns).sort(month_column)
    months = frame[month_column].to_list()
    benchmark = frame[benchmark_column].to_numpy()
    output: dict[str, list[list[float | None]]] = {}
    for start in range(len(months)):
        for end in range(start, len(months)):
            rows: list[list[float | None]] = []
            for strategy_column in strategy_columns:
                statistics = advanced_performance_statistics(
                    frame[strategy_column].to_numpy()[start : end + 1],
                    benchmark_returns=benchmark[start : end + 1],
                )
                rows.append([statistics[field] for field in metric_fields])
            output[f"{months[start].isoformat()}|{months[end].isoformat()}"] = rows
    return output


def performance_by_start_year(
    series_by_strategy: Mapping[str, pl.DataFrame],
    *,
    benchmark_strategy: str,
    strategy_order: Sequence[str],
    first_year: int,
    end_month: date,
    month_column: str = "holding_month",
    return_column: str = "net_return",
) -> pl.DataFrame:
    """Compare named monthly series from each requested January to one end."""

    missing_strategies = sorted(set(strategy_order) - set(series_by_strategy))
    if benchmark_strategy not in series_by_strategy:
        missing_strategies.append(benchmark_strategy)
    if missing_strategies:
        raise ValueError(
            "Comparison is missing strategies: " + ", ".join(sorted(set(missing_strategies)))
        )
    benchmark = series_by_strategy[benchmark_strategy].select(
        month_column,
        pl.col(return_column).alias("benchmark_return"),
    )
    rows: list[dict[str, Any]] = []
    for start_year in range(first_year, end_month.year + 1):
        requested_start = date(start_year, 1, 1)
        for strategy in strategy_order:
            frame = (
                series_by_strategy[strategy]
                .filter(
                    pl.col(month_column).is_between(
                        requested_start,
                        end_month,
                    )
                )
                .join(benchmark, on=month_column, how="inner")
                .sort(month_column)
            )
            if frame.is_empty():
                continue
            effective_start = frame[month_column].min()
            metrics = advanced_performance_statistics(
                frame[return_column].to_numpy(),
                benchmark_returns=frame["benchmark_return"].to_numpy(),
            )
            rows.append(
                {
                    "requested_start_year": start_year,
                    "strategy": strategy,
                    "effective_start_month": effective_start,
                    "end_month": end_month,
                    "months": frame.height,
                    "coverage": (
                        "full_from_january"
                        if effective_start == requested_start
                        else f"partial_from_{effective_start:%Y-%m}"
                    ),
                    **metrics,
                }
            )
    return pl.DataFrame(rows).sort(["requested_start_year", "strategy"])
