from __future__ import annotations

from typing import Any

import polars as pl


_SHARE_METRICS = ("outstanding_shares", "weighted_average_diluted_shares")


def quarantine_implausible_share_candidates(
    financials: pl.DataFrame,
    *,
    minimum_shares: float = 100_000.0,
    maximum_shares: float = 100_000_000_000.0,
    maximum_scale_from_series_median: float = 100.0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Exclude unit-corrupted share facts from selection while preserving raw input."""
    required = {"ticker", "metric", "value"}
    if financials.is_empty() or not required.issubset(financials.columns):
        return financials, _share_quarantine_report(
            financials.head(0),
            input_rows=financials.height,
            output_rows=financials.height,
            minimum_shares=minimum_shares,
            maximum_shares=maximum_shares,
            maximum_scale_from_series_median=maximum_scale_from_series_median,
        )

    indexed = financials.with_row_index("_quality_row_id")
    share_candidates = indexed.filter(pl.col("metric").is_in(_SHARE_METRICS))
    non_share_candidates = indexed.filter(pl.col("metric").is_in(_SHARE_METRICS).not_())
    plausible = (
        share_candidates.filter(
            pl.col("value").cast(pl.Float64, strict=False).is_finite()
            & pl.col("value").cast(pl.Float64, strict=False).is_between(
                minimum_shares,
                maximum_shares,
            )
        )
        .with_columns(
            pl.col("value")
            .cast(pl.Float64, strict=False)
            .median()
            .over(["ticker", "metric"])
            .alias("_series_median")
        )
        .with_columns(
            pl.max_horizontal("value", "_series_median")
            .truediv(pl.min_horizontal("value", "_series_median"))
            .alias("_scale_from_series_median")
        )
    )
    accepted_shares = plausible.filter(
        pl.col("_scale_from_series_median") < maximum_scale_from_series_median
    ).drop("_series_median", "_scale_from_series_median")
    rejected = share_candidates.join(
        accepted_shares.select("_quality_row_id"),
        on="_quality_row_id",
        how="anti",
    ).drop("_quality_row_id")
    cleaned = pl.concat(
        [non_share_candidates, accepted_shares],
        how="vertical_relaxed",
    ).drop("_quality_row_id").sort(
        [column for column in ("ticker", "statement", "metric", "date") if column in financials.columns]
    )
    return cleaned, _share_quarantine_report(
        rejected,
        input_rows=financials.height,
        output_rows=cleaned.height,
        minimum_shares=minimum_shares,
        maximum_shares=maximum_shares,
        maximum_scale_from_series_median=maximum_scale_from_series_median,
    )


def _share_quarantine_report(
    rejected: pl.DataFrame,
    *,
    input_rows: int,
    output_rows: int,
    minimum_shares: float,
    maximum_shares: float,
    maximum_scale_from_series_median: float,
) -> dict[str, Any]:
    example_columns = _available_columns(
        rejected,
        "ticker",
        "metric",
        "date",
        "filing_date",
        "value",
        "source",
        "source_label",
    )
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "quarantined_rows": input_rows - output_rows,
        "minimum_shares": minimum_shares,
        "maximum_shares": maximum_shares,
        "maximum_scale_from_series_median": maximum_scale_from_series_median,
        "quarantined_examples": (
            rejected.select(example_columns).head(20).to_dicts()
            if example_columns
            else []
        ),
    }


def audit_fundamental_quality(
    financials: pl.DataFrame,
    *,
    max_share_scale_ratio: float = 1_000.0,
) -> dict[str, Any]:
    """Detect publication-blocking numeric and share-scale anomalies."""
    non_finite = _non_finite_rows(financials)
    share_discontinuities = _share_scale_discontinuities(
        financials,
        max_ratio=max_share_scale_ratio,
    )
    return {
        "guard_version": 1,
        "max_share_scale_ratio": max_share_scale_ratio,
        "non_finite_value_count": non_finite.height,
        "non_finite_examples": non_finite.head(20).to_dicts(),
        "share_scale_discontinuity_count": share_discontinuities.height,
        "share_scale_discontinuity_examples": share_discontinuities.head(20).to_dicts(),
        "quality_failures_detected": bool(non_finite.height or share_discontinuities.height),
    }


def validate_fundamental_quality(report: dict[str, Any]) -> None:
    if report["quality_failures_detected"]:
        raise RuntimeError(
            "Fundamental quality gate failed before publication: "
            f"non_finite={report['non_finite_value_count']}, "
            f"share_scale_discontinuities={report['share_scale_discontinuity_count']}"
        )


def _non_finite_rows(financials: pl.DataFrame) -> pl.DataFrame:
    if financials.is_empty() or "value" not in financials.columns:
        return pl.DataFrame()
    return financials.filter(
        pl.col("value").is_not_null()
        & pl.col("value").cast(pl.Float64, strict=False).is_finite().not_()
    ).select(_available_columns(financials, "ticker", "statement", "metric", "date", "value", "selected_source"))


def _share_scale_discontinuities(
    financials: pl.DataFrame,
    *,
    max_ratio: float,
) -> pl.DataFrame:
    required = {"ticker", "metric", "date", "value"}
    if financials.is_empty() or not required.issubset(financials.columns):
        return pl.DataFrame()
    shares = (
        financials.filter(
            pl.col("metric").is_in(
                ["outstanding_shares", "weighted_average_diluted_shares"]
            )
            & (pl.col("value").cast(pl.Float64, strict=False) > 0.0)
        )
        .sort(["ticker", "metric", "date"])
        .with_columns(
            pl.col("value")
            .cast(pl.Float64, strict=False)
            .shift(1)
            .over(["ticker", "metric"])
            .alias("previous_value")
        )
        .with_columns(
            pl.max_horizontal("value", "previous_value")
            .truediv(pl.min_horizontal("value", "previous_value"))
            .alias("scale_ratio")
        )
        .filter(
            pl.col("previous_value").is_not_null()
            & pl.col("scale_ratio").is_finite()
            & (pl.col("scale_ratio") >= max_ratio)
        )
    )
    columns = _available_columns(
        shares,
        "ticker",
        "metric",
        "date",
        "previous_value",
        "value",
        "scale_ratio",
        "selected_source",
        "selected_source_label",
    )
    return shares.select(columns).sort("scale_ratio", descending=True)


def _available_columns(frame: pl.DataFrame, *columns: str) -> list[str]:
    return [column for column in columns if column in frame.columns]
