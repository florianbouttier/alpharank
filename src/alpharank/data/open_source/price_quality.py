from __future__ import annotations

from datetime import date

import polars as pl


EXTREME_ADJUSTED_RETURN_THRESHOLD = 0.40


def find_extreme_adjusted_price_moves(
    prices: pl.DataFrame,
    *,
    event_since: date | str,
    tickers: tuple[str, ...] | list[str] | None = None,
    threshold: float = EXTREME_ADJUSTED_RETURN_THRESHOLD,
) -> pl.DataFrame:
    """Identify recent adjusted-close jumps that require explicit review."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    required = {"ticker", "date", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price quality columns: {sorted(missing)}")

    normalized_tickers = (
        [str(ticker).upper() for ticker in tickers]
        if tickers is not None
        else None
    )
    selected = prices.select(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
        pl.col("adjusted_close").cast(pl.Float64, strict=False),
    ).filter(pl.col("adjusted_close").is_not_null() & (pl.col("adjusted_close") > 0))
    if normalized_tickers is not None:
        selected = selected.filter(pl.col("ticker").is_in(normalized_tickers))

    return (
        selected.sort(["ticker", "date"])
        .with_columns(
            pl.col("adjusted_close").shift(1).over("ticker").alias("prior_adjusted_close")
        )
        .with_columns(
            (
                pl.col("adjusted_close") / pl.col("prior_adjusted_close") - 1.0
            ).alias("one_day_return")
        )
        .filter(
            (pl.col("date") >= pl.lit(event_since).cast(pl.Date))
            & (pl.col("one_day_return").abs() >= threshold)
        )
        .select(
            "ticker",
            "date",
            "prior_adjusted_close",
            "adjusted_close",
            "one_day_return",
        )
        .sort(["date", "ticker"])
    )


def assert_no_extreme_adjusted_price_moves(
    prices: pl.DataFrame,
    *,
    event_since: date | str,
    tickers: tuple[str, ...] | list[str] | None = None,
    threshold: float = EXTREME_ADJUSTED_RETURN_THRESHOLD,
) -> pl.DataFrame:
    findings = find_extreme_adjusted_price_moves(
        prices,
        event_since=event_since,
        tickers=tickers,
        threshold=threshold,
    )
    if not findings.is_empty():
        examples = findings.head(10).to_dicts()
        raise RuntimeError(
            "Recent adjusted-close discontinuities require review before publication: "
            f"count={findings.height}, threshold={threshold:.0%}, examples={examples}"
        )
    return findings
