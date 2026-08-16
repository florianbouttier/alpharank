from __future__ import annotations

from datetime import date

import polars as pl


EXTREME_ADJUSTED_RETURN_THRESHOLD = 0.40
SPLIT_RATIO_RELATIVE_TOLERANCE = 0.15


def build_split_detection_prices(
    *,
    existing_prices: pl.DataFrame,
    fresh_prices: pl.DataFrame,
    full_history_refresh: bool,
) -> pl.DataFrame:
    """Return one coherent source vintage for corporate-action detection."""

    required = ["ticker", "date", "adjusted_close"]
    if full_history_refresh:
        return fresh_prices.select(
            [column for column in PRICE_QUALITY_COLUMNS if column in fresh_prices.columns]
        ).sort(["ticker", "date"])
    existing = existing_prices.with_columns(pl.lit(0).alias("_fresh_priority"))
    fresh = fresh_prices.with_columns(pl.lit(1).alias("_fresh_priority"))
    return (
        pl.concat([existing, fresh], how="diagonal_relaxed")
        .sort(["ticker", "date", "_fresh_priority"])
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
        .drop("_fresh_priority")
        .sort(["ticker", "date"])
    )


PRICE_QUALITY_COLUMNS = (
    "ticker",
    "date",
    "adjusted_close",
    "close",
    "open",
    "high",
    "low",
    "volume",
)


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


def repair_confirmed_split_discontinuities(
    prices_delta: pl.DataFrame,
    *,
    findings: pl.DataFrame,
    splits: pl.DataFrame,
    relative_tolerance: float = SPLIT_RATIO_RELATIVE_TOLERANCE,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Back-adjust a fresh source delta only when its split event explains the jump."""

    repaired = prices_delta
    repairs: list[dict[str, object]] = []
    if repaired.is_empty() or findings.is_empty() or splits.is_empty():
        return repaired, repairs

    split_rows = splits.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
        pl.col("split_ratio").cast(pl.Float64, strict=False),
    )
    for finding in findings.iter_rows(named=True):
        ticker = str(finding["ticker"]).upper()
        event_date = finding["date"]
        observed_multiplier = float(finding["adjusted_close"]) / float(finding["prior_adjusted_close"])
        event = split_rows.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") == pl.lit(event_date))
        )
        if event.height != 1:
            continue
        ratio = float(event.get_column("split_ratio").item())
        expected_multiplier = 1.0 / ratio
        relative_error = abs(observed_multiplier / expected_multiplier - 1.0)
        if ratio <= 0.0 or relative_error > relative_tolerance:
            continue

        before_event = (pl.col("ticker").cast(pl.String).str.to_uppercase() == ticker) & (
            pl.col("date").cast(pl.Date) < pl.lit(event_date)
        )
        price_columns = [
            column
            for column in ("open", "high", "low", "close", "adjusted_close")
            if column in repaired.columns
        ]
        expressions = [
            pl.when(before_event)
            .then(pl.col(column) / ratio)
            .otherwise(pl.col(column))
            .alias(column)
            for column in price_columns
        ]
        if "volume" in repaired.columns:
            expressions.append(
                pl.when(before_event)
                .then(pl.col("volume") * ratio)
                .otherwise(pl.col("volume"))
                .alias("volume")
            )
        repaired = repaired.with_columns(expressions)
        repairs.append(
            {
                "ticker": ticker,
                "date": str(event_date),
                "split_ratio": ratio,
                "observed_multiplier": observed_multiplier,
                "expected_multiplier": expected_multiplier,
                "relative_error": relative_error,
                "source": str(event.get_column("source").item()) if "source" in event.columns else None,
            }
        )
    return repaired, repairs
