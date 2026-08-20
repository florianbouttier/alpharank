"""Calculate SEC-only data quality tables without rendering HTML."""

from __future__ import annotations

import polars as pl

INCOME_METRICS: tuple[str, ...] = ("revenue", "net_income")

SHARE_METRICS: tuple[str, ...] = ("outstanding_shares",)

CORE_METRICS: tuple[str, ...] = INCOME_METRICS + SHARE_METRICS

AUDIT_METRICS: tuple[str, ...] = CORE_METRICS + ("epsActual",)

METRIC_LABELS: dict[str, str] = {
    "revenue": "Chiffre d'affaires",
    "net_income": "Resultat net",
    "outstanding_shares": "Actions en circulation",
    "epsActual": "EPS publie",
}

PERIOD_ORDER: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _build_coverage_summary(
    *, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame
) -> pl.DataFrame:
    total_tickers = general.height
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        subset = financials.filter(pl.col("metric") == metric)
        tickers_with_data = subset.get_column("ticker").n_unique() if not subset.is_empty() else 0
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "tickers_with_data": tickers_with_data,
                "total_tickers": total_tickers,
                "coverage_pct": _pct(tickers_with_data, total_tickers),
                "first_date": subset.get_column("date").min() if not subset.is_empty() else None,
                "last_date": subset.get_column("date").max() if not subset.is_empty() else None,
            }
        )
    eps = earnings.filter(pl.col("epsActual").is_not_null())
    eps_tickers = eps.get_column("ticker").n_unique() if not eps.is_empty() else 0
    rows.append(
        {
            "metric": "epsActual",
            "metric_label": METRIC_LABELS["epsActual"],
            "tickers_with_data": eps_tickers,
            "total_tickers": total_tickers,
            "coverage_pct": _pct(eps_tickers, total_tickers),
            "first_date": eps.get_column("period_end").min() if not eps.is_empty() else None,
            "last_date": eps.get_column("period_end").max() if not eps.is_empty() else None,
        }
    )
    return pl.DataFrame(rows)


def _build_missing_ticker_table(
    *, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame
) -> pl.DataFrame:
    ticker_frame = general.select(
        [
            (pl.col("Code") + pl.lit(".US")).alias("ticker"),
            pl.col("Code").alias("ticker_code"),
            pl.col("Sector").alias("sector"),
            pl.col("Industry").alias("industry"),
        ]
    )
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        present = financials.filter(pl.col("metric") == metric).select("ticker").unique()
        rows.extend(
            ticker_frame.join(present, on="ticker", how="anti")
            .with_columns(
                pl.lit(metric).alias("metric"), pl.lit(METRIC_LABELS[metric]).alias("metric_label")
            )
            .to_dicts()
        )
    eps_present = earnings.filter(pl.col("epsActual").is_not_null()).select("ticker").unique()
    rows.extend(
        ticker_frame.join(eps_present, on="ticker", how="anti")
        .with_columns(
            pl.lit("epsActual").alias("metric"),
            pl.lit(METRIC_LABELS["epsActual"]).alias("metric_label"),
        )
        .to_dicts()
    )
    if not rows:
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "metric": pl.String,
                "metric_label": pl.String,
            }
        )
    return pl.DataFrame(rows).sort(["metric", "ticker"])


def _build_zero_coverage_summary(*, missing: pl.DataFrame) -> pl.DataFrame:
    if missing.is_empty():
        return pl.DataFrame(
            schema={
                "metric": pl.String,
                "metric_label": pl.String,
                "zero_coverage_tickers": pl.Int64,
            }
        )
    return (
        missing.group_by(["metric", "metric_label"])
        .agg(pl.len().alias("zero_coverage_tickers"))
        .sort("metric")
    )


def _build_quarterly_presence(
    *, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame
) -> pl.DataFrame:
    ticker_info = general.select(
        [
            (pl.col("Code") + pl.lit(".US")).alias("ticker"),
            pl.col("Code").alias("ticker_code"),
            pl.col("Sector").alias("sector"),
            pl.col("Industry").alias("industry"),
        ]
    ).unique(subset=["ticker"])
    financial_fiscal_year_col = (
        "fiscal_year" if "fiscal_year" in financials.columns else "selected_fiscal_year"
    )
    financial_fiscal_period_col = (
        "fiscal_period" if "fiscal_period" in financials.columns else "selected_fiscal_period"
    )
    earnings_fiscal_year_col = "fiscal_year" if "fiscal_year" in earnings.columns else None
    earnings_fiscal_period_col = "fiscal_period" if "fiscal_period" in earnings.columns else None

    financial_base = financials.filter(pl.col("metric").is_in(list(CORE_METRICS)))
    if "date" in financial_base.columns:
        financial_presence = (
            financial_base.with_columns(
                pl.col("date").str.strptime(pl.Date, strict=False).alias("_quarter_dt")
            )
            .filter(pl.col("_quarter_dt").is_not_null())
            .with_columns(
                [
                    _coalesce_fiscal_year_expr(
                        fiscal_year=pl.col(financial_fiscal_year_col).cast(pl.Int64, strict=False),
                        quarter_dt=pl.col("_quarter_dt"),
                    ).alias("fiscal_year"),
                    _coalesce_fiscal_period_expr(
                        fiscal_period=pl.col(financial_fiscal_period_col).cast(
                            pl.Utf8, strict=False
                        ),
                        quarter_dt=pl.col("_quarter_dt"),
                    ).alias("fiscal_period"),
                ]
            )
            .select(["ticker", "metric", "fiscal_year", "fiscal_period"])
            .unique()
        )
    else:
        financial_presence = (
            financial_base.select(
                [
                    "ticker",
                    "metric",
                    pl.col(financial_fiscal_year_col)
                    .cast(pl.Int64, strict=False)
                    .alias("fiscal_year"),
                    pl.col(financial_fiscal_period_col)
                    .cast(pl.Utf8, strict=False)
                    .alias("fiscal_period"),
                ]
            )
            .filter(pl.col("fiscal_year").is_not_null() & pl.col("fiscal_period").is_not_null())
            .unique()
        )

    earnings_base = earnings.filter(pl.col("epsActual").is_not_null()).with_columns(
        pl.lit("epsActual").alias("metric")
    )
    if "period_end" in earnings_base.columns:
        earnings_presence = (
            earnings_base.with_columns(
                pl.col("period_end").str.strptime(pl.Date, strict=False).alias("_quarter_dt")
            )
            .filter(pl.col("_quarter_dt").is_not_null())
            .with_columns(
                [
                    _coalesce_fiscal_year_expr(
                        fiscal_year=(
                            pl.col(earnings_fiscal_year_col).cast(pl.Int64, strict=False)
                            if earnings_fiscal_year_col is not None
                            else pl.lit(None).cast(pl.Int64)
                        ),
                        quarter_dt=pl.col("_quarter_dt"),
                    ).alias("fiscal_year"),
                    _coalesce_fiscal_period_expr(
                        fiscal_period=(
                            pl.col(earnings_fiscal_period_col).cast(pl.Utf8, strict=False)
                            if earnings_fiscal_period_col is not None
                            else pl.lit(None).cast(pl.Utf8)
                        ),
                        quarter_dt=pl.col("_quarter_dt"),
                    ).alias("fiscal_period"),
                ]
            )
            .select(["ticker", "metric", "fiscal_year", "fiscal_period"])
            .unique()
        )
    else:
        earnings_presence = (
            earnings_base.select(
                [
                    "ticker",
                    "metric",
                    (
                        pl.col(earnings_fiscal_year_col).cast(pl.Int64, strict=False)
                        if earnings_fiscal_year_col is not None
                        else pl.lit(None).cast(pl.Int64)
                    ).alias("fiscal_year"),
                    (
                        pl.col(earnings_fiscal_period_col).cast(pl.Utf8, strict=False)
                        if earnings_fiscal_period_col is not None
                        else pl.lit(None).cast(pl.Utf8)
                    ).alias("fiscal_period"),
                ]
            )
            .filter(pl.col("fiscal_year").is_not_null() & pl.col("fiscal_period").is_not_null())
            .unique()
        )
    if financial_presence.is_empty() and earnings_presence.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "fiscal_year": pl.Int64,
                "fiscal_period": pl.String,
                "quarter_label": pl.String,
                "quarter_sort": pl.Int64,
                "metric": pl.String,
                "metric_label": pl.String,
                "present": pl.Boolean,
            }
        )

    observed = pl.concat([financial_presence, earnings_presence], how="diagonal_relaxed").unique()
    expected_parts: list[pl.DataFrame] = []
    for metric in INCOME_METRICS + SHARE_METRICS:
        metric_observed = financial_presence.filter(pl.col("metric") == metric).select(
            ["ticker", "fiscal_year", "fiscal_period", "metric"]
        )
        expected_parts.append(_build_continuous_quarter_grid(observed=metric_observed))
    expected_parts.append(
        _build_continuous_quarter_grid(
            observed=earnings_presence.filter(pl.col("metric") == "epsActual").select(
                ["ticker", "fiscal_year", "fiscal_period", "metric"]
            )
        )
    )
    expected = pl.concat(expected_parts, how="diagonal_relaxed").unique()
    return (
        expected.join(
            observed.with_columns(pl.lit(True).alias("present")),
            on=["ticker", "fiscal_year", "fiscal_period", "metric"],
            how="left",
        )
        .with_columns(
            [
                pl.col("present").fill_null(False),
                pl.col("metric")
                .replace_strict(METRIC_LABELS, default=pl.col("metric"))
                .alias("metric_label"),
                (pl.col("fiscal_year").cast(pl.Utf8) + pl.lit(" ") + pl.col("fiscal_period")).alias(
                    "quarter_label"
                ),
                _period_order_expr(pl.col("fiscal_period")).alias("quarter_sort"),
                (
                    pl.col("fiscal_year") * pl.lit(10) + _period_order_expr(pl.col("fiscal_period"))
                ).alias("quarter_index"),
            ]
        )
        .join(ticker_info, on="ticker", how="left")
        .select(
            [
                "ticker",
                "ticker_code",
                "sector",
                "industry",
                "fiscal_year",
                "fiscal_period",
                "quarter_label",
                "quarter_sort",
                "quarter_index",
                "metric",
                "metric_label",
                "present",
            ]
        )
        .sort(["ticker", "fiscal_year", "quarter_sort", "metric"])
    )


def _build_continuous_quarter_grid(*, observed: pl.DataFrame) -> pl.DataFrame:
    if observed.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "fiscal_year": pl.Int64,
                "fiscal_period": pl.String,
                "metric": pl.String,
            }
        )

    rows: list[dict[str, object]] = []
    quarter_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    indexed = (
        observed.with_columns(
            (
                pl.col("fiscal_year") * pl.lit(10) + _period_order_expr(pl.col("fiscal_period"))
            ).alias("quarter_index")
        )
        .select(["ticker", "metric", "quarter_index"])
        .unique()
        .sort(["ticker", "metric", "quarter_index"])
    )
    for group_key, group in indexed.group_by(["ticker", "metric"], maintain_order=True):
        if isinstance(group_key, tuple):
            ticker, metric = group_key
        else:
            ticker, metric = group_key, None
        quarter_indexes = [
            int(value) for value in group["quarter_index"].to_list() if value is not None
        ]
        if not quarter_indexes or metric is None:
            continue
        start = min(quarter_indexes)
        end = max(quarter_indexes)
        for quarter_index in range(start, end + 1):
            quarter_number = quarter_index % 10
            if quarter_number not in quarter_map:
                continue
            fiscal_year = quarter_index // 10
            fiscal_period = quarter_map[quarter_number]
            rows.append(
                {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "metric": metric,
                }
            )
    return pl.DataFrame(rows)


def _coalesce_fiscal_year_expr(*, fiscal_year: pl.Expr, quarter_dt: pl.Expr) -> pl.Expr:
    return pl.coalesce([fiscal_year, quarter_dt.dt.year()])


def _coalesce_fiscal_period_expr(*, fiscal_period: pl.Expr, quarter_dt: pl.Expr) -> pl.Expr:
    derived = _calendar_period_expr(quarter_dt)
    return (
        pl.when(fiscal_period.cast(pl.Utf8, strict=False).is_in(list(PERIOD_ORDER)))
        .then(fiscal_period.cast(pl.Utf8, strict=False))
        .otherwise(derived)
    )


def _build_ticker_metric_holes(*, quarterly_presence: pl.DataFrame) -> pl.DataFrame:
    if quarterly_presence.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "metric": pl.String,
                "metric_label": pl.String,
                "expected_quarters": pl.Int64,
                "present_quarters": pl.Int64,
                "hole_count": pl.Int64,
                "hole_pct": pl.Float64,
                "first_quarter": pl.String,
                "last_quarter": pl.String,
                "sample_missing_dates": pl.String,
            }
        )

    return (
        quarterly_presence.group_by(
            ["ticker", "ticker_code", "sector", "industry", "metric", "metric_label"]
        )
        .agg(
            [
                pl.len().alias("expected_quarters"),
                pl.col("present").cast(pl.Int64).sum().alias("present_quarters"),
                pl.col("quarter_label").sort_by("quarter_index").first().alias("first_quarter"),
                pl.col("quarter_label").sort_by("quarter_index").last().alias("last_quarter"),
                pl.col("quarter_label")
                .filter(~pl.col("present"))
                .sort_by(pl.col("quarter_index").filter(~pl.col("present")))
                .head(8)
                .alias("missing_dates"),
            ]
        )
        .with_columns(
            [
                (pl.col("expected_quarters") - pl.col("present_quarters")).alias("hole_count"),
                (
                    (pl.col("expected_quarters") - pl.col("present_quarters"))
                    * 100.0
                    / pl.col("expected_quarters").clip(lower_bound=1)
                ).alias("hole_pct"),
                pl.col("missing_dates").list.join(", ").alias("sample_missing_dates"),
            ]
        )
        .drop("missing_dates")
        .sort(["hole_count", "ticker", "metric"], descending=[True, False, False])
    )


def _build_kpi_hole_summary(*, quarterly_presence: pl.DataFrame) -> pl.DataFrame:
    if quarterly_presence.is_empty():
        return pl.DataFrame(
            schema={
                "metric": pl.String,
                "metric_label": pl.String,
                "expected_quarters": pl.Int64,
                "present_quarters": pl.Int64,
                "tickers_with_holes": pl.Int64,
                "hole_count": pl.Int64,
                "hole_pct": pl.Float64,
            }
        )
    return (
        quarterly_presence.group_by(["metric", "metric_label"])
        .agg(
            [
                pl.len().alias("expected_quarters"),
                pl.col("present").cast(pl.Int64).sum().alias("present_quarters"),
                pl.col("ticker").filter(~pl.col("present")).n_unique().alias("tickers_with_holes"),
            ]
        )
        .with_columns(
            [
                (pl.col("expected_quarters") - pl.col("present_quarters")).alias("hole_count"),
                (
                    (pl.col("expected_quarters") - pl.col("present_quarters"))
                    * 100.0
                    / pl.col("expected_quarters").clip(lower_bound=1)
                ).alias("hole_pct"),
            ]
        )
        .sort("hole_count", descending=True)
    )


def _build_sector_gap_summary(*, ticker_metric_holes: pl.DataFrame) -> pl.DataFrame:
    if ticker_metric_holes.is_empty():
        return pl.DataFrame(
            schema={
                "sector": pl.String,
                "metric": pl.String,
                "metric_label": pl.String,
                "hole_count": pl.Int64,
            }
        )
    return (
        ticker_metric_holes.group_by(["sector", "metric", "metric_label"])
        .agg(pl.col("hole_count").sum())
        .sort(["hole_count", "sector"], descending=[True, False])
    )


def _build_ticker_gap_summary(*, ticker_metric_holes: pl.DataFrame) -> pl.DataFrame:
    if ticker_metric_holes.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "total_expected_quarters": pl.Int64,
                "total_present_quarters": pl.Int64,
                "total_holes": pl.Int64,
                "total_hole_pct": pl.Float64,
                "worst_metric": pl.String,
                "worst_metric_label": pl.String,
                "worst_metric_holes": pl.Int64,
                "revenue_holes": pl.Int64,
                "net_income_holes": pl.Int64,
                "outstanding_shares_holes": pl.Int64,
                "epsActual_holes": pl.Int64,
            }
        )

    pivot = ticker_metric_holes.select(
        ["ticker", "ticker_code", "sector", "industry", "metric", "hole_count"]
    ).pivot(
        index=["ticker", "ticker_code", "sector", "industry"],
        on="metric",
        values="hole_count",
        aggregate_function="first",
    )
    for metric in AUDIT_METRICS:
        if metric not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).cast(pl.Int64).alias(metric))
    pivot = pivot.with_columns(
        [pl.col(metric).fill_null(0).cast(pl.Int64).alias(metric) for metric in AUDIT_METRICS]
    )
    totals = ticker_metric_holes.group_by(["ticker", "ticker_code", "sector", "industry"]).agg(
        [
            pl.col("expected_quarters").sum().alias("total_expected_quarters"),
            pl.col("present_quarters").sum().alias("total_present_quarters"),
        ]
    )
    pivot = pivot.join(totals, on=["ticker", "ticker_code", "sector", "industry"], how="left")
    pivot = pivot.with_columns(
        [
            (
                pl.col("revenue")
                + pl.col("net_income")
                + pl.col("outstanding_shares")
                + pl.col("epsActual")
            ).alias("total_holes"),
            (
                (
                    pl.col("revenue")
                    + pl.col("net_income")
                    + pl.col("outstanding_shares")
                    + pl.col("epsActual")
                )
                * 100.0
                / pl.col("total_expected_quarters").clip(lower_bound=1)
            ).alias("total_hole_pct"),
        ]
    )

    rows: list[dict[str, object]] = []
    for row in pivot.to_dicts():
        metric_pairs = [(metric, int(row.get(metric) or 0)) for metric in AUDIT_METRICS]
        worst_metric, worst_value = max(metric_pairs, key=lambda item: item[1])
        rows.append(
            {
                "ticker": row["ticker"],
                "ticker_code": row["ticker_code"],
                "sector": row["sector"],
                "industry": row["industry"],
                "total_expected_quarters": int(row.get("total_expected_quarters") or 0),
                "total_present_quarters": int(row.get("total_present_quarters") or 0),
                "total_holes": int(row["total_holes"] or 0),
                "total_hole_pct": float(row.get("total_hole_pct") or 0.0),
                "worst_metric": worst_metric,
                "worst_metric_label": METRIC_LABELS[worst_metric],
                "worst_metric_holes": worst_value,
                "revenue_holes": int(row.get("revenue") or 0),
                "net_income_holes": int(row.get("net_income") or 0),
                "outstanding_shares_holes": int(row.get("outstanding_shares") or 0),
                "epsActual_holes": int(row.get("epsActual") or 0),
            }
        )
    return pl.DataFrame(rows).sort(["total_holes", "ticker"], descending=[True, False])


def _build_share_split_candidates(*, shares: pl.DataFrame, ratio_threshold: float) -> pl.DataFrame:
    if shares.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "date": pl.String,
                "shares": pl.Float64,
                "prev_shares": pl.Float64,
                "share_ratio": pl.Float64,
                "candidate_kind": pl.String,
            }
        )
    return (
        shares.select(
            [
                "ticker",
                pl.col("dateFormatted").cast(pl.Utf8).alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares"),
            ]
        )
        .sort(["ticker", "date"])
        .with_columns(pl.col("shares").shift(1).over("ticker").alias("prev_shares"))
        .with_columns((pl.col("shares") / pl.col("prev_shares")).alias("share_ratio"))
        .with_columns(
            pl.when(pl.col("share_ratio") >= ratio_threshold)
            .then(pl.lit("hausse forte"))
            .when(pl.col("share_ratio") <= (1.0 / ratio_threshold))
            .then(pl.lit("baisse forte"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("candidate_kind")
        )
        .filter(pl.col("candidate_kind").is_not_null())
        .sort(["ticker", "date"])
    )


def _build_share_anomaly_summary(*, share_candidates: pl.DataFrame) -> pl.DataFrame:
    if share_candidates.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "candidate_count": pl.Int64,
                "max_ratio": pl.Float64,
                "min_ratio": pl.Float64,
            }
        )
    return (
        share_candidates.group_by("ticker")
        .agg(
            [
                pl.len().alias("candidate_count"),
                pl.col("share_ratio").max().alias("max_ratio"),
                pl.col("share_ratio").min().alias("min_ratio"),
            ]
        )
        .sort(["candidate_count", "ticker"], descending=[True, False])
    )


def _build_overview_frame(
    *,
    coverage: pl.DataFrame,
    kpi_hole_summary: pl.DataFrame,
    zero_coverage: pl.DataFrame,
) -> pl.DataFrame:
    return (
        coverage.join(
            kpi_hole_summary.select(["metric", "tickers_with_holes", "hole_count", "hole_pct"]),
            on="metric",
            how="left",
        )
        .join(zero_coverage, on=["metric", "metric_label"], how="left")
        .with_columns(
            [
                pl.col("tickers_with_holes").fill_null(0).cast(pl.Int64),
                pl.col("hole_count").fill_null(0).cast(pl.Int64),
                pl.col("hole_pct").fill_null(0.0).cast(pl.Float64),
                pl.col("zero_coverage_tickers").fill_null(0).cast(pl.Int64),
            ]
        )
        .sort("metric")
    )


def _select_deep_dive_tickers(
    *,
    ticker_gap_summary: pl.DataFrame,
    share_anomaly_summary: pl.DataFrame,
    limit: int,
) -> list[str]:
    frames = [
        ticker_gap_summary.head(limit).select("ticker")
        if not ticker_gap_summary.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String}),
        share_anomaly_summary.head(limit).select("ticker")
        if not share_anomaly_summary.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String}),
    ]
    return sorted(pl.concat(frames, how="vertical").unique().get_column("ticker").to_list())[:limit]


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _normalize_period_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr == "FY")
        .then(pl.lit("Q4"))
        .when(expr.is_in(["Q1", "Q2", "Q3", "Q4"]))
        .then(expr)
        .otherwise(pl.lit(None).cast(pl.Utf8))
    )


def _calendar_period_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.dt.month().is_in([1, 2, 3]))
        .then(pl.lit("Q1"))
        .when(expr.dt.month().is_in([4, 5, 6]))
        .then(pl.lit("Q2"))
        .when(expr.dt.month().is_in([7, 8, 9]))
        .then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4"))
    )


def _period_order_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr == "Q1")
        .then(pl.lit(1))
        .when(expr == "Q2")
        .then(pl.lit(2))
        .when(expr == "Q3")
        .then(pl.lit(3))
        .when(expr == "Q4")
        .then(pl.lit(4))
        .otherwise(pl.lit(99))
    )
