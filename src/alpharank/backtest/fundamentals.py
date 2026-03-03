from __future__ import annotations

import warnings
from typing import Dict, Iterable, List

import polars as pl

from alpharank.backtest.data_loading import find_existing_column


def _safe_div(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    valid_denominator = denominator.is_not_null() & (denominator.abs() > 1e-12)
    return pl.when(valid_denominator).then(numerator / denominator).otherwise(None).alias(alias)


def _prepare_quarterly_statement(
    df: pl.DataFrame,
    value_map: Dict[str, Iterable[str]],
    filing_candidates: Iterable[str],
) -> pl.DataFrame:
    ticker_col = find_existing_column(df, ["ticker"])
    date_col = find_existing_column(df, ["date"])
    filing_col = find_existing_column(df, filing_candidates)

    if ticker_col is None or date_col is None:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "report_date": pl.Date})

    select_exprs: List[pl.Expr] = [
        pl.col(ticker_col).cast(pl.Utf8).alias("ticker"),
        pl.col(date_col).cast(pl.Date, strict=False).alias("date"),
    ]

    if filing_col is not None:
        select_exprs.append(pl.col(filing_col).cast(pl.Date, strict=False).alias("filing_date"))
    else:
        select_exprs.append(pl.col(date_col).cast(pl.Date, strict=False).alias("filing_date"))

    for target_col, source_candidates in value_map.items():
        source_col = find_existing_column(df, source_candidates)
        if source_col is None:
            select_exprs.append(pl.lit(None, dtype=pl.Float64).alias(target_col))
        else:
            select_exprs.append(pl.col(source_col).cast(pl.Float64, strict=False).alias(target_col))

    selected = (
        df.select(select_exprs)
        .with_columns(pl.col("date").dt.truncate("1q").alias("quarter_end"))
        .sort(["ticker", "quarter_end", "filing_date"])
    )

    agg_exprs: List[pl.Expr] = [pl.col("filing_date").last().alias("filing_date")]
    agg_exprs.extend(pl.col(col_name).last().alias(col_name) for col_name in value_map.keys())

    aggregated = (
        selected.group_by(["ticker", "quarter_end"]).agg(agg_exprs)
        .with_columns(pl.coalesce([pl.col("filing_date"), pl.col("quarter_end")]).alias("report_date"))
        .select(["ticker", "report_date", *value_map.keys()])
        .sort(["ticker", "report_date"])
    )

    return aggregated


def _add_ttm_features(
    quarterly_df: pl.DataFrame,
    sum_cols: Iterable[str],
    mean_cols: Iterable[str],
) -> pl.DataFrame:
    if quarterly_df.is_empty():
        return quarterly_df

    exprs: List[pl.Expr] = []
    output_cols: List[str] = []

    for col_name in sum_cols:
        if col_name in quarterly_df.columns:
            output_name = f"{col_name}_ttm"
            output_cols.append(output_name)
            exprs.append(pl.col(col_name).rolling_sum(window_size=4).over("ticker").alias(output_name))

    for col_name in mean_cols:
        if col_name in quarterly_df.columns:
            output_name = f"{col_name}_avg4q"
            output_cols.append(output_name)
            exprs.append(pl.col(col_name).rolling_mean(window_size=4).over("ticker").alias(output_name))

    with_ttm = (
        quarterly_df.sort(["ticker", "report_date"])
        .with_columns(exprs)
        .select(["ticker", "report_date", *output_cols])
    )

    return with_ttm


def _asof_join_monthly(monthly_df: pl.DataFrame, statement_df: pl.DataFrame) -> pl.DataFrame:
    if statement_df.is_empty():
        return monthly_df

    monthly_sorted = monthly_df.sort(["ticker", "date"])
    statement_sorted = statement_df.sort(["ticker", "report_date"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        try:
            joined = monthly_sorted.join_asof(
                statement_sorted,
                left_on="date",
                right_on="report_date",
                by="ticker",
                strategy="backward",
                check_sortedness=False,
            )
        except TypeError:
            joined = monthly_sorted.join_asof(
                statement_sorted,
                left_on="date",
                right_on="report_date",
                by="ticker",
                strategy="backward",
            )

    if "report_date" in joined.columns:
        joined = joined.drop("report_date")

    return joined


def build_monthly_fundamental_features(
    monthly_prices: pl.DataFrame,
    balance_sheet: pl.DataFrame,
    income_statement: pl.DataFrame,
    cash_flow: pl.DataFrame,
    earnings: pl.DataFrame,
) -> pl.DataFrame:
    income_map = {
        "total_revenue": ["totalRevenue"],
        "net_income": ["netIncome"],
        "ebitda": ["ebitda"],
        "ebit": ["ebit"],
        "gross_profit": ["grossProfit"],
    }
    balance_map = {
        "shares_outstanding": ["commonStockSharesOutstanding"],
        "equity": ["totalStockholderEquity"],
        "net_debt": ["netDebt"],
        "total_assets": ["totalAssets"],
        "cash_short_term": ["cashAndShortTermInvestments"],
    }
    cash_map = {"free_cashflow": ["freeCashFlow"]}
    earnings_map = {"eps_actual": ["epsActual"]}

    income_ttm = _add_ttm_features(
        _prepare_quarterly_statement(income_statement, income_map, filing_candidates=["filing_date"]),
        sum_cols=["total_revenue", "net_income", "ebitda", "ebit", "gross_profit"],
        mean_cols=[],
    )

    balance_ttm = _add_ttm_features(
        _prepare_quarterly_statement(balance_sheet, balance_map, filing_candidates=["filing_date"]),
        sum_cols=[],
        mean_cols=["shares_outstanding", "equity", "net_debt", "total_assets", "cash_short_term"],
    )

    cash_ttm = _add_ttm_features(
        _prepare_quarterly_statement(cash_flow, cash_map, filing_candidates=["filing_date"]),
        sum_cols=["free_cashflow"],
        mean_cols=[],
    )

    earnings_ttm = _add_ttm_features(
        _prepare_quarterly_statement(earnings, earnings_map, filing_candidates=["reportDate", "report_date"]),
        sum_cols=["eps_actual"],
        mean_cols=[],
    )

    monthly = monthly_prices.select(["ticker", "date", "year_month", "last_close"]) \
        .sort(["ticker", "date"])

    monthly = _asof_join_monthly(monthly, income_ttm)
    monthly = _asof_join_monthly(monthly, balance_ttm)
    monthly = _asof_join_monthly(monthly, cash_ttm)
    monthly = _asof_join_monthly(monthly, earnings_ttm)

    required_raw_cols = [
        "shares_outstanding_avg4q",
        "net_debt_avg4q",
        "cash_short_term_avg4q",
        "total_revenue_ttm",
        "net_income_ttm",
        "ebitda_ttm",
        "ebit_ttm",
        "gross_profit_ttm",
        "equity_avg4q",
        "total_assets_avg4q",
        "free_cashflow_ttm",
        "eps_actual_ttm",
    ]
    missing_exprs: List[pl.Expr] = []
    for col_name in required_raw_cols:
        if col_name not in monthly.columns:
            missing_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col_name))
    if missing_exprs:
        monthly = monthly.with_columns(missing_exprs)

    with_ratios = (
        monthly.with_columns(
            (pl.col("last_close") * pl.col("shares_outstanding_avg4q")).alias("market_cap"),
        )
        .with_columns(
            (
                pl.col("market_cap")
                + pl.col("net_debt_avg4q").fill_null(0.0)
                - pl.col("cash_short_term_avg4q").fill_null(0.0)
            ).alias("enterprise_value")
        )
        .with_columns(
            _safe_div(pl.col("net_income_ttm"), pl.col("total_revenue_ttm"), "net_margin_ttm"),
            _safe_div(pl.col("ebitda_ttm"), pl.col("total_revenue_ttm"), "ebitda_margin_ttm"),
            _safe_div(pl.col("gross_profit_ttm"), pl.col("total_revenue_ttm"), "gross_margin_ttm"),
            _safe_div(pl.col("net_income_ttm"), pl.col("equity_avg4q"), "roe_ttm"),
            _safe_div(pl.col("net_income_ttm"), pl.col("total_assets_avg4q"), "roa_ttm"),
            _safe_div(pl.col("net_debt_avg4q"), pl.col("equity_avg4q"), "debt_to_equity"),
            _safe_div(pl.col("free_cashflow_ttm"), pl.col("total_revenue_ttm"), "fcf_margin_ttm"),
            _safe_div(pl.col("last_close"), pl.col("eps_actual_ttm"), "pe_ttm"),
            _safe_div(pl.col("market_cap"), pl.col("total_revenue_ttm"), "price_to_sales"),
            _safe_div(pl.col("market_cap"), pl.col("equity_avg4q"), "price_to_book"),
            _safe_div(pl.col("enterprise_value"), pl.col("ebitda_ttm"), "ev_to_ebitda"),
        )
        .sort(["ticker", "year_month"])
        .with_columns(
            (pl.col("total_revenue_ttm") / pl.col("total_revenue_ttm").shift(4).over("ticker") - 1.0).alias(
                "revenue_growth_yoy"
            ),
            (pl.col("net_income_ttm") / pl.col("net_income_ttm").shift(4).over("ticker") - 1.0).alias(
                "net_income_growth_yoy"
            ),
            (pl.col("eps_actual_ttm") / pl.col("eps_actual_ttm").shift(1).over("ticker") - 1.0).alias(
                "eps_growth_qoq"
            ),
        )
    )

    candidate_columns = [
        "ticker",
        "year_month",
        "market_cap",
        "enterprise_value",
        "net_margin_ttm",
        "ebitda_margin_ttm",
        "gross_margin_ttm",
        "roe_ttm",
        "roa_ttm",
        "debt_to_equity",
        "fcf_margin_ttm",
        "pe_ttm",
        "price_to_sales",
        "price_to_book",
        "ev_to_ebitda",
        "revenue_growth_yoy",
        "net_income_growth_yoy",
        "eps_growth_qoq",
        "total_revenue_ttm",
        "net_income_ttm",
        "ebitda_ttm",
        "free_cashflow_ttm",
    ]
    selected_columns = [col for col in candidate_columns if col in with_ratios.columns]

    return with_ratios.select(selected_columns)
