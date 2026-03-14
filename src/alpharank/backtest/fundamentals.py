from __future__ import annotations

import warnings
from typing import Dict, Iterable, List

import polars as pl

from alpharank.backtest.config import FundamentalFeatureConfig
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


def _add_growth_features(
    df: pl.DataFrame,
    base_cols: Iterable[str],
    quarter_lags: Iterable[int],
) -> pl.DataFrame:
    if df.is_empty():
        return df

    exprs: List[pl.Expr] = []
    for col_name in base_cols:
        if col_name not in df.columns:
            continue
        for lag in quarter_lags:
            lagged = pl.col(col_name).shift(lag).over("ticker")
            valid_denominator = lagged.is_not_null() & (lagged.abs() > 1e-12)
            exprs.append(
                pl.when(valid_denominator)
                .then(pl.col(col_name) / lagged - 1.0)
                .otherwise(None)
                .alias(f"{col_name}_growth_{int(lag)}q")
            )

    return df.sort(["ticker", "report_date"]).with_columns(exprs)


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
    config: FundamentalFeatureConfig | None = None,
) -> pl.DataFrame:
    feature_config = config or FundamentalFeatureConfig()

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

    growth_lags = feature_config.quarterly_growth_lags

    income_ttm = _add_growth_features(
        _add_ttm_features(
            _prepare_quarterly_statement(income_statement, income_map, filing_candidates=["filing_date"]),
            sum_cols=["total_revenue", "net_income", "ebitda", "ebit", "gross_profit"],
            mean_cols=[],
        ),
        base_cols=[
            "total_revenue_ttm",
            "net_income_ttm",
            "ebitda_ttm",
            "ebit_ttm",
            "gross_profit_ttm",
        ],
        quarter_lags=growth_lags,
    )

    balance_ttm = _add_growth_features(
        _add_ttm_features(
            _prepare_quarterly_statement(balance_sheet, balance_map, filing_candidates=["filing_date"]),
            sum_cols=[],
            mean_cols=["shares_outstanding", "equity", "net_debt", "total_assets", "cash_short_term"],
        ),
        base_cols=["shares_outstanding_avg4q"],
        quarter_lags=growth_lags,
    )

    cash_ttm = _add_growth_features(
        _add_ttm_features(
            _prepare_quarterly_statement(cash_flow, cash_map, filing_candidates=["filing_date"]),
            sum_cols=["free_cashflow"],
            mean_cols=[],
        ),
        base_cols=["free_cashflow_ttm"],
        quarter_lags=growth_lags,
    )

    earnings_ttm = _add_growth_features(
        _add_ttm_features(
            _prepare_quarterly_statement(earnings, earnings_map, filing_candidates=["reportDate", "report_date"]),
            sum_cols=["eps_actual"],
            mean_cols=[],
        ),
        base_cols=["eps_actual_ttm"],
        quarter_lags=growth_lags,
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
            ).alias("enterprise_value")
        )
        .with_columns(
            _safe_div(pl.col("gross_profit_ttm"), pl.col("total_revenue_ttm"), "gross_margin_ttm"),
            _safe_div(pl.col("ebit_ttm"), pl.col("total_revenue_ttm"), "ebit_margin_ttm"),
            _safe_div(pl.col("ebitda_ttm"), pl.col("total_revenue_ttm"), "ebitda_margin_ttm"),
            _safe_div(pl.col("net_income_ttm"), pl.col("total_revenue_ttm"), "net_margin_ttm"),
            _safe_div(pl.col("free_cashflow_ttm"), pl.col("total_revenue_ttm"), "fcf_margin_ttm"),
            _safe_div(pl.col("net_income_ttm"), pl.col("equity_avg4q"), "roe_ttm"),
            _safe_div(pl.col("net_income_ttm"), pl.col("total_assets_avg4q"), "roa_ttm"),
            _safe_div(pl.col("gross_profit_ttm"), pl.col("total_assets_avg4q"), "gross_profit_to_assets"),
            _safe_div(pl.col("ebit_ttm"), pl.col("total_assets_avg4q"), "ebit_to_assets"),
            _safe_div(pl.col("free_cashflow_ttm"), pl.col("total_assets_avg4q"), "fcf_to_assets"),
            _safe_div(pl.col("total_revenue_ttm"), pl.col("total_assets_avg4q"), "asset_turnover_ttm"),
            _safe_div(
                pl.col("net_income_ttm") - pl.col("free_cashflow_ttm"),
                pl.col("total_assets_avg4q"),
                "accrual_ratio",
            ),
            _safe_div(pl.col("free_cashflow_ttm"), pl.col("net_income_ttm"), "fcf_to_net_income"),
            _safe_div(pl.col("net_debt_avg4q"), pl.col("equity_avg4q"), "debt_to_equity"),
            _safe_div(pl.col("net_debt_avg4q"), pl.col("total_assets_avg4q"), "net_debt_to_assets"),
            _safe_div(pl.col("net_debt_avg4q"), pl.col("ebitda_ttm"), "net_debt_to_ebitda"),
            _safe_div(pl.col("equity_avg4q"), pl.col("total_assets_avg4q"), "equity_to_assets"),
            _safe_div(pl.col("cash_short_term_avg4q"), pl.col("total_assets_avg4q"), "cash_to_assets"),
            _safe_div(pl.col("eps_actual_ttm"), pl.col("last_close"), "earnings_yield"),
            _safe_div(pl.col("total_revenue_ttm"), pl.col("market_cap"), "sales_yield"),
            _safe_div(pl.col("equity_avg4q"), pl.col("market_cap"), "book_to_price"),
            _safe_div(pl.col("free_cashflow_ttm"), pl.col("market_cap"), "fcf_yield"),
            _safe_div(pl.col("ebitda_ttm"), pl.col("enterprise_value"), "ebitda_to_ev"),
        )
        .with_columns(
            [
                pl.col(f"shares_outstanding_avg4q_growth_{lag}q").alias(f"share_dilution_{lag}q")
                for lag in growth_lags
                if f"shares_outstanding_avg4q_growth_{lag}q" in monthly.columns
            ]
        )
        .sort(["ticker", "year_month"])
    )

    candidate_columns = [
        "ticker",
        "year_month",
        "gross_margin_ttm",
        "ebit_margin_ttm",
        "ebitda_margin_ttm",
        "net_margin_ttm",
        "fcf_margin_ttm",
        "roe_ttm",
        "roa_ttm",
        "gross_profit_to_assets",
        "ebit_to_assets",
        "fcf_to_assets",
        "asset_turnover_ttm",
        "accrual_ratio",
        "fcf_to_net_income",
        "debt_to_equity",
        "net_debt_to_assets",
        "net_debt_to_ebitda",
        "equity_to_assets",
        "cash_to_assets",
        "earnings_yield",
        "sales_yield",
        "book_to_price",
        "fcf_yield",
        "ebitda_to_ev",
    ]
    growth_bases = [
        "total_revenue_ttm",
        "net_income_ttm",
        "ebitda_ttm",
        "ebit_ttm",
        "gross_profit_ttm",
        "free_cashflow_ttm",
        "eps_actual_ttm",
    ]
    for base_name in growth_bases:
        candidate_columns.extend(f"{base_name}_growth_{lag}q" for lag in growth_lags)
    candidate_columns.extend(f"share_dilution_{lag}q" for lag in growth_lags)
    selected_columns = [col for col in candidate_columns if col in with_ratios.columns]

    return with_ratios.select(selected_columns)
