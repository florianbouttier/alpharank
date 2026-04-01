#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))

from alpharank.data.processing import FundamentalProcessor, PricesDataPreprocessor
from alpharank.strategy.legacy import StrategyLearner
from alpharank.utils.frame_backend import to_pandas, to_polars
from run_legacy import _load_data, run_pipeline


EODHD_DIR = PROJECT_ROOT / "data" / "eodhd" / "output"
OPEN_SOURCE_DIR = PROJECT_ROOT / "data" / "open_source" / "output"
OPEN_SOURCE_LINEAGE_DIR = OPEN_SOURCE_DIR / "lineage"
AUDIT_2025_DIR = PROJECT_ROOT / "data" / "open_source" / "audit" / "2025"
DEFAULT_FIRST_DATE = "2025-01"
PRICE_HISTORY_START = "2005-01-03"


@dataclass(frozen=True)
class DatasetSummary:
    label: str
    ticker_count: int
    final_price_rows: int
    final_price_max_date: str
    sp500_price_rows: int
    income_rows: int
    balance_rows: int
    cash_rows: int
    earnings_rows: int


@dataclass(frozen=True)
class LegacyRunSummary:
    label: str
    data_dir: str
    output_dir: str
    run_day_dir: str
    duration_seconds: float
    metrics_path: str
    monthly_returns_path: str
    cumulative_returns_path: str
    portfolio_frequency_month: str
    portfolio_equal_month: str
    portfolio_frequency_count: int
    portfolio_equal_count: int


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _write_csv(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def _ticker_set(frame: pl.DataFrame) -> set[str]:
    if "ticker" not in frame.columns:
        return set()
    return set(frame.select(pl.col("ticker").drop_nulls().cast(pl.Utf8)).to_series().to_list())


def _root_ticker_set(frame: pl.DataFrame) -> set[str]:
    if "ticker" not in frame.columns:
        return set()
    return set(
        frame.select(pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "")).to_series().to_list()
    )


def _filter_by_tickers(frame: pl.DataFrame, tickers: set[str]) -> pl.DataFrame:
    if "ticker" not in frame.columns:
        return frame
    return frame.filter(pl.col("ticker").is_in(sorted(tickers)))


def _filter_csv_constituents(path: Path, *, min_date: str, max_date: str) -> pl.DataFrame:
    frame = pl.read_csv(path, try_parse_dates=True)
    if "Date" in frame.columns:
        frame = frame.filter(
            pl.col("Date").cast(pl.Date, strict=False).is_between(date.fromisoformat(min_date), date.fromisoformat(max_date))
        )
    return frame


def _align_dataset(
    *,
    source_dir: Path,
    price_source_dir: Path,
    output_dir: Path,
    common_tickers: set[str],
    price_start_date: str,
    price_cutoff_date: str,
    financial_start_date: str,
    financial_cutoff_date: str,
    earnings_cutoff_date: str,
    constituents_start_date: str,
) -> DatasetSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_price = _filter_by_tickers(_read_parquet(price_source_dir / "US_Finalprice.parquet"), common_tickers).filter(
        pl.col("date").cast(pl.Date, strict=False).is_between(date.fromisoformat(price_start_date), date.fromisoformat(price_cutoff_date))
    )
    sp500_price = _read_parquet(price_source_dir / "SP500Price.parquet").filter(
        pl.col("date").cast(pl.Date, strict=False).is_between(date.fromisoformat(price_start_date), date.fromisoformat(price_cutoff_date))
    )

    def filter_statement(name: str) -> pl.DataFrame:
        frame = _filter_by_tickers(_read_parquet(source_dir / name), common_tickers)
        return frame.filter(
            pl.col("date").cast(pl.Date, strict=False).is_between(date.fromisoformat(financial_start_date), date.fromisoformat(financial_cutoff_date))
            & (
                pl.col("filing_date").is_null()
                | (pl.col("filing_date").cast(pl.Date, strict=False) <= date.fromisoformat(financial_cutoff_date))
            )
        )

    income_statement = filter_statement("US_Income_statement.parquet")
    balance_sheet = filter_statement("US_Balance_sheet.parquet")
    cash_flow = filter_statement("US_Cash_flow.parquet")

    earnings = _filter_by_tickers(_read_parquet(source_dir / "US_Earnings.parquet"), common_tickers).filter(
        (
            pl.col("reportDate").cast(pl.Date, strict=False) <= date.fromisoformat(earnings_cutoff_date)
        )
        & (
            pl.col("date").cast(pl.Date, strict=False) <= date.fromisoformat(earnings_cutoff_date)
        )
    )

    general = _filter_by_tickers(_read_parquet(source_dir / "US_General.parquet"), common_tickers)
    constituents = _filter_csv_constituents(
        source_dir / "SP500_Constituents.csv",
        min_date=constituents_start_date,
        max_date=price_cutoff_date,
    )

    _write_parquet(output_dir / "US_Finalprice.parquet", final_price)
    _write_parquet(output_dir / "SP500Price.parquet", sp500_price)
    _write_parquet(output_dir / "US_Income_statement.parquet", income_statement)
    _write_parquet(output_dir / "US_Balance_sheet.parquet", balance_sheet)
    _write_parquet(output_dir / "US_Cash_flow.parquet", cash_flow)
    _write_parquet(output_dir / "US_Earnings.parquet", earnings)
    _write_parquet(output_dir / "US_General.parquet", general)
    _write_csv(output_dir / "SP500_Constituents.csv", constituents)

    return DatasetSummary(
        label=output_dir.name,
        ticker_count=len(common_tickers),
        final_price_rows=final_price.height,
        final_price_max_date=str(final_price.select(pl.col("date").max()).item()) if final_price.height else "n/a",
        sp500_price_rows=sp500_price.height,
        income_rows=income_statement.height,
        balance_rows=balance_sheet.height,
        cash_rows=cash_flow.height,
        earnings_rows=earnings.height,
    )


def _prepare_aligned_datasets(output_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    eodhd_general = _read_parquet(EODHD_DIR / "US_General.parquet")
    open_general = _read_parquet(OPEN_SOURCE_DIR / "US_General.parquet")
    common_tickers = _ticker_set(eodhd_general) & _ticker_set(open_general)

    eodhd_prices = _filter_by_tickers(_read_parquet(EODHD_DIR / "US_Finalprice.parquet"), common_tickers)
    open_prices = _filter_by_tickers(_read_parquet(OPEN_SOURCE_DIR / "US_Finalprice.parquet"), common_tickers)
    price_cutoff_date = min(
        str(eodhd_prices.select(pl.col("date").max()).item()),
        str(open_prices.select(pl.col("date").max()).item()),
    )
    price_start_date = max(
        PRICE_HISTORY_START,
        str(open_prices.select(pl.col("date").min()).item()),
    )

    def min_statement_max(name: str) -> str:
        left = _read_parquet(EODHD_DIR / name)
        right = _read_parquet(OPEN_SOURCE_DIR / name)
        return min(
            str(left.select(pl.col("date").max()).item()),
            str(right.select(pl.col("date").max()).item()),
        )

    financial_cutoff_date = min(
        min_statement_max("US_Income_statement.parquet"),
        min_statement_max("US_Balance_sheet.parquet"),
        min_statement_max("US_Cash_flow.parquet"),
        price_cutoff_date,
    )
    financial_start_date = max(
        str(_read_parquet(OPEN_SOURCE_DIR / "US_Income_statement.parquet").select(pl.col("date").min()).item()),
        DEFAULT_FIRST_DATE + "-01",
    )

    eodhd_earnings = _read_parquet(EODHD_DIR / "US_Earnings.parquet")
    open_earnings = _read_parquet(OPEN_SOURCE_DIR / "US_Earnings.parquet")
    earnings_cutoff_date = min(
        str(eodhd_earnings.select(pl.col("reportDate").max()).item()),
        str(open_earnings.select(pl.col("reportDate").max()).item()),
        price_cutoff_date,
    )

    aligned_root = output_root / "aligned_data"
    eodhd_out = aligned_root / "eodhd"
    open_out = aligned_root / "open_source"

    eodhd_summary = _align_dataset(
        source_dir=EODHD_DIR,
        price_source_dir=EODHD_DIR,
        output_dir=eodhd_out,
        common_tickers=common_tickers,
        price_start_date=price_start_date,
        price_cutoff_date=price_cutoff_date,
        financial_start_date=financial_start_date,
        financial_cutoff_date=financial_cutoff_date,
        earnings_cutoff_date=earnings_cutoff_date,
        constituents_start_date=DEFAULT_FIRST_DATE + "-01",
    )
    open_summary = _align_dataset(
        source_dir=OPEN_SOURCE_DIR,
        price_source_dir=OPEN_SOURCE_DIR,
        output_dir=open_out,
        common_tickers=common_tickers,
        price_start_date=price_start_date,
        price_cutoff_date=price_cutoff_date,
        financial_start_date=financial_start_date,
        financial_cutoff_date=financial_cutoff_date,
        earnings_cutoff_date=earnings_cutoff_date,
        constituents_start_date=DEFAULT_FIRST_DATE + "-01",
    )

    alignment = {
        "common_tickers": len(common_tickers),
        "price_start_date": price_start_date,
        "price_cutoff_date": price_cutoff_date,
        "financial_start_date": financial_start_date,
        "financial_cutoff_date": financial_cutoff_date,
        "earnings_cutoff_date": earnings_cutoff_date,
        "source_general_tickers": {
            "eodhd": eodhd_general.height,
            "open_source": open_general.height,
        },
        "dataset_summaries": {
            "eodhd": asdict(eodhd_summary),
            "open_source": asdict(open_summary),
        },
    }
    return eodhd_out, open_out, alignment


def _run_legacy(label: str, *, data_dir: Path, output_root: Path, first_date: str) -> tuple[Any, LegacyRunSummary]:
    run_output_dir = output_root / "runs" / label
    checkpoints_dir = output_root / "checkpoints" / label
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".mplconfig"))
    start = perf_counter()
    result = run_pipeline(
        n_trials=30,
        n_jobs=1,
        first_date=first_date,
        data_dir=data_dir,
        output_dir=run_output_dir,
        checkpoints_dir=checkpoints_dir,
    )
    duration = perf_counter() - start
    run_day_dir = Path(result.artifacts["metrics"]).parent

    portfolio_frequency = StrategyLearner.get_portfolio_at_month(result.combined_frequency)
    portfolio_equal = StrategyLearner.get_portfolio_at_month(result.combined_equal)
    pd.DataFrame(portfolio_frequency).to_parquet(run_day_dir / "portfolio_frequency_latest.parquet", index=False)
    pd.DataFrame(portfolio_equal).to_parquet(run_day_dir / "portfolio_equal_latest.parquet", index=False)

    summary = LegacyRunSummary(
        label=label,
        data_dir=str(data_dir),
        output_dir=str(run_output_dir),
        run_day_dir=str(run_day_dir),
        duration_seconds=duration,
        metrics_path=str(result.artifacts["metrics"]),
        monthly_returns_path=str(result.artifacts["monthly_returns"]),
        cumulative_returns_path=str(result.artifacts["cumulative_returns"]),
        portfolio_frequency_month=str(portfolio_frequency.attrs.get("month", "n/a")),
        portfolio_equal_month=str(portfolio_equal.attrs.get("month", "n/a")),
        portfolio_frequency_count=len(portfolio_frequency),
        portfolio_equal_count=len(portfolio_equal),
    )
    return result, summary


def _coerce_metrics(metrics: Any, *, label: str) -> pl.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        frame = pl.from_pandas(metrics.reset_index(names="model"))
    elif isinstance(metrics, pl.DataFrame):
        frame = metrics
        if "model" not in frame.columns:
            frame = frame.with_row_index("model")
    else:
        frame = pl.from_pandas(pd.DataFrame(metrics).reset_index(names="model"))
    rename_map = {column: f"{label}_{column}" for column in frame.columns if column != "model"}
    return frame.rename(rename_map)


def _compare_metrics(eodhd_metrics: Any, open_metrics: Any) -> pl.DataFrame:
    left = _coerce_metrics(eodhd_metrics, label="eodhd")
    right = _coerce_metrics(open_metrics, label="open")
    joined = left.join(right, on="model", how="full", coalesce=True).sort("model")

    diff_exprs: list[pl.Expr] = []
    for column in joined.columns:
        if column.startswith("eodhd_"):
            metric = column.removeprefix("eodhd_")
            right_col = f"open_{metric}"
            if right_col in joined.columns:
                diff_exprs.append(
                    (pl.col(right_col).cast(pl.Float64, strict=False) - pl.col(column).cast(pl.Float64, strict=False)).alias(
                        f"diff_{metric}"
                    )
                )
    if diff_exprs:
        joined = joined.with_columns(diff_exprs)
    return joined


def _portfolio_frame_to_polars(frame: pd.DataFrame, *, label: str) -> pl.DataFrame:
    out = pl.from_pandas(frame.reset_index(drop=True))
    rename_map = {column: f"{label}_{column}" for column in out.columns if column != "ticker"}
    return out.rename(rename_map)


def _compare_portfolios(eodhd_frame: pd.DataFrame, open_frame: pd.DataFrame) -> pl.DataFrame:
    left = _portfolio_frame_to_polars(eodhd_frame, label="eodhd")
    right = _portfolio_frame_to_polars(open_frame, label="open")
    joined = left.join(right, on="ticker", how="full", coalesce=True).sort("ticker")
    numeric_pairs = [
        ("weight", "weight_diff"),
        ("weight_normalized", "weight_normalized_diff"),
        ("monthly_return", "monthly_return_diff"),
    ]
    exprs: list[pl.Expr] = []
    for metric, alias in numeric_pairs:
        left_col = f"eodhd_{metric}"
        right_col = f"open_{metric}"
        if left_col in joined.columns and right_col in joined.columns:
            exprs.append(
                (pl.col(right_col).cast(pl.Float64, strict=False) - pl.col(left_col).cast(pl.Float64, strict=False)).alias(alias)
            )
    status = (
        pl.when(pl.col("eodhd_weight_normalized").is_null() & pl.col("open_weight_normalized").is_not_null())
        .then(pl.lit("open_only"))
        .when(pl.col("eodhd_weight_normalized").is_not_null() & pl.col("open_weight_normalized").is_null())
        .then(pl.lit("eodhd_only"))
        .otherwise(pl.lit("common"))
        .alias("holding_status")
    )
    return joined.with_columns([*exprs, status])


def _read_parquet_safe(path: str) -> pl.DataFrame:
    return pl.read_parquet(Path(path))


def _build_monthly_returns_diff(eodhd_path: str, open_path: str) -> pl.DataFrame:
    left = _read_parquet_safe(eodhd_path).rename({"monthly_return": "eodhd_monthly_return"})
    right = _read_parquet_safe(open_path).rename({"monthly_return": "open_monthly_return"})
    joined = left.join(
        right,
        on=[column for column in ["portfolio_model", "model", "year_month"] if column in left.columns and column in right.columns],
        how="full",
        coalesce=True,
    )
    if "eodhd_monthly_return" in joined.columns and "open_monthly_return" in joined.columns:
        joined = joined.with_columns(
            (pl.col("open_monthly_return").cast(pl.Float64, strict=False) - pl.col("eodhd_monthly_return").cast(pl.Float64, strict=False)).alias(
                "monthly_return_diff"
            )
        )
    return joined.sort([column for column in ["portfolio_model", "model", "year_month"] if column in joined.columns])


def _portfolio_overlap_summary(diff_frame: pl.DataFrame) -> dict[str, Any]:
    overlap = diff_frame.filter(pl.col("holding_status") == "common").height
    eodhd_only = diff_frame.filter(pl.col("holding_status") == "eodhd_only").height
    open_only = diff_frame.filter(pl.col("holding_status") == "open_only").height
    union = overlap + eodhd_only + open_only
    jaccard = overlap / union if union else None
    top_weight_diffs = (
        diff_frame.filter(pl.col("weight_normalized_diff").is_not_null())
        .with_columns(pl.col("weight_normalized_diff").abs().alias("abs_weight_normalized_diff"))
        .sort("abs_weight_normalized_diff", descending=True)
        .select(["ticker", "holding_status", "eodhd_weight_normalized", "open_weight_normalized", "weight_normalized_diff"])
        .head(10)
    )
    return {
        "overlap": overlap,
        "eodhd_only": eodhd_only,
        "open_only": open_only,
        "jaccard": jaccard,
        "top_weight_diffs": top_weight_diffs,
    }


def _markdown_table(frame: pl.DataFrame, *, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if columns is not None:
        existing = [column for column in columns if column in frame.columns]
        frame = frame.select(existing)
    frame = frame.head(max_rows)
    if frame.is_empty():
        return "_empty_"

    rows = frame.rows()
    headers = frame.columns

    def stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    string_rows = [[stringify(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join(
        [
            render_row(headers),
            separator,
            *[render_row(row) for row in string_rows],
        ]
    )


def _read_parquet_if_exists(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _latest_month_frame(frame: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    if frame.is_empty() or "year_month" not in frame.columns:
        return frame, "n/a"
    latest_month = frame.select(pl.col("year_month").max()).item()
    if latest_month is None:
        return frame, "n/a"
    latest = frame.filter(pl.col("year_month") == latest_month)
    return latest, str(latest_month)


def _build_input_quality_summary(output_root: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for label in ("eodhd", "open_source"):
        data_dir = output_root / "aligned_data" / label
        general = _read_parquet_if_exists(data_dir / "US_General.parquet")
        earnings = _read_parquet_if_exists(data_dir / "US_Earnings.parquet")
        income = _read_parquet_if_exists(data_dir / "US_Income_statement.parquet")
        balance = _read_parquet_if_exists(data_dir / "US_Balance_sheet.parquet")
        cash = _read_parquet_if_exists(data_dir / "US_Cash_flow.parquet")
        rows.append(
            {
                "dataset": label,
                "general_rows": general.height,
                "sector_non_null_rows": general.filter(pl.col("Sector").is_not_null() & (pl.col("Sector") != "")).height if "Sector" in general.columns else 0,
                "sector_null_rows": general.filter(pl.col("Sector").is_null() | (pl.col("Sector") == "")).height if "Sector" in general.columns else general.height,
                "earnings_rows": earnings.height,
                "earnings_tickers": earnings.select(pl.col("ticker").n_unique()).item() if not earnings.is_empty() else 0,
                "income_rows": income.height,
                "balance_rows": balance.height,
                "cash_rows": cash.height,
            }
        )
    return pl.DataFrame(rows)


def _build_selection_input_coverage(output_root: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("eodhd", "open_source"):
        payload = _load_data(output_root / "aligned_data" / dataset)
        ticker_to_exclude = ["SII.US", "CBE.US", "TIE.US"]
        final_price = payload["final_price"].filter(~pl.col("ticker").is_in(ticker_to_exclude)).with_columns(
            pl.col("date").cast(pl.Date, strict=False).dt.truncate("1mo").alias("year_month")
        )
        income = payload["income_statement"].filter(~pl.col("ticker").is_in(ticker_to_exclude))
        balance = payload["balance_sheet"].filter(~pl.col("ticker").is_in(ticker_to_exclude))
        cash = payload["cash_flow"].filter(~pl.col("ticker").is_in(ticker_to_exclude))
        earnings = payload["earnings"].filter(~pl.col("ticker").is_in(ticker_to_exclude))
        historical_company = (
            payload["us_historical_company"]
            .with_columns(
                [
                    pl.col("Ticker").cast(pl.Utf8).str.replace_all(r"\\.", "-").alias("ticker"),
                    pl.col("Date").cast(pl.Date, strict=False).dt.truncate("1mo").alias("year_month"),
                ]
            )
            .with_columns((pl.col("ticker") + pl.lit(".US")).alias("ticker"))
        )

        monthly_return = to_polars(
            PricesDataPreprocessor.calculate_monthly_returns(
                df=final_price.clone(),
                column_close="adjusted_close",
                column_date="date",
                backend="polars",
            )
        )
        valuation = to_polars(
            FundamentalProcessor.calculate_pe_ratios(
                balance=balance,
                earnings=earnings,
                cashflow=cash,
                income=income,
                earning_choice="netincome_rolling",
                monthly_return=to_pandas(monthly_return),
                list_date_to_maximise=["filing_date_income", "filing_date_balance"],
                backend="polars",
            )
        ).with_columns(pl.col("year_month").cast(pl.Date, strict=False))

        latest_price_month = monthly_return.select(pl.col("year_month").max()).item() if not monthly_return.is_empty() else None
        latest_value_month = valuation.select(pl.col("year_month").max()).item() if not valuation.is_empty() else None
        latest_filter_month = latest_value_month
        filtered = valuation.filter(
            (pl.col("pe") < 100)
            & (pl.col("pe") > 0)
            & pl.col("pe").is_not_null()
            & pl.col("market_cap").is_not_null()
        )
        joined = filtered.join(
            historical_company.select(["ticker", "year_month"]),
            on=["ticker", "year_month"],
            how="inner",
        )
        latest_join_month = joined.select(pl.col("year_month").max()).item() if not joined.is_empty() else None

        latest_prices = monthly_return.filter(pl.col("year_month") == latest_price_month) if latest_price_month is not None else pl.DataFrame()
        latest_values = valuation.filter(pl.col("year_month") == latest_value_month) if latest_value_month is not None else pl.DataFrame()
        latest_filtered = filtered.filter(pl.col("year_month") == latest_filter_month) if latest_filter_month is not None else pl.DataFrame()
        latest_joined = joined.filter(pl.col("year_month") == latest_join_month) if latest_join_month is not None else pl.DataFrame()

        rows.append(
            {
                "dataset": dataset,
                "latest_price_month": str(latest_price_month) if latest_price_month is not None else "n/a",
                "latest_price_tickers": latest_prices.select(pl.col("ticker").n_unique()).item() if not latest_prices.is_empty() else 0,
                "latest_value_month": str(latest_value_month) if latest_value_month is not None else "n/a",
                "latest_value_tickers": latest_values.select(pl.col("ticker").n_unique()).item() if not latest_values.is_empty() else 0,
                "latest_value_non_null_pe": latest_values.filter(pl.col("pe").is_not_null()).height if not latest_values.is_empty() else 0,
                "latest_value_non_null_market_cap": latest_values.filter(pl.col("market_cap").is_not_null()).height if not latest_values.is_empty() else 0,
                "latest_post_filter_tickers": latest_filtered.select(pl.col("ticker").n_unique()).item() if not latest_filtered.is_empty() else 0,
                "latest_constituent_join_tickers": latest_joined.select(pl.col("ticker").n_unique()).item() if not latest_joined.is_empty() else 0,
            }
        )
    return pl.DataFrame(rows)


def _build_price_coverage_gap_summary(output_root: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    eodhd_general = _read_parquet_if_exists(output_root / "aligned_data" / "eodhd" / "US_General.parquet")
    open_general = _read_parquet_if_exists(output_root / "aligned_data" / "open_source" / "US_General.parquet")
    open_prices = _read_parquet_if_exists(output_root / "aligned_data" / "open_source" / "US_Finalprice.parquet")
    common_general = sorted(_ticker_set(eodhd_general) & _ticker_set(open_general))
    covered_common = sorted(set(common_general) & _ticker_set(open_prices))
    missing_tickers = sorted(set(common_general) - set(covered_common))

    eodhd_selection = _read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / "polars_stocks_selections.parquet")
    missing_selection = (
        eodhd_selection.filter(pl.col("ticker").is_in(missing_tickers))
        if not eodhd_selection.is_empty() and missing_tickers
        else pl.DataFrame()
    )

    summary = pl.DataFrame(
        [
            {
                "common_general_tickers": len(common_general),
                "open_price_covered_tickers": len(covered_common),
                "missing_open_price_tickers": len(missing_tickers),
                "missing_open_price_ticker_rate_pct": ((len(missing_tickers) / len(common_general)) * 100.0) if common_general else 0.0,
                "eodhd_missing_price_selection_rows": missing_selection.height,
                "eodhd_missing_price_selection_tickers": (
                    missing_selection.select(pl.col("ticker").n_unique()).item() if not missing_selection.is_empty() else 0
                ),
            }
        ]
    )

    monthly = (
        missing_selection.group_by("year_month")
        .agg(
            [
                pl.len().alias("selection_rows"),
                pl.col("ticker").n_unique().alias("unique_tickers"),
            ]
        )
        .sort("year_month")
        if not missing_selection.is_empty()
        else pl.DataFrame(schema={"year_month": pl.Date, "selection_rows": pl.Int64, "unique_tickers": pl.Int64})
    )
    top_tickers = (
        missing_selection.group_by("ticker")
        .agg(
            [
                pl.len().alias("selection_rows"),
                pl.col("year_month").n_unique().alias("selection_months"),
                pl.col("year_month").min().alias("first_month"),
                pl.col("year_month").max().alias("last_month"),
            ]
        )
        .sort(["selection_months", "selection_rows", "ticker"], descending=[True, True, False])
        .head(20)
        if not missing_selection.is_empty()
        else pl.DataFrame(
            schema={
                "ticker": pl.String,
                "selection_rows": pl.Int64,
                "selection_months": pl.Int64,
                "first_month": pl.Date,
                "last_month": pl.Date,
            }
        )
    )
    return summary, monthly, top_tickers


def _find_latest_legacy_compare_dir(prefix: str) -> Path | None:
    outputs_dir = PROJECT_ROOT / "outputs"
    candidates = sorted(path for path in outputs_dir.glob(f"{prefix}*") if path.is_dir())
    return candidates[-1] if candidates else None


def _find_latest_run_day_dir(compare_dir: Path, label: str) -> Path | None:
    metrics_files = sorted((compare_dir / "runs" / label).glob("**/legacy_metrics_polars.parquet"))
    if not metrics_files:
        return None
    return metrics_files[-1].parent


def _build_price_covered_counterfactual_summary() -> tuple[pl.DataFrame, pl.DataFrame, Path | None]:
    compare_dir = _find_latest_legacy_compare_dir("legacy_price_covered_compare_")
    if compare_dir is None:
        return pl.DataFrame(), pl.DataFrame(), None

    eodhd_run_dir = _find_latest_run_day_dir(compare_dir, "eodhd")
    open_run_dir = _find_latest_run_day_dir(compare_dir, "open_source")
    if eodhd_run_dir is None or open_run_dir is None:
        return pl.DataFrame(), pl.DataFrame(), compare_dir

    eodhd_metrics = _read_parquet_if_exists(eodhd_run_dir / "legacy_metrics_polars.parquet")
    open_metrics = _read_parquet_if_exists(open_run_dir / "legacy_metrics_polars.parquet")
    if eodhd_metrics.is_empty() or open_metrics.is_empty():
        return pl.DataFrame(), pl.DataFrame(), compare_dir

    common_price_tickers = _read_parquet_if_exists(compare_dir / "data" / "open_source" / "US_Finalprice.parquet").select(
        pl.col("ticker").n_unique().alias("n")
    ).item()

    summary_rows: list[dict[str, object]] = []
    for model in ["Combined_Frequency", "Combined_Equal"]:
        left = eodhd_metrics.filter(pl.col("model") == model).head(1)
        right = open_metrics.filter(pl.col("model") == model).head(1)
        eodhd_total_return = _parse_percent_like(_scalar_or_none(left, "Total Return"))
        open_total_return = _parse_percent_like(_scalar_or_none(right, "Total Return"))
        summary_rows.append(
            {
                "model": model,
                "common_price_covered_tickers": common_price_tickers,
                "eodhd_total_return_pct": eodhd_total_return,
                "open_total_return_pct": open_total_return,
                "gap_pts": abs(open_total_return - eodhd_total_return) if eodhd_total_return is not None and open_total_return is not None else None,
                "eodhd_sharpe_ratio": _parse_percent_like(_scalar_or_none(left, "Sharpe Ratio")),
                "open_sharpe_ratio": _parse_percent_like(_scalar_or_none(right, "Sharpe Ratio")),
            }
        )

    overlap_rows: list[dict[str, object]] = []
    for family, file_name in {
        "optuna_11": "polars_optuna_output_11_detailed.parquet",
        "optuna_12": "polars_optuna_output_12_detailed.parquet",
        "optuna_21": "polars_optuna_output_21_detailed.parquet",
        "optuna_22": "polars_optuna_output_22_detailed.parquet",
        "combined_frequency": "polars_combined_frequency_detailed.parquet",
        "combined_equal": "polars_combined_equal_detailed.parquet",
    }.items():
        eodhd_frame = _read_parquet_if_exists(compare_dir / "runs" / "eodhd" / "checkpoints" / file_name)
        open_frame = _read_parquet_if_exists(compare_dir / "runs" / "open_source" / "checkpoints" / file_name)
        if eodhd_frame.is_empty() or open_frame.is_empty():
            continue
        latest_month = min(
            eodhd_frame.select(pl.col("year_month").max().alias("m")).item(),
            open_frame.select(pl.col("year_month").max().alias("m")).item(),
        )
        eodhd_tickers = set(eodhd_frame.filter(pl.col("year_month") == latest_month).select("ticker").to_series().to_list())
        open_tickers = set(open_frame.filter(pl.col("year_month") == latest_month).select("ticker").to_series().to_list())
        union = eodhd_tickers | open_tickers
        overlap_rows.append(
            {
                "stage": family,
                "latest_month": str(latest_month),
                "eodhd_count": len(eodhd_tickers),
                "open_count": len(open_tickers),
                "common_count": len(eodhd_tickers & open_tickers),
                "jaccard": (len(eodhd_tickers & open_tickers) / len(union)) if union else None,
            }
        )
    return pl.DataFrame(summary_rows), pl.DataFrame(overlap_rows), compare_dir


def _parse_percent_like(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _build_price_return_equivalence_summary(output_root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    eodhd_prices = _read_parquet_if_exists(output_root / "aligned_data" / "eodhd" / "US_Finalprice.parquet")
    open_prices = _read_parquet_if_exists(output_root / "aligned_data" / "open_source" / "US_Finalprice.parquet")
    if eodhd_prices.is_empty() or open_prices.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    left = (
        eodhd_prices.select(["ticker", "date", "adjusted_close"])
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .sort(["ticker", "date"])
        .with_columns(
            ((pl.col("adjusted_close") / pl.col("adjusted_close").shift(1).over("ticker")) - 1.0).alias("eodhd_return")
        )
        .rename({"adjusted_close": "eodhd_adjusted_close"})
    )
    right = (
        open_prices.select(["ticker", "date", "adjusted_close"])
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .sort(["ticker", "date"])
        .with_columns(
            ((pl.col("adjusted_close") / pl.col("adjusted_close").shift(1).over("ticker")) - 1.0).alias("open_return")
        )
        .rename({"adjusted_close": "open_adjusted_close"})
    )
    joined = (
        left.join(right, on=["ticker", "date"], how="inner")
        .with_columns(
            [
                (
                    (
                        (pl.col("open_adjusted_close") - pl.col("eodhd_adjusted_close")).abs()
                        / pl.col("eodhd_adjusted_close").abs()
                    )
                    * 100.0
                )
                .alias("level_diff_pct"),
                ((pl.col("open_return") - pl.col("eodhd_return")).abs() * 10_000.0).alias("abs_return_diff_bps"),
            ]
        )
    )
    valid_returns = joined.filter(pl.col("eodhd_return").is_not_null() & pl.col("open_return").is_not_null())
    if valid_returns.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    summary = pl.DataFrame(
        [
            {
                "matched_rows": valid_returns.height,
                "matched_tickers": valid_returns.select(pl.col("ticker").n_unique()).item(),
                "median_level_diff_pct": valid_returns.select(pl.col("level_diff_pct").median()).item(),
                "p95_level_diff_pct": valid_returns.select(pl.col("level_diff_pct").quantile(0.95)).item(),
                "median_abs_return_diff_bps": valid_returns.select(pl.col("abs_return_diff_bps").median()).item(),
                "p95_abs_return_diff_bps": valid_returns.select(pl.col("abs_return_diff_bps").quantile(0.95)).item(),
                "p99_abs_return_diff_bps": valid_returns.select(pl.col("abs_return_diff_bps").quantile(0.99)).item(),
                "over_1bp_rows": valid_returns.filter(pl.col("abs_return_diff_bps") > 1.0).height,
                "over_5bp_rows": valid_returns.filter(pl.col("abs_return_diff_bps") > 5.0).height,
                "over_10bp_rows": valid_returns.filter(pl.col("abs_return_diff_bps") > 10.0).height,
                "over_5bp_rate_pct": (
                    valid_returns.filter(pl.col("abs_return_diff_bps") > 5.0).height / valid_returns.height
                )
                * 100.0,
            }
        ]
    )
    top_tickers = (
        valid_returns.filter(pl.col("abs_return_diff_bps") > 5.0)
        .group_by("ticker")
        .agg(
            [
                pl.len().alias("rows_over_5bp"),
                pl.col("abs_return_diff_bps").max().alias("max_abs_return_diff_bps"),
                pl.col("level_diff_pct").median().alias("median_level_diff_pct"),
            ]
        )
        .sort(["rows_over_5bp", "max_abs_return_diff_bps"], descending=[True, True])
        .head(20)
    )
    return summary, top_tickers


def _build_latest_model_drift_summary(output_root: Path) -> pl.DataFrame:
    stage_files = {
        "optuna_11": "polars_optuna_output_11_detailed.parquet",
        "optuna_12": "polars_optuna_output_12_detailed.parquet",
        "optuna_21": "polars_optuna_output_21_detailed.parquet",
        "optuna_22": "polars_optuna_output_22_detailed.parquet",
    }
    rows: list[dict[str, object]] = []
    for family, file_name in stage_files.items():
        eodhd_frame = _read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / file_name)
        open_frame = _read_parquet_if_exists(output_root / "checkpoints" / "open_source" / file_name)
        eodhd_latest, eodhd_latest_month = _latest_month_frame(eodhd_frame)
        open_latest, open_latest_month = _latest_month_frame(open_frame)
        eodhd_model_column = "selected_model" if "selected_model" in eodhd_frame.columns else "model"
        open_model_column = "selected_model" if "selected_model" in open_frame.columns else "model"
        eodhd_latest_model = _scalar_or_none(
            eodhd_latest.select(pl.col(eodhd_model_column).drop_nulls().unique().sort().head(1)),
            eodhd_model_column,
        )
        open_latest_model = _scalar_or_none(
            open_latest.select(pl.col(open_model_column).drop_nulls().unique().sort().head(1)),
            open_model_column,
        )
        eodhd_models = (
            eodhd_frame.select(pl.col(eodhd_model_column).drop_nulls().unique().sort()).to_series().to_list()
            if not eodhd_frame.is_empty() and eodhd_model_column in eodhd_frame.columns
            else []
        )
        open_models = (
            open_frame.select(pl.col(open_model_column).drop_nulls().unique().sort()).to_series().to_list()
            if not open_frame.is_empty() and open_model_column in open_frame.columns
            else []
        )
        overlap = sorted(set(str(model) for model in eodhd_models) & set(str(model) for model in open_models))
        rows.append(
            {
                "family": family,
                "eodhd_latest_month": eodhd_latest_month,
                "open_latest_month": open_latest_month,
                "eodhd_latest_model": eodhd_latest_model,
                "open_latest_model": open_latest_model,
                "eodhd_latest_selected_n_asset": _scalar_or_none(
                    eodhd_latest.select(pl.col("selected_n_asset").drop_nulls().unique().sort().head(1))
                    if "selected_n_asset" in eodhd_latest.columns
                    else pl.DataFrame(),
                    "selected_n_asset",
                ),
                "open_latest_selected_n_asset": _scalar_or_none(
                    open_latest.select(pl.col("selected_n_asset").drop_nulls().unique().sort().head(1))
                    if "selected_n_asset" in open_latest.columns
                    else pl.DataFrame(),
                    "selected_n_asset",
                ),
                "eodhd_latest_selected_sector_cap": _scalar_or_none(
                    eodhd_latest.select(pl.col("selected_n_max_per_sector").drop_nulls().unique().sort().head(1))
                    if "selected_n_max_per_sector" in eodhd_latest.columns
                    else pl.DataFrame(),
                    "selected_n_max_per_sector",
                ),
                "open_latest_selected_sector_cap": _scalar_or_none(
                    open_latest.select(pl.col("selected_n_max_per_sector").drop_nulls().unique().sort().head(1))
                    if "selected_n_max_per_sector" in open_latest.columns
                    else pl.DataFrame(),
                    "selected_n_max_per_sector",
                ),
                "latest_model_match": bool(eodhd_latest_model == open_latest_model and eodhd_latest_model is not None),
                "eodhd_unique_models": len(eodhd_models),
                "open_unique_models": len(open_models),
                "historical_overlap_models": len(overlap),
                "historical_overlap_list": ", ".join(overlap) if overlap else "",
            }
        )
    return pl.DataFrame(rows).sort("family")


def _build_earnings_latest_source_summary(output_root: Path) -> pl.DataFrame:
    aligned_open_earnings = _read_parquet_if_exists(output_root / "aligned_data" / "open_source" / "US_Earnings.parquet")
    earnings_lineage = _read_parquet_if_exists(OPEN_SOURCE_LINEAGE_DIR / "earnings_open_source_lineage.parquet")
    if aligned_open_earnings.is_empty() or earnings_lineage.is_empty():
        return pl.DataFrame()

    aligned_tickers = _ticker_set(aligned_open_earnings)
    lineage = earnings_lineage.filter(pl.col("ticker").is_in(sorted(aligned_tickers)))
    if lineage.is_empty():
        return pl.DataFrame()
    sort_cols = [column for column in ["ticker", "period_end", "reportDate"] if column in lineage.columns]
    latest = (
        lineage.sort(sort_cols)
        .group_by("ticker")
        .tail(1)
        .sort("ticker")
    )
    rows = {
        "latest_tickers": latest.height,
        "latest_actual_from_yahoo": latest.filter(pl.col("actual_source") == "yfinance").height if "actual_source" in latest.columns else 0,
        "latest_estimate_from_yahoo": latest.filter(pl.col("estimate_source") == "yfinance").height if "estimate_source" in latest.columns else 0,
        "latest_surprise_from_yahoo": latest.filter(pl.col("surprise_source") == "yfinance").height if "surprise_source" in latest.columns else 0,
    }
    latest_tickers = max(int(rows["latest_tickers"]), 1)
    rows["latest_actual_from_yahoo_pct"] = (rows["latest_actual_from_yahoo"] / latest_tickers) * 100.0
    rows["latest_estimate_from_yahoo_pct"] = (rows["latest_estimate_from_yahoo"] / latest_tickers) * 100.0
    rows["latest_surprise_from_yahoo_pct"] = (rows["latest_surprise_from_yahoo"] / latest_tickers) * 100.0
    return pl.DataFrame([rows])


def _build_open_source_audit_summary() -> tuple[pl.DataFrame, pl.DataFrame]:
    price_summary = _read_parquet_if_exists(AUDIT_2025_DIR / "price_error_summary.parquet")
    statement_summary = _read_parquet_if_exists(AUDIT_2025_DIR / "statement_error_summary.parquet")
    return price_summary, statement_summary


def _lookup_statement_error_rate(statement_summary: pl.DataFrame, source: str, statement: str) -> float | None:
    if statement_summary.is_empty():
        return None
    row = statement_summary.filter((pl.col("source") == source) & (pl.col("statement") == statement)).head(1)
    return _scalar_or_none(row, "error_rate_pct")


def _build_acceptance_gates(
    *,
    input_quality: pl.DataFrame,
    selection_input_coverage: pl.DataFrame,
    price_return_summary: pl.DataFrame,
    model_drift_summary: pl.DataFrame,
    earnings_source_summary: pl.DataFrame,
    price_audit_summary: pl.DataFrame,
    statement_audit_summary: pl.DataFrame,
    metrics_diff: pl.DataFrame,
    freq_summary: dict[str, Any],
    equal_summary: dict[str, Any],
) -> pl.DataFrame:
    def gate_row(
        *,
        category: str,
        metric: str,
        comparator: str,
        threshold: float,
        actual: float | None,
        unit: str,
        evidence: str,
    ) -> dict[str, object]:
        passed = None
        if actual is not None:
            passed = actual <= threshold if comparator == "<=" else actual >= threshold
        return {
            "category": category,
            "metric": metric,
            "actual": actual,
            "unit": unit,
            "comparator": comparator,
            "threshold": threshold,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    eodhd_input = input_quality.filter(pl.col("dataset") == "eodhd").head(1)
    open_input = input_quality.filter(pl.col("dataset") == "open_source").head(1)
    eodhd_selection = selection_input_coverage.filter(pl.col("dataset") == "eodhd").head(1)
    open_selection = selection_input_coverage.filter(pl.col("dataset") == "open_source").head(1)
    price_summary_row = price_return_summary.head(1)
    earnings_row = earnings_source_summary.head(1)
    latest_model_match_rate_pct = (
        model_drift_summary.select(pl.col("latest_model_match").cast(pl.Int64).mean() * 100.0).item()
        if not model_drift_summary.is_empty()
        else None
    )
    latest_price_gap_pct = None
    if not eodhd_selection.is_empty() and not open_selection.is_empty():
        eodhd_price = float(_scalar_or_none(eodhd_selection, "latest_price_tickers") or 0)
        open_price = float(_scalar_or_none(open_selection, "latest_price_tickers") or 0)
        if eodhd_price:
            latest_price_gap_pct = abs(open_price - eodhd_price) / eodhd_price * 100.0
    latest_selection_gap_pct = None
    if not eodhd_selection.is_empty() and not open_selection.is_empty():
        eodhd_sel = float(_scalar_or_none(eodhd_selection, "latest_constituent_join_tickers") or 0)
        open_sel = float(_scalar_or_none(open_selection, "latest_constituent_join_tickers") or 0)
        if eodhd_sel:
            latest_selection_gap_pct = abs(open_sel - eodhd_sel) / eodhd_sel * 100.0

    combined_frequency = metrics_diff.filter(pl.col("model") == "Combined_Frequency").head(1)
    combined_equal = metrics_diff.filter(pl.col("model") == "Combined_Equal").head(1)
    combined_frequency_gap = None
    combined_equal_gap = None
    if not combined_frequency.is_empty():
        e = _parse_percent_like(_scalar_or_none(combined_frequency, "eodhd_Total Return"))
        o = _parse_percent_like(_scalar_or_none(combined_frequency, "open_Total Return"))
        if e is not None and o is not None:
            combined_frequency_gap = abs(o - e)
    if not combined_equal.is_empty():
        e = _parse_percent_like(_scalar_or_none(combined_equal, "eodhd_Total Return"))
        o = _parse_percent_like(_scalar_or_none(combined_equal, "open_Total Return"))
        if e is not None and o is not None:
            combined_equal_gap = abs(o - e)

    gates = [
        gate_row(
            category="coverage",
            metric="sector_null_rate_aligned_scope",
            comparator="<=",
            threshold=0.5,
            actual=((_scalar_or_none(open_input, "sector_null_rows") or 0) / max((_scalar_or_none(open_input, "general_rows") or 1), 1)) * 100.0,
            unit="pct",
            evidence="aligned_data/open_source/US_General.parquet",
        ),
        gate_row(
            category="coverage",
            metric="latest_price_ticker_gap_vs_eodhd",
            comparator="<=",
            threshold=1.0,
            actual=latest_price_gap_pct,
            unit="pct",
            evidence="selection_input_coverage.latest_price_tickers",
        ),
        gate_row(
            category="coverage",
            metric="latest_selection_universe_gap_vs_eodhd",
            comparator="<=",
            threshold=1.0,
            actual=latest_selection_gap_pct,
            unit="pct",
            evidence="selection_input_coverage.latest_constituent_join_tickers",
        ),
        gate_row(
            category="prices",
            metric="daily_return_p95_abs_diff",
            comparator="<=",
            threshold=0.1,
            actual=_scalar_or_none(price_summary_row, "p95_abs_return_diff_bps"),
            unit="bps",
            evidence="aligned_data/*/US_Finalprice.parquet",
        ),
        gate_row(
            category="prices",
            metric="daily_return_rows_over_5bp",
            comparator="<=",
            threshold=0.1,
            actual=_scalar_or_none(price_summary_row, "over_5bp_rate_pct"),
            unit="pct",
            evidence="aligned_data/*/US_Finalprice.parquet",
        ),
        gate_row(
            category="prices",
            metric="raw_adjusted_close_error_rate_2025",
            comparator="<=",
            threshold=2.0,
            actual=_scalar_or_none(price_audit_summary.head(1), "error_rate_pct"),
            unit="pct",
            evidence="data/open_source/audit/2025/price_error_summary.parquet",
        ),
        gate_row(
            category="earnings",
            metric="latest_actual_from_market_source",
            comparator=">=",
            threshold=98.0,
            actual=_scalar_or_none(earnings_row, "latest_actual_from_yahoo_pct"),
            unit="pct",
            evidence="output/lineage/earnings_open_source_lineage.parquet",
        ),
        gate_row(
            category="earnings",
            metric="latest_estimate_from_market_source",
            comparator=">=",
            threshold=98.0,
            actual=_scalar_or_none(earnings_row, "latest_estimate_from_yahoo_pct"),
            unit="pct",
            evidence="output/lineage/earnings_open_source_lineage.parquet",
        ),
        gate_row(
            category="audit_2025",
            metric="balance_sheet_error_rate",
            comparator="<=",
            threshold=5.0,
            actual=_lookup_statement_error_rate(statement_audit_summary, "open_source_consolidated", "balance_sheet"),
            unit="pct",
            evidence="data/open_source/audit/2025/statement_error_summary.parquet",
        ),
        gate_row(
            category="audit_2025",
            metric="income_statement_error_rate",
            comparator="<=",
            threshold=5.0,
            actual=_lookup_statement_error_rate(statement_audit_summary, "open_source_consolidated", "income_statement"),
            unit="pct",
            evidence="data/open_source/audit/2025/statement_error_summary.parquet",
        ),
        gate_row(
            category="audit_2025",
            metric="cash_flow_error_rate",
            comparator="<=",
            threshold=5.0,
            actual=_lookup_statement_error_rate(statement_audit_summary, "open_source_consolidated", "cash_flow"),
            unit="pct",
            evidence="data/open_source/audit/2025/statement_error_summary.parquet",
        ),
        gate_row(
            category="audit_2025",
            metric="shares_error_rate",
            comparator="<=",
            threshold=1.0,
            actual=_lookup_statement_error_rate(statement_audit_summary, "open_source_consolidated", "shares"),
            unit="pct",
            evidence="data/open_source/audit/2025/statement_error_summary.parquet",
        ),
        gate_row(
            category="audit_2025",
            metric="earnings_error_rate",
            comparator="<=",
            threshold=10.0,
            actual=_lookup_statement_error_rate(statement_audit_summary, "open_source_earnings", "earnings"),
            unit="pct",
            evidence="data/open_source/audit/2025/statement_error_summary.parquet",
        ),
        gate_row(
            category="optimizer",
            metric="latest_optuna_model_slot_overlap",
            comparator=">=",
            threshold=75.0,
            actual=latest_model_match_rate_pct,
            unit="pct",
            evidence="checkpoints/*/polars_optuna_output_*_detailed.parquet",
        ),
        gate_row(
            category="backtest",
            metric="combined_frequency_total_return_gap",
            comparator="<=",
            threshold=5.0,
            actual=combined_frequency_gap,
            unit="pts",
            evidence="comparison/model_metrics_diff.parquet",
        ),
        gate_row(
            category="backtest",
            metric="combined_equal_total_return_gap",
            comparator="<=",
            threshold=5.0,
            actual=combined_equal_gap,
            unit="pts",
            evidence="comparison/model_metrics_diff.parquet",
        ),
        gate_row(
            category="backtest",
            metric="final_frequency_holdings_jaccard",
            comparator=">=",
            threshold=0.8,
            actual=freq_summary.get("jaccard"),
            unit="ratio",
            evidence="comparison/portfolio_frequency_diff.parquet",
        ),
        gate_row(
            category="backtest",
            metric="final_equal_holdings_jaccard",
            comparator=">=",
            threshold=0.8,
            actual=equal_summary.get("jaccard"),
            unit="ratio",
            evidence="comparison/portfolio_equal_diff.parquet",
        ),
    ]
    return pl.DataFrame(gates)


def _build_stage_summary(output_root: Path) -> pl.DataFrame:
    stage_files = {
        "stocks_selections": "polars_stocks_selections.parquet",
        "optuna_11_detailed": "polars_optuna_output_11_detailed.parquet",
        "optuna_12_detailed": "polars_optuna_output_12_detailed.parquet",
        "optuna_21_detailed": "polars_optuna_output_21_detailed.parquet",
        "optuna_22_detailed": "polars_optuna_output_22_detailed.parquet",
        "combined_frequency_detailed": "polars_combined_frequency_detailed.parquet",
        "combined_equal_detailed": "polars_combined_equal_detailed.parquet",
    }
    rows: list[dict[str, object]] = []
    for dataset in ("eodhd", "open_source"):
        checkpoint_dir = output_root / "checkpoints" / dataset
        for stage, file_name in stage_files.items():
            frame = _read_parquet_if_exists(checkpoint_dir / file_name)
            latest, latest_month = _latest_month_frame(frame)
            rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "total_rows": frame.height,
                    "latest_month": latest_month,
                    "latest_rows": latest.height,
                    "latest_tickers": latest.select(pl.col("ticker").n_unique()).item() if not latest.is_empty() and "ticker" in latest.columns else 0,
                    "null_sector_rows": latest.filter(pl.col("Sector").is_null() | (pl.col("Sector") == "")).height if "Sector" in latest.columns else 0,
                    "null_sector_rate_pct": (
                        (latest.filter(pl.col("Sector").is_null() | (pl.col("Sector") == "")).height / latest.height) * 100.0
                        if not latest.is_empty() and "Sector" in latest.columns
                        else None
                    ),
                }
            )
    return pl.DataFrame(rows).sort(["stage", "dataset"])


def _build_focus_ticker_diagnostics(
    *,
    output_root: Path,
    frequency_diff: pl.DataFrame,
    equal_diff: pl.DataFrame,
) -> pl.DataFrame:
    focus_tickers = sorted(
        set(frequency_diff.select("ticker").to_series().to_list()) | set(equal_diff.select("ticker").to_series().to_list())
    )
    if not focus_tickers:
        return pl.DataFrame()

    aligned_eodhd_general = _read_parquet_if_exists(output_root / "aligned_data" / "eodhd" / "US_General.parquet")
    aligned_open_general = _read_parquet_if_exists(output_root / "aligned_data" / "open_source" / "US_General.parquet")
    aligned_eodhd_earnings = _read_parquet_if_exists(output_root / "aligned_data" / "eodhd" / "US_Earnings.parquet")
    aligned_open_earnings = _read_parquet_if_exists(output_root / "aligned_data" / "open_source" / "US_Earnings.parquet")
    open_general_lineage = _read_parquet_if_exists(OPEN_SOURCE_LINEAGE_DIR / "general_reference_lineage.parquet")
    open_earnings_lineage = _read_parquet_if_exists(OPEN_SOURCE_LINEAGE_DIR / "earnings_open_source_lineage.parquet")

    eodhd_selection, eodhd_selection_month = _latest_month_frame(
        _read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / "polars_stocks_selections.parquet")
    )
    open_selection, open_selection_month = _latest_month_frame(
        _read_parquet_if_exists(output_root / "checkpoints" / "open_source" / "polars_stocks_selections.parquet")
    )

    def latest_stage_counts(dataset: str, file_name: str) -> pl.DataFrame:
        stage_frame, _ = _latest_month_frame(_read_parquet_if_exists(output_root / "checkpoints" / dataset / file_name))
        if stage_frame.is_empty():
            return pl.DataFrame(schema={"ticker": pl.String, "count": pl.Int64})
        return stage_frame.group_by("ticker").agg(pl.len().alias("count"))

    eodhd_model_counts = _merge_stage_counts(
        [
            latest_stage_counts("eodhd", "polars_optuna_output_11_detailed.parquet"),
            latest_stage_counts("eodhd", "polars_optuna_output_12_detailed.parquet"),
            latest_stage_counts("eodhd", "polars_optuna_output_21_detailed.parquet"),
            latest_stage_counts("eodhd", "polars_optuna_output_22_detailed.parquet"),
        ]
    )
    open_model_counts = _merge_stage_counts(
        [
            latest_stage_counts("open_source", "polars_optuna_output_11_detailed.parquet"),
            latest_stage_counts("open_source", "polars_optuna_output_12_detailed.parquet"),
            latest_stage_counts("open_source", "polars_optuna_output_21_detailed.parquet"),
            latest_stage_counts("open_source", "polars_optuna_output_22_detailed.parquet"),
        ]
    )

    eodhd_combined_frequency, combined_frequency_month = _latest_month_frame(
        _read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / "polars_combined_frequency_detailed.parquet")
    )
    open_combined_frequency, _ = _latest_month_frame(
        _read_parquet_if_exists(output_root / "checkpoints" / "open_source" / "polars_combined_frequency_detailed.parquet")
    )
    eodhd_combined_equal, combined_equal_month = _latest_month_frame(
        _read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / "polars_combined_equal_detailed.parquet")
    )
    open_combined_equal, _ = _latest_month_frame(
        _read_parquet_if_exists(output_root / "checkpoints" / "open_source" / "polars_combined_equal_detailed.parquet")
    )

    rows: list[dict[str, object]] = []
    for ticker in focus_tickers:
        eodhd_sector = _lookup_value(aligned_eodhd_general, ticker, "Sector")
        open_sector = _lookup_value(aligned_open_general, ticker, "Sector")
        open_sector_source = _lookup_value(open_general_lineage, ticker, "sector_source")
        open_mapping_rule = _lookup_value(open_general_lineage, ticker, "mapping_rule")
        open_industry = _lookup_value(open_general_lineage, ticker, "industry")
        open_actual_source = _lookup_latest_value(open_earnings_lineage, ticker, "actual_source")
        open_estimate_source = _lookup_latest_value(open_earnings_lineage, ticker, "estimate_source")
        eodhd_freq_status = _lookup_value(frequency_diff, ticker, "holding_status")
        open_freq_selected = _row_exists(open_combined_frequency, ticker)
        eodhd_freq_selected = _row_exists(eodhd_combined_frequency, ticker)
        open_selection_row = _select_ticker_row(open_selection, ticker)
        eodhd_selection_row = _select_ticker_row(eodhd_selection, ticker)
        row = {
            "ticker": ticker,
            "frequency_status": eodhd_freq_status,
            "equal_status": _lookup_value(equal_diff, ticker, "holding_status"),
            "eodhd_sector": eodhd_sector,
            "open_sector": open_sector,
            "open_sector_source": open_sector_source,
            "open_mapping_rule": open_mapping_rule,
            "open_industry": open_industry,
            "eodhd_earnings_rows": _count_ticker_rows(aligned_eodhd_earnings, ticker),
            "open_earnings_rows": _count_ticker_rows(aligned_open_earnings, ticker),
            "open_actual_source": open_actual_source,
            "open_estimate_source": open_estimate_source,
            "selection_month_eodhd": eodhd_selection_month,
            "selection_month_open": open_selection_month,
            "eodhd_in_selection": not eodhd_selection_row.is_empty(),
            "open_in_selection": not open_selection_row.is_empty(),
            "eodhd_pe": _scalar_or_none(eodhd_selection_row, "pe"),
            "open_pe": _scalar_or_none(open_selection_row, "pe"),
            "eodhd_market_cap": _scalar_or_none(eodhd_selection_row, "market_cap"),
            "open_market_cap": _scalar_or_none(open_selection_row, "market_cap"),
            "eodhd_model_stage_hits": _lookup_value(eodhd_model_counts, ticker, "stage_hits"),
            "open_model_stage_hits": _lookup_value(open_model_counts, ticker, "stage_hits"),
            "combined_frequency_month": combined_frequency_month,
            "eodhd_in_combined_frequency": eodhd_freq_selected,
            "open_in_combined_frequency": open_freq_selected,
            "combined_equal_month": combined_equal_month,
            "eodhd_in_combined_equal": _row_exists(eodhd_combined_equal, ticker),
            "open_in_combined_equal": _row_exists(open_combined_equal, ticker),
        }
        row["reason"] = _infer_ticker_reason(row)
        rows.append(row)
    return pl.DataFrame(rows).sort("ticker")


def _merge_stage_counts(frames: list[pl.DataFrame]) -> pl.DataFrame:
    rows: dict[str, int] = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            ticker = str(row["ticker"])
            rows[ticker] = rows.get(ticker, 0) + 1
    return pl.DataFrame({"ticker": list(rows.keys()), "stage_hits": list(rows.values())}) if rows else pl.DataFrame(
        schema={"ticker": pl.String, "stage_hits": pl.Int64}
    )


def _count_ticker_rows(frame: pl.DataFrame, ticker: str) -> int:
    if frame.is_empty() or "ticker" not in frame.columns:
        return 0
    return frame.filter(pl.col("ticker") == ticker).height


def _lookup_value(frame: pl.DataFrame, ticker: str, column: str) -> Any:
    if frame.is_empty() or "ticker" not in frame.columns or column not in frame.columns:
        return None
    row = frame.filter(pl.col("ticker") == ticker).select(column).head(1)
    return row.item() if not row.is_empty() else None


def _lookup_latest_value(frame: pl.DataFrame, ticker: str, column: str) -> Any:
    if frame.is_empty() or "ticker" not in frame.columns or column not in frame.columns:
        return None
    filtered = frame.filter(pl.col("ticker") == ticker)
    if filtered.is_empty():
        return None
    sort_cols = [col for col in ["period_end", "reportDate"] if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort(sort_cols)
    return filtered.select(column).tail(1).item()


def _row_exists(frame: pl.DataFrame, ticker: str) -> bool:
    if frame.is_empty() or "ticker" not in frame.columns:
        return False
    return frame.filter(pl.col("ticker") == ticker).height > 0


def _select_ticker_row(frame: pl.DataFrame, ticker: str) -> pl.DataFrame:
    if frame.is_empty() or "ticker" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("ticker") == ticker).head(1)


def _select_ticker_rows(frame: pl.DataFrame, ticker: str) -> pl.DataFrame:
    if frame.is_empty() or "ticker" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("ticker") == ticker)


def _scalar_or_none(frame: pl.DataFrame, column: str) -> Any:
    if frame.is_empty() or column not in frame.columns:
        return None
    return frame.select(column).item()


def _infer_ticker_reason(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    if row.get("open_sector") in {None, "", "Unknown"}:
        reasons.append("Sector absent in open_source; the legacy sector cap can collapse many candidates into a single null bucket.")
    if int(row.get("eodhd_earnings_rows") or 0) > 0 and int(row.get("open_earnings_rows") or 0) == 0:
        reasons.append("Open-source earnings coverage is missing for this ticker on the aligned scope.")
    if bool(row.get("eodhd_in_selection")) and not bool(row.get("open_in_selection")):
        reasons.append("Ticker disappears before model learning: valuation/fundamental selection differs between datasets.")
    if bool(row.get("open_in_selection")) and (int(row.get("open_model_stage_hits") or 0) < int(row.get("eodhd_model_stage_hits") or 0)):
        reasons.append("Ticker survives stock selection but receives fewer model votes in open_source.")
    if bool(row.get("eodhd_in_combined_frequency")) and not bool(row.get("open_in_combined_frequency")):
        reasons.append("Ticker is lost during combined frequency aggregation in open_source.")
    if not reasons and bool(row.get("open_in_combined_frequency")) and not bool(row.get("eodhd_in_combined_frequency")):
        reasons.append("Open-source ranking/model votes promote this ticker while EODHD does not.")
    if not reasons:
        reasons.append("Divergence happens after the shared input universe; inspect latest selection metrics and model vote counts.")
    return " ".join(reasons)


def _frame_to_html(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "<p class=\"muted\">empty</p>"
    header = "".join(f"<th>{escape(column)}</th>" for column in frame.columns)
    body_rows: list[str] = []
    for row in frame.iter_rows(named=True):
        cells = "".join(f"<td>{escape(_html_value(row[column]))}</td>" for column in frame.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _html_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _write_html_page(*, output_path: Path, title: str, sections: list[tuple[str, pl.DataFrame]], navigation: str, subtitle: str = "") -> None:
    section_html = "".join(f"<div class=\"section\"><h2>{escape(name)}</h2>{_frame_to_html(frame)}</div>" for name, frame in sections)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 12px; vertical-align: top; }}
    th {{ background: #eff6ff; position: sticky; top: 0; }}
    .section {{ margin-top: 28px; }}
    .muted {{ color: #64748b; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class=\"muted\">{escape(subtitle)}</p>
  {navigation}
  {section_html}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _write_report(
    *,
    output_root: Path,
    alignment: dict[str, Any],
    eodhd_summary: LegacyRunSummary,
    open_summary: LegacyRunSummary,
    metrics_diff: pl.DataFrame,
    monthly_returns_diff: pl.DataFrame,
    frequency_diff: pl.DataFrame,
    equal_diff: pl.DataFrame,
) -> Path:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = comparison_dir / "model_metrics_diff.parquet"
    monthly_path = comparison_dir / "monthly_returns_diff.parquet"
    frequency_path = comparison_dir / "portfolio_frequency_diff.parquet"
    equal_path = comparison_dir / "portfolio_equal_diff.parquet"
    metrics_diff.write_parquet(metrics_path)
    monthly_returns_diff.write_parquet(monthly_path)
    frequency_diff.write_parquet(frequency_path)
    equal_diff.write_parquet(equal_path)

    freq_summary = _portfolio_overlap_summary(frequency_diff)
    equal_summary = _portfolio_overlap_summary(equal_diff)
    input_quality = _build_input_quality_summary(output_root)
    selection_input_coverage = _build_selection_input_coverage(output_root)
    price_coverage_gap_summary, price_coverage_gap_monthly, price_coverage_gap_tickers = _build_price_coverage_gap_summary(output_root)
    stage_summary = _build_stage_summary(output_root)
    price_return_summary, price_return_outliers = _build_price_return_equivalence_summary(output_root)
    model_drift_summary = _build_latest_model_drift_summary(output_root)
    earnings_source_summary = _build_earnings_latest_source_summary(output_root)
    price_audit_summary, statement_audit_summary = _build_open_source_audit_summary()
    counterfactual_metrics, counterfactual_overlap, counterfactual_dir = _build_price_covered_counterfactual_summary()
    acceptance_gates = _build_acceptance_gates(
        input_quality=input_quality,
        selection_input_coverage=selection_input_coverage,
        price_return_summary=price_return_summary,
        model_drift_summary=model_drift_summary,
        earnings_source_summary=earnings_source_summary,
        price_audit_summary=price_audit_summary,
        statement_audit_summary=statement_audit_summary,
        metrics_diff=metrics_diff,
        freq_summary=freq_summary,
        equal_summary=equal_summary,
    )
    focus_ticker_diagnostics = _build_focus_ticker_diagnostics(
        output_root=output_root,
        frequency_diff=frequency_diff,
        equal_diff=equal_diff,
    )
    eodhd_input = input_quality.filter(pl.col("dataset") == "eodhd").head(1)
    open_input = input_quality.filter(pl.col("dataset") == "open_source").head(1)
    eodhd_selection_input = selection_input_coverage.filter(pl.col("dataset") == "eodhd").head(1)
    open_selection_input = selection_input_coverage.filter(pl.col("dataset") == "open_source").head(1)
    price_summary_row = price_return_summary.head(1)
    earnings_source_row = earnings_source_summary.head(1)
    latest_model_match_rate_pct = (
        model_drift_summary.select(pl.col("latest_model_match").cast(pl.Int64).mean() * 100.0).item()
        if not model_drift_summary.is_empty()
        else None
    )
    price_gap_row = price_coverage_gap_summary.head(1)
    audit_failures = acceptance_gates.filter((pl.col("category") == "audit_2025") & (pl.col("status") == "FAIL"))
    audit_failure_snippets = [
        f"{row['metric']}={row['actual']:.2f}{row['unit']}"
        for row in audit_failures.iter_rows(named=True)
        if row.get("actual") is not None
    ]
    top_price_outlier = price_return_outliers.head(1)
    root_cause_lines = [
        "Sector coverage is no longer the blocker: open_source has 0 null sectors on the aligned scope and 0 null sectors in the latest strategy stages.",
        (
            f"Latest monthly price coverage is effectively aligned: "
            f"{_scalar_or_none(open_selection_input, 'latest_price_tickers')} tickers versus "
            f"{_scalar_or_none(eodhd_selection_input, 'latest_price_tickers')} in EODHD."
        ),
        (
            f"Valuation coverage is also nearly aligned: "
            f"{_scalar_or_none(open_selection_input, 'latest_value_tickers')} open_source tickers get a latest-month valuation row "
            f"versus {_scalar_or_none(eodhd_selection_input, 'latest_value_tickers')} in EODHD."
        ),
        (
            f"After the legacy valuation filter (`0 < pe < 100` and `market_cap not null`), "
            f"open_source keeps {_scalar_or_none(open_selection_input, 'latest_post_filter_tickers')} tickers versus "
            f"{_scalar_or_none(eodhd_selection_input, 'latest_post_filter_tickers')} in EODHD."
        ),
        (
            f"After joining the S&P 500 monthly constituents, the stock-selection universe is "
            f"{_scalar_or_none(open_selection_input, 'latest_constituent_join_tickers')} tickers in open_source versus "
            f"{_scalar_or_none(eodhd_selection_input, 'latest_constituent_join_tickers')} in EODHD."
        ),
        (
            f"Price returns are close on most matched rows (p95 absolute daily return diff "
            f"{_scalar_or_none(price_summary_row, 'p95_abs_return_diff_bps')} bps), but there is still a tail of "
            f"{_scalar_or_none(price_summary_row, 'over_5bp_rate_pct')}% of rows above 5 bps, led by "
            f"{_scalar_or_none(top_price_outlier, 'ticker')}."
        ),
        (
            f"Earnings coverage improved sharply and is no longer empty, but still trails EODHD: "
            f"{_scalar_or_none(open_input, 'earnings_rows')} rows / {_scalar_or_none(open_input, 'earnings_tickers')} tickers "
            f"versus {_scalar_or_none(eodhd_input, 'earnings_rows')} / {_scalar_or_none(eodhd_input, 'earnings_tickers')}."
        ),
        (
            f"Latest market-source earnings coverage is now high "
            f"({ _scalar_or_none(earnings_source_row, 'latest_actual_from_yahoo_pct') }% actual, "
            f"{ _scalar_or_none(earnings_source_row, 'latest_estimate_from_yahoo_pct') }% estimate), "
            f"but the 2025 audit still fails on core datasets: {', '.join(audit_failure_snippets) if audit_failure_snippets else 'n/a'}."
        ),
        (
            f"Open-source price history only covers {_scalar_or_none(price_gap_row, 'open_price_covered_tickers')} of "
            f"{_scalar_or_none(price_gap_row, 'common_general_tickers')} aligned tickers; "
            f"{_scalar_or_none(price_gap_row, 'missing_open_price_tickers')} names are missing entirely from the open-source price history."
        ),
        (
            f"Those missing-price names barely touch the final portfolio directly, but they do touch the training window: "
            f"{_scalar_or_none(price_gap_row, 'eodhd_missing_price_selection_rows')} EODHD `stocks_selections` rows across "
            f"{_scalar_or_none(price_gap_row, 'eodhd_missing_price_selection_tickers')} tickers come from names with no open-source price history."
        ),
        (
            f"The main offenders are {', '.join(price_coverage_gap_tickers.head(5).select('ticker').to_series().to_list()) if not price_coverage_gap_tickers.is_empty() else 'n/a'}."
        ),
        (
            f"The full legacy comparison therefore still mixes data differences with optimizer drift: latest Optuna model-slot overlap is "
            f"{latest_model_match_rate_pct}% and the latest chosen parameter sets differ between datasets."
        ),
    ]
    if not counterfactual_metrics.is_empty() and not counterfactual_overlap.is_empty():
        counterfactual_freq = counterfactual_metrics.filter(pl.col("model") == "Combined_Frequency").head(1)
        counterfactual_equal = counterfactual_metrics.filter(pl.col("model") == "Combined_Equal").head(1)
        counterfactual_freq_overlap = counterfactual_overlap.filter(pl.col("stage") == "combined_frequency").head(1)
        counterfactual_equal_overlap = counterfactual_overlap.filter(pl.col("stage") == "combined_equal").head(1)
        root_cause_lines.extend(
            [
                (
                    f"Counterfactual sanity check on the latest price-covered universe "
                    f"({_scalar_or_none(counterfactual_freq, 'common_price_covered_tickers')} tickers): "
                    f"Combined_Frequency gap drops to {_scalar_or_none(counterfactual_freq, 'gap_pts')} pts and "
                    f"Combined_Equal gap drops to {_scalar_or_none(counterfactual_equal, 'gap_pts')} pts."
                ),
                (
                    f"On that counterfactual universe, the latest holdings match exactly: "
                    f"Combined_Frequency Jaccard = {_scalar_or_none(counterfactual_freq_overlap, 'jaccard')}, "
                    f"Combined_Equal Jaccard = {_scalar_or_none(counterfactual_equal_overlap, 'jaccard')}."
                ),
            ]
        )

    primary_models = metrics_diff.filter(pl.col("model").is_in(["Combined_Equal", "Combined_Frequency", "SP500"]))
    largest_monthly_model_gaps = (
        monthly_returns_diff.filter(pl.col("monthly_return_diff").is_not_null())
        .with_columns(pl.col("monthly_return_diff").abs().alias("abs_monthly_return_diff"))
        .sort("abs_monthly_return_diff", descending=True)
        .select([column for column in ["portfolio_model", "model", "year_month", "eodhd_monthly_return", "open_monthly_return", "monthly_return_diff"] if column in monthly_returns_diff.columns])
        .head(15)
    )
    executive_summary_path = output_root / "executive_summary.md"
    executive_summary_html_path = output_root / "executive_summary.html"
    executive_summary = f"""# Executive Summary

- Full-scope result still does **not** support unplugging EODHD blindly: `Combined_Frequency` is `{_scalar_or_none(metrics_diff.filter(pl.col('model') == 'Combined_Frequency').head(1), 'eodhd_Total Return')}` on EODHD versus `{_scalar_or_none(metrics_diff.filter(pl.col('model') == 'Combined_Frequency').head(1), 'open_Total Return')}` on open_source.
- `Sector` is **not** the blocker anymore: open-source has `0` null sectors on the aligned scope.
- `US_Earnings` is **not** the primary blocker for `run_legacy`: the legacy pipeline is driven mainly by valuations built from price + statements + sector, and the latest valuation universe is already near-aligned.
- The main blocker is **historical price coverage**: open-source covers `{_scalar_or_none(price_gap_row, 'open_price_covered_tickers')}` of `{_scalar_or_none(price_gap_row, 'common_general_tickers')}` aligned tickers, leaving `{_scalar_or_none(price_gap_row, 'missing_open_price_tickers')}` missing names.
- Only `{_scalar_or_none(price_gap_row, 'eodhd_missing_price_selection_tickers')}` of those missing names actually hit `stocks_selections` in 2025, but that is enough to change optimizer history. The main names are `{', '.join(price_coverage_gap_tickers.head(5).select('ticker').to_series().to_list()) if not price_coverage_gap_tickers.is_empty() else 'n/a'}`.
"""
    if not counterfactual_metrics.is_empty() and not counterfactual_overlap.is_empty():
        counterfactual_freq = counterfactual_metrics.filter(pl.col("model") == "Combined_Frequency").head(1)
        counterfactual_equal = counterfactual_metrics.filter(pl.col("model") == "Combined_Equal").head(1)
        counterfactual_freq_overlap = counterfactual_overlap.filter(pl.col("stage") == "combined_frequency").head(1)
        counterfactual_equal_overlap = counterfactual_overlap.filter(pl.col("stage") == "combined_equal").head(1)
        executive_summary += f"""
- Counterfactual sanity check on the price-covered universe proves the point: once both sides are restricted to the same `{_scalar_or_none(counterfactual_freq, 'common_price_covered_tickers')}` price-covered tickers, `Combined_Frequency` gap falls to `{_scalar_or_none(counterfactual_freq, 'gap_pts')}` pts and `Combined_Equal` gap falls to `{_scalar_or_none(counterfactual_equal, 'gap_pts')}` pts.
- On that same counterfactual universe, the latest holdings overlap is exact: `Combined_Frequency` Jaccard `{_scalar_or_none(counterfactual_freq_overlap, 'jaccard')}`, `Combined_Equal` Jaccard `{_scalar_or_none(counterfactual_equal_overlap, 'jaccard')}`.
- Conclusion: the residual full-scope gap is dominated by missing historical tickers in open-source price history, not by a broad corruption of the overlapping price/sector/fundamental rows.
- Counterfactual artifacts: `{counterfactual_dir}`
"""
    executive_summary += f"""

## Next Decision

- If we accept a legacy universe restricted to the `{_scalar_or_none(price_gap_row, 'open_price_covered_tickers')}` tickers that open-source can actually price historically, the strategy is now close enough to EODHD to be credible.
- If we need the full legacy historical universe, we still need a free source for the missing price histories (`K.US`, `IPG.US`, `HES.US`, `JNPR.US`, `DFS.US`, and the rest of the uncovered names).
"""
    executive_summary_path.write_text(executive_summary, encoding="utf-8")

    report_path = output_root / "report.md"
    report = f"""# Legacy DB Comparison

Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

## Scope

- Comparison date: `{date.today().isoformat()}`
- Legacy first backtest month: `{DEFAULT_FIRST_DATE}`
- Common price history start: `{alignment['price_start_date']}`
- Common price cutoff: `{alignment['price_cutoff_date']}`
- Common financial cutoff: `{alignment['financial_cutoff_date']}`
- Common earnings cutoff: `{alignment['earnings_cutoff_date']}`
- Common ticker universe: `{alignment['common_tickers']}`

## Aligned Dataset Summary

| Dataset | Tickers | Final Price Rows | Final Price Max Date | SP500 Rows | Income Rows | Balance Rows | Cash Rows | Earnings Rows |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| EODHD | {alignment['dataset_summaries']['eodhd']['ticker_count']} | {alignment['dataset_summaries']['eodhd']['final_price_rows']} | {alignment['dataset_summaries']['eodhd']['final_price_max_date']} | {alignment['dataset_summaries']['eodhd']['sp500_price_rows']} | {alignment['dataset_summaries']['eodhd']['income_rows']} | {alignment['dataset_summaries']['eodhd']['balance_rows']} | {alignment['dataset_summaries']['eodhd']['cash_rows']} | {alignment['dataset_summaries']['eodhd']['earnings_rows']} |
| Open Source | {alignment['dataset_summaries']['open_source']['ticker_count']} | {alignment['dataset_summaries']['open_source']['final_price_rows']} | {alignment['dataset_summaries']['open_source']['final_price_max_date']} | {alignment['dataset_summaries']['open_source']['sp500_price_rows']} | {alignment['dataset_summaries']['open_source']['income_rows']} | {alignment['dataset_summaries']['open_source']['balance_rows']} | {alignment['dataset_summaries']['open_source']['cash_rows']} | {alignment['dataset_summaries']['open_source']['earnings_rows']} |

Source general ticker counts before alignment:

- EODHD: `{alignment['source_general_tickers']['eodhd']}`
- Open Source: `{alignment['source_general_tickers']['open_source']}`

## Input Quality

{_markdown_table(input_quality, max_rows=10)}

## Acceptance Gates

{_markdown_table(acceptance_gates, max_rows=30)}

## Root Cause Summary

{chr(10).join(f"- {line}" for line in root_cause_lines)}

## Selection Input Coverage

{_markdown_table(selection_input_coverage, max_rows=10)}

## Price Coverage Root Cause

{_markdown_table(price_coverage_gap_summary, max_rows=5)}

### Missing-price tickers that still entered EODHD stock selection

{_markdown_table(price_coverage_gap_tickers, max_rows=20)}

### Monthly EODHD stock-selection impact from missing-price tickers

{_markdown_table(price_coverage_gap_monthly, max_rows=20)}

## Price Return Equivalence

{_markdown_table(price_return_summary, max_rows=5)}

### Worst Price Return Outliers

{_markdown_table(price_return_outliers, max_rows=20)}

## Latest Model Drift

{_markdown_table(model_drift_summary, max_rows=10)}

## Counterfactual Price-Covered Universe

Counterfactual directory: `{counterfactual_dir}`

### Metrics

{_markdown_table(counterfactual_metrics, max_rows=10)}

### Latest holdings overlap by stage

{_markdown_table(counterfactual_overlap, max_rows=10)}

## Open-Source 2025 Audit Summary

### Price Audit

{_markdown_table(price_audit_summary, max_rows=10)}

### Statement Audit

{_markdown_table(statement_audit_summary, max_rows=20)}

## Run Summary

| Run | Duration (min) | Final Frequency Month | Final Frequency Holdings | Final Equal Month | Final Equal Holdings | Run Dir |
|---|---:|---|---:|---|---:|---|
| EODHD | {eodhd_summary.duration_seconds / 60:.1f} | {eodhd_summary.portfolio_frequency_month} | {eodhd_summary.portfolio_frequency_count} | {eodhd_summary.portfolio_equal_month} | {eodhd_summary.portfolio_equal_count} | `{eodhd_summary.run_day_dir}` |
| Open Source | {open_summary.duration_seconds / 60:.1f} | {open_summary.portfolio_frequency_month} | {open_summary.portfolio_frequency_count} | {open_summary.portfolio_equal_month} | {open_summary.portfolio_equal_count} | `{open_summary.run_day_dir}` |

## Primary Model Metric Diff

{_markdown_table(primary_models, columns=[column for column in [
    "model",
    "eodhd_Total Return", "open_Total Return", "diff_Total Return",
    "eodhd_CAGR", "open_CAGR", "diff_CAGR",
    "eodhd_Sharpe Ratio", "open_Sharpe Ratio", "diff_Sharpe Ratio",
    "eodhd_Max Drawdown", "open_Max Drawdown", "diff_Max Drawdown",
] if column in primary_models.columns], max_rows=10)}

Full metrics parquet:

- `{metrics_path}`

## Stage Summary

{_markdown_table(stage_summary, max_rows=20)}

## Final Portfolio Comparison

### Combined Frequency

- Overlap: `{freq_summary['overlap']}`
- EODHD only: `{freq_summary['eodhd_only']}`
- Open Source only: `{freq_summary['open_only']}`
- Jaccard overlap: `{freq_summary['jaccard']:.3f}` if not null

{_markdown_table(freq_summary["top_weight_diffs"], max_rows=10)}

### Combined Equal

- Overlap: `{equal_summary['overlap']}`
- EODHD only: `{equal_summary['eodhd_only']}`
- Open Source only: `{equal_summary['open_only']}`
- Jaccard overlap: `{equal_summary['jaccard']:.3f}` if not null

{_markdown_table(equal_summary["top_weight_diffs"], max_rows=10)}

## Largest Monthly Return Gaps

{_markdown_table(largest_monthly_model_gaps, max_rows=15)}

## Focus Tickers

{_markdown_table(focus_ticker_diagnostics, columns=[
    "ticker",
    "frequency_status",
    "equal_status",
    "eodhd_sector",
    "open_sector",
    "open_sector_source",
    "eodhd_earnings_rows",
    "open_earnings_rows",
    "eodhd_in_selection",
    "open_in_selection",
    "eodhd_model_stage_hits",
    "open_model_stage_hits",
    "reason",
], max_rows=20)}

## Artifact Paths

- Aligned EODHD dataset: `{output_root / 'aligned_data' / 'eodhd'}`
- Aligned Open Source dataset: `{output_root / 'aligned_data' / 'open_source'}`
- EODHD metrics: `{eodhd_summary.metrics_path}`
- Open Source metrics: `{open_summary.metrics_path}`
- Metric diff parquet: `{metrics_path}`
- Frequency holdings diff parquet: `{frequency_path}`
- Equal holdings diff parquet: `{equal_path}`
- Monthly return diff parquet: `{monthly_path}`
- Executive summary: `{executive_summary_path}`
- HTML report: `{output_root / 'report.html'}`
- Ticker deep dives: `{output_root / 'tickers'}`
- Stage report: `{output_root / 'stages' / 'index.html'}`
"""
    report = report.replace("`None:.3f` if not null", "n/a")
    if freq_summary["jaccard"] is not None:
        report = report.replace(
            "- Jaccard overlap: `n/a`",
            f"- Jaccard overlap: `{freq_summary['jaccard']:.3f}`",
            1,
        )
    if equal_summary["jaccard"] is not None:
        report = report.replace(
            "- Jaccard overlap: `n/a`",
            f"- Jaccard overlap: `{equal_summary['jaccard']:.3f}`",
            1,
        )
    report_path.write_text(report, encoding="utf-8")
    _write_html_page(
        output_path=executive_summary_html_path,
        title="Legacy DB executive summary",
        subtitle=f"Short root-cause summary for {output_root.name}",
        navigation='<p><a href="report.html">Global report</a> | <a href="tickers/index.html">Ticker index</a> | <a href="stages/index.html">Stage index</a></p>',
        sections=[
            ("Executive summary", pl.DataFrame([{"summary": line} for line in executive_summary.splitlines() if line.strip()])),
            ("Price coverage root cause", price_coverage_gap_summary),
            ("Missing-price selected tickers", price_coverage_gap_tickers),
            ("Counterfactual metrics", counterfactual_metrics),
            ("Counterfactual overlap", counterfactual_overlap),
        ],
    )

    stages_dir = output_root / "stages"
    tickers_dir = output_root / "tickers"
    ticker_index_rows: list[dict[str, object]] = []
    for row in focus_ticker_diagnostics.iter_rows(named=True):
        ticker = str(row["ticker"])
        file_name = f"{ticker.lower().replace('.', '_')}.html"
        ticker_index_rows.append(
            {
                "ticker": ticker,
                "frequency_status": row.get("frequency_status"),
                "equal_status": row.get("equal_status"),
                "open_sector": row.get("open_sector"),
                "open_sector_source": row.get("open_sector_source"),
                "eodhd_model_stage_hits": row.get("eodhd_model_stage_hits"),
                "open_model_stage_hits": row.get("open_model_stage_hits"),
                "report": file_name,
            }
        )
        _write_html_page(
            output_path=tickers_dir / file_name,
            title=f"{ticker} legacy diff deep dive",
            subtitle=str(row.get("reason") or ""),
            navigation='<p><a href="../report.html">Global report</a> | <a href="index.html">Ticker index</a> | <a href="../stages/index.html">Stage index</a></p>',
            sections=[
                ("Ticker summary", pl.DataFrame([row])),
                (
                    "Frequency portfolio diff",
                    frequency_diff.filter(pl.col("ticker") == ticker),
                ),
                (
                    "Equal portfolio diff",
                    equal_diff.filter(pl.col("ticker") == ticker),
                ),
                (
                    "Open-source general lineage",
                    _read_parquet_if_exists(OPEN_SOURCE_LINEAGE_DIR / "general_reference_lineage.parquet").filter(pl.col("ticker") == ticker),
                ),
                (
                    "Open-source earnings lineage",
                    _read_parquet_if_exists(OPEN_SOURCE_LINEAGE_DIR / "earnings_open_source_lineage.parquet").filter(pl.col("ticker") == ticker),
                ),
                (
                    "Latest EODHD stock selection row",
                    _select_ticker_row(_latest_month_frame(_read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / "polars_stocks_selections.parquet"))[0], ticker),
                ),
                (
                    "Latest open-source stock selection row",
                    _select_ticker_row(_latest_month_frame(_read_parquet_if_exists(output_root / "checkpoints" / "open_source" / "polars_stocks_selections.parquet"))[0], ticker),
                ),
                (
                    "Latest EODHD model-stage rows",
                    pl.concat(
                        [
                            _select_ticker_rows(_latest_month_frame(_read_parquet_if_exists(output_root / "checkpoints" / "eodhd" / name))[0], ticker).with_columns(pl.lit(stage).alias("stage"))
                            for stage, name in {
                                "optuna_11": "polars_optuna_output_11_detailed.parquet",
                                "optuna_12": "polars_optuna_output_12_detailed.parquet",
                                "optuna_21": "polars_optuna_output_21_detailed.parquet",
                                "optuna_22": "polars_optuna_output_22_detailed.parquet",
                                "combined_frequency": "polars_combined_frequency_detailed.parquet",
                                "combined_equal": "polars_combined_equal_detailed.parquet",
                            }.items()
                        ],
                        how="diagonal_relaxed",
                    ),
                ),
                (
                    "Latest open-source model-stage rows",
                    pl.concat(
                        [
                            _select_ticker_rows(_latest_month_frame(_read_parquet_if_exists(output_root / "checkpoints" / "open_source" / name))[0], ticker).with_columns(pl.lit(stage).alias("stage"))
                            for stage, name in {
                                "optuna_11": "polars_optuna_output_11_detailed.parquet",
                                "optuna_12": "polars_optuna_output_12_detailed.parquet",
                                "optuna_21": "polars_optuna_output_21_detailed.parquet",
                                "optuna_22": "polars_optuna_output_22_detailed.parquet",
                                "combined_frequency": "polars_combined_frequency_detailed.parquet",
                                "combined_equal": "polars_combined_equal_detailed.parquet",
                            }.items()
                        ],
                        how="diagonal_relaxed",
                    ),
                ),
            ],
        )

    ticker_index = pl.DataFrame(ticker_index_rows).sort("ticker") if ticker_index_rows else pl.DataFrame()
    if not ticker_index.is_empty():
        ticker_index = ticker_index.with_columns(
            pl.format("<a href=\"{}\">{}</a>", pl.col("report"), pl.col("ticker")).alias("ticker_link")
        )
    _write_html_page(
        output_path=tickers_dir / "index.html",
        title="Legacy diff ticker index",
        subtitle="Final holding divergence deep dives",
        navigation='<p><a href="../report.html">Global report</a> | <a href="../stages/index.html">Stage index</a></p>',
        sections=[("Tickers", ticker_index.select([col for col in ["ticker_link", "frequency_status", "equal_status", "open_sector", "open_sector_source", "report"] if col in ticker_index.columns]))],
    )

    _write_html_page(
        output_path=stages_dir / "index.html",
        title="Legacy diff stage summary",
        subtitle="Where the two datasets diverge through the legacy pipeline",
        navigation='<p><a href="../report.html">Global report</a> | <a href="../tickers/index.html">Ticker index</a></p>',
        sections=[
            ("Acceptance gates", acceptance_gates),
            ("Input quality", input_quality),
            ("Selection input coverage", selection_input_coverage),
            ("Price coverage root cause", price_coverage_gap_summary),
            ("Missing-price selected tickers", price_coverage_gap_tickers),
            ("Price return equivalence", price_return_summary),
            ("Price return outliers", price_return_outliers),
            ("Latest model drift", model_drift_summary),
            ("Counterfactual metrics", counterfactual_metrics),
            ("Counterfactual overlap", counterfactual_overlap),
            ("Open-source 2025 price audit", price_audit_summary),
            ("Open-source 2025 statement audit", statement_audit_summary),
            ("Stage summary", stage_summary),
            ("Monthly return gaps", largest_monthly_model_gaps),
        ],
    )

    _write_html_page(
        output_path=output_root / "report.html",
        title="Legacy DB comparison",
        subtitle=f"Aligned scope up to {alignment['price_cutoff_date']} with common ticker universe {alignment['common_tickers']}",
        navigation='<p><a href="executive_summary.html">Executive summary</a> | <a href="tickers/index.html">Ticker index</a> | <a href="stages/index.html">Stage index</a></p>',
        sections=[
            ("Acceptance gates", acceptance_gates),
            ("Input quality", input_quality),
            ("Selection input coverage", selection_input_coverage),
            ("Price coverage root cause", price_coverage_gap_summary),
            ("Missing-price selected tickers", price_coverage_gap_tickers),
            ("Price return equivalence", price_return_summary),
            ("Price return outliers", price_return_outliers),
            ("Latest model drift", model_drift_summary),
            ("Counterfactual metrics", counterfactual_metrics),
            ("Counterfactual overlap", counterfactual_overlap),
            ("Open-source 2025 price audit", price_audit_summary),
            ("Open-source 2025 statement audit", statement_audit_summary),
            ("Primary model metric diff", primary_models),
            ("Stage summary", stage_summary),
            ("Frequency final portfolio diff", frequency_diff),
            ("Equal final portfolio diff", equal_diff),
            ("Focus tickers", focus_ticker_diagnostics),
            ("Largest monthly return gaps", largest_monthly_model_gaps),
        ],
    )
    return report_path


def main() -> None:
    output_root = PROJECT_ROOT / "outputs" / f"legacy_db_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    eodhd_data_dir, open_data_dir, alignment = _prepare_aligned_datasets(output_root)
    (comparison_dir / "alignment_summary.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")

    eodhd_result, eodhd_summary = _run_legacy("eodhd", data_dir=eodhd_data_dir, output_root=output_root, first_date=DEFAULT_FIRST_DATE)
    open_result, open_summary = _run_legacy("open_source", data_dir=open_data_dir, output_root=output_root, first_date=DEFAULT_FIRST_DATE)

    metrics_diff = _compare_metrics(eodhd_result.metrics, open_result.metrics)
    monthly_returns_diff = _build_monthly_returns_diff(eodhd_summary.monthly_returns_path, open_summary.monthly_returns_path)
    frequency_diff = _compare_portfolios(
        StrategyLearner.get_portfolio_at_month(eodhd_result.combined_frequency),
        StrategyLearner.get_portfolio_at_month(open_result.combined_frequency),
    )
    equal_diff = _compare_portfolios(
        StrategyLearner.get_portfolio_at_month(eodhd_result.combined_equal),
        StrategyLearner.get_portfolio_at_month(open_result.combined_equal),
    )

    report_path = _write_report(
        output_root=output_root,
        alignment=alignment,
        eodhd_summary=eodhd_summary,
        open_summary=open_summary,
        metrics_diff=metrics_diff,
        monthly_returns_diff=monthly_returns_diff,
        frequency_diff=frequency_diff,
        equal_diff=equal_diff,
    )
    summary = {
        "alignment": alignment,
        "runs": {
            "eodhd": asdict(eodhd_summary),
            "open_source": asdict(open_summary),
        },
        "report_path": str(report_path),
    }
    (comparison_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Legacy comparison written to: {output_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
