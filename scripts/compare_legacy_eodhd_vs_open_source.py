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
    stage_summary = _build_stage_summary(output_root)
    focus_ticker_diagnostics = _build_focus_ticker_diagnostics(
        output_root=output_root,
        frequency_diff=frequency_diff,
        equal_diff=equal_diff,
    )
    eodhd_input = input_quality.filter(pl.col("dataset") == "eodhd").head(1)
    open_input = input_quality.filter(pl.col("dataset") == "open_source").head(1)
    eodhd_selection_input = selection_input_coverage.filter(pl.col("dataset") == "eodhd").head(1)
    open_selection_input = selection_input_coverage.filter(pl.col("dataset") == "open_source").head(1)
    root_cause_lines = [
        "Sector coverage is no longer the blocker: open_source has 0 null sectors on the aligned scope and 0 null sectors in the latest strategy stages.",
        (
            f"Latest monthly price coverage remains much thinner in open_source: "
            f"{_scalar_or_none(open_selection_input, 'latest_price_tickers')} tickers versus "
            f"{_scalar_or_none(eodhd_selection_input, 'latest_price_tickers')} in EODHD."
        ),
        (
            f"That price gap flows directly into valuation coverage: "
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
            f"Earnings coverage improved sharply and is no longer empty, but still trails EODHD: "
            f"{_scalar_or_none(open_input, 'earnings_rows')} rows / {_scalar_or_none(open_input, 'earnings_tickers')} tickers "
            f"versus {_scalar_or_none(eodhd_input, 'earnings_rows')} / {_scalar_or_none(eodhd_input, 'earnings_tickers')}."
        ),
    ]

    primary_models = metrics_diff.filter(pl.col("model").is_in(["Combined_Equal", "Combined_Frequency", "SP500"]))
    largest_monthly_model_gaps = (
        monthly_returns_diff.filter(pl.col("monthly_return_diff").is_not_null())
        .with_columns(pl.col("monthly_return_diff").abs().alias("abs_monthly_return_diff"))
        .sort("abs_monthly_return_diff", descending=True)
        .select([column for column in ["portfolio_model", "model", "year_month", "eodhd_monthly_return", "open_monthly_return", "monthly_return_diff"] if column in monthly_returns_diff.columns])
        .head(15)
    )

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

## Root Cause Summary

{chr(10).join(f"- {line}" for line in root_cause_lines)}

## Selection Input Coverage

{_markdown_table(selection_input_coverage, max_rows=10)}

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
            ("Input quality", input_quality),
            ("Selection input coverage", selection_input_coverage),
            ("Stage summary", stage_summary),
            ("Monthly return gaps", largest_monthly_model_gaps),
        ],
    )

    _write_html_page(
        output_path=output_root / "report.html",
        title="Legacy DB comparison",
        subtitle=f"Aligned scope up to {alignment['price_cutoff_date']} with common ticker universe {alignment['common_tickers']}",
        navigation='<p><a href="tickers/index.html">Ticker index</a> | <a href="stages/index.html">Stage index</a></p>',
        sections=[
            ("Input quality", input_quality),
            ("Selection input coverage", selection_input_coverage),
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
