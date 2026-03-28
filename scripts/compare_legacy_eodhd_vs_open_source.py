#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import polars as pl

from alpharank.strategy.legacy import StrategyLearner
from run_legacy import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EODHD_DIR = PROJECT_ROOT / "data" / "eodhd" / "output"
OPEN_SOURCE_DIR = PROJECT_ROOT / "data" / "open_source" / "output"
OPEN_SOURCE_PRICE_DIR = PROJECT_ROOT / "data" / "open_source" / "audit" / "price_transition_20050101"
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
    open_prices = _filter_by_tickers(_read_parquet(OPEN_SOURCE_PRICE_DIR / "US_Finalprice.parquet"), common_tickers)
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
        price_source_dir=OPEN_SOURCE_PRICE_DIR,
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

## Artifact Paths

- Aligned EODHD dataset: `{output_root / 'aligned_data' / 'eodhd'}`
- Aligned Open Source dataset: `{output_root / 'aligned_data' / 'open_source'}`
- EODHD metrics: `{eodhd_summary.metrics_path}`
- Open Source metrics: `{open_summary.metrics_path}`
- Metric diff parquet: `{metrics_path}`
- Frequency holdings diff parquet: `{frequency_path}`
- Equal holdings diff parquet: `{equal_path}`
- Monthly return diff parquet: `{monthly_path}`
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
