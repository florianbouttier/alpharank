# %%
import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.data.processing import FundamentalProcessor, IndexDataManager, PricesDataPreprocessor
from alpharank.features.indicators import TechnicalIndicators
from alpharank.strategy.legacy import ModelEvaluator, StrategyLearner
from alpharank.utils.frame_backend import normalize_year_month_to_period, to_pandas, to_polars
from alpharank.visualization.plotting import PortfolioVisualizer


@dataclass
class PipelineOutput:
    monthly_return: pl.DataFrame
    final_price_vs_index: pl.DataFrame
    stocks_selections: pl.DataFrame
    optuna_outputs: Dict[str, Dict[str, Any]]
    combined_equal: Dict[str, Any]
    combined_frequency: Dict[str, Any]
    metrics: Any
    artifacts: Dict[str, Path]


def _load_data(data_dir: Path) -> Dict[str, pl.DataFrame]:
    print("Loading data...")
    return {
        "final_price": pl.read_parquet(data_dir / "US_Finalprice.parquet"),
        "general": pl.read_parquet(data_dir / "US_General.parquet"),
        "income_statement": pl.read_parquet(data_dir / "US_Income_statement.parquet"),
        "balance_sheet": pl.read_parquet(data_dir / "US_Balance_sheet.parquet"),
        "cash_flow": pl.read_parquet(data_dir / "US_Cash_flow.parquet"),
        "earnings": pl.read_parquet(data_dir / "US_Earnings.parquet"),
        "us_historical_company": pl.read_csv(data_dir / "SP500_Constituents.csv", try_parse_dates=True),
        "sp500_price": pl.read_parquet(data_dir / "SP500Price.parquet"),
    }


def _write_checkpoint(df: Any, checkpoints_dir: Optional[Path], name: str) -> None:
    if checkpoints_dir is None:
        return
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoints_dir / f"{name}.parquet"
    to_polars(df).write_parquet(out)


def _save_html(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def _sort_monthly_frame(df: Any) -> pl.DataFrame:
    out = to_polars(df)
    sort_cols = [c for c in ["year_month", "model", "ticker"] if c in out.columns]
    if sort_cols:
        out = out.sort(sort_cols)
    return out


def _extract_start_year(first_date: str) -> int:
    try:
        return int(str(first_date).split("-")[0])
    except Exception as exc:
        raise ValueError(f"Invalid --first-date format: {first_date!r}. Expected YYYY-MM.") from exc


def run_pipeline(
    *,
    n_trials: int,
    n_jobs: int,
    first_date: str,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
) -> PipelineOutput:
    backend = "polars"
    project_root = Path(__file__).parent.parent
    data_dir = data_dir if data_dir is not None else project_root / "data"
    output_dir = output_dir if output_dir is not None else project_root / "outputs"
    os.chdir(data_dir)  # Keep legacy behaviour.

    payload = _load_data(data_dir)
    final_price = payload["final_price"]
    general = payload["general"]
    income_statement = payload["income_statement"]
    balance_sheet = payload["balance_sheet"]
    cash_flow = payload["cash_flow"]
    earnings = payload["earnings"]
    us_historical_company = payload["us_historical_company"]
    sp500_price = payload["sp500_price"]

    print(f"Preprocessing ({backend})...")
    ticker_to_exclude = ["SII.US", "CBE.US", "TIE.US"]
    final_price = final_price.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    general = general.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    income_statement = income_statement.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    balance_sheet = balance_sheet.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    cash_flow = cash_flow.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    earnings = earnings.filter(~pl.col("ticker").is_in(ticker_to_exclude))

    final_price = final_price.with_columns(
        pl.col("date").cast(pl.Date, strict=False).dt.truncate("1mo").alias("year_month")
    )
    us_historical_company = (
        us_historical_company
        .with_columns([
            pl.col("Ticker").cast(pl.Utf8).str.replace_all(r"\\.", "-").alias("ticker"),
            pl.col("Date").cast(pl.Date, strict=False).dt.truncate("1mo").alias("year_month"),
        ])
        .with_columns((pl.col("ticker") + pl.lit(".US")).alias("ticker"))
    )

    index_data = IndexDataManager(
        daily_prices_df=sp500_price.clone(),
        components_df=us_historical_company.clone(),
        backend=backend,
    )

    monthly_return = to_polars(
        PricesDataPreprocessor.calculate_monthly_returns(
            df=final_price.clone(),
            column_close="adjusted_close",
            column_date="date",
            backend=backend,
        )
    )
    _write_checkpoint(monthly_return, checkpoints_dir, f"{backend}_monthly_return")

    print("Calculating prices vs index...")
    sp500_price = sp500_price.rename({"close": "sp500_close"})
    final_price_vs_index = to_polars(
        PricesDataPreprocessor.prices_vs_index(
            index=sp500_price.clone(),
            prices=final_price.clone(),
            column_close_index="sp500_close",
            column_close_prices="adjusted_close",
            backend=backend,
        )
    )
    final_price_vs_index = to_polars(
        PricesDataPreprocessor.compute_dr(
            df=final_price_vs_index,
            column_date="date",
            column_close="adjusted_close",
            backend=backend,
        )
    )
    _write_checkpoint(
        final_price_vs_index.select(["ticker", "date", "close_vs_index", "dr_vs_index", "dr"]),
        checkpoints_dir,
        f"{backend}_final_price_vs_index",
    )

    print("Calculating ratios...")
    stocks_selections = to_polars(
        FundamentalProcessor.calculate_pe_ratios(
            balance=balance_sheet,
            earnings=earnings,
            cashflow=cash_flow,
            income=income_statement,
            earning_choice="netincome_rolling",
            monthly_return=to_pandas(monthly_return),
            list_date_to_maximise=["filing_date_income", "filing_date_balance"],
            backend=backend,
        )
    )
    _ = FundamentalProcessor.calculate_all_ratios(
        balance_sheet=balance_sheet,
        income_statement=income_statement,
        cash_flow=cash_flow,
        earnings=earnings,
        monthly_return=to_pandas(monthly_return),
        backend=backend,
    )

    stocks_selections = (
        stocks_selections
        .with_columns(pl.col("year_month").cast(pl.Date, strict=False))
        .filter(
            (pl.col("pe") < 100)
            & (pl.col("pe") > 0)
            & pl.col("pe").is_not_null()
            & pl.col("market_cap").is_not_null()
        )
        .join(
            us_historical_company.select(
                pl.col("year_month").cast(pl.Date, strict=False).alias("year_month"),
                pl.col("ticker"),
            ),
            how="inner",
            on=["ticker", "year_month"],
        )
    )
    _write_checkpoint(stocks_selections, checkpoints_dir, f"{backend}_stocks_selections")

    print("Running strategy learning (Optuna)...")
    prices_for_learning = to_pandas(final_price_vs_index)
    stocks_filter_for_learning = normalize_year_month_to_period(to_pandas(stocks_selections), col="year_month")
    sector_for_learning = to_pandas(general.select(["ticker", "Sector"]))

    optuna_output_1 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=2,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=42,
        backend=backend,
    )
    optuna_output_12 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=2,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=41,
        backend=backend,
    )
    optuna_output_21 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=1,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=42,
        backend=backend,
    )
    optuna_output_22 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=1,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=41,
        backend=backend,
    )
    optuna_outputs = {
        "11": optuna_output_1,
        "12": optuna_output_12,
        "21": optuna_output_21,
        "22": optuna_output_22,
    }
    for key, out in optuna_outputs.items():
        _write_checkpoint(out["aggregated"], checkpoints_dir, f"{backend}_optuna_output_{key}_aggregated")
        detail_key = "detailed" if "detailed" in out else "detailled"
        _write_checkpoint(out[detail_key], checkpoints_dir, f"{backend}_optuna_output_{key}_detailed")

    combined_equal = StrategyLearner.aggregate_portfolios(
        [optuna_output_1, optuna_output_12, optuna_output_21, optuna_output_22],
        mode="equal",
        index=index_data,
        backend=backend,
    )
    combined_frequency = StrategyLearner.aggregate_portfolios(
        [optuna_output_1, optuna_output_12, optuna_output_21, optuna_output_22],
        mode="frequency",
        index=index_data,
        backend=backend,
    )
    _write_checkpoint(combined_equal["aggregated"], checkpoints_dir, f"{backend}_combined_equal")
    _write_checkpoint(combined_frequency["aggregated"], checkpoints_dir, f"{backend}_combined_frequency")

    models = {
        "Legacy_Optuna_11": to_pandas(_sort_monthly_frame(optuna_output_1["aggregated"])),
        "Legacy_Optuna_12": to_pandas(_sort_monthly_frame(optuna_output_12["aggregated"])),
        "Legacy_Optuna_21": to_pandas(_sort_monthly_frame(optuna_output_21["aggregated"])),
        "Legacy_Optuna_22": to_pandas(_sort_monthly_frame(optuna_output_22["aggregated"])),
        "Combined_Equal": to_pandas(_sort_monthly_frame(combined_equal["aggregated"])),
        "Combined_Frequency": to_pandas(_sort_monthly_frame(combined_frequency["aggregated"])),
        "SP500": to_pandas(_sort_monthly_frame(index_data.monthly_returns)),
    }
    metrics, cumulative, correlation, worst_periods, drawdowns, annual_returns, cumulative_metrics, annual_metrics, monthly_returns = ModelEvaluator.compare_models(
        models,
        start_year=_extract_start_year(first_date),
    )
    _write_checkpoint(metrics.reset_index().rename(columns={"index": "model"}), checkpoints_dir, f"{backend}_metrics")

    print("Generating reports...")
    comparison_html = PortfolioVisualizer.make_comparison_report(
        metrics_df=metrics,
        cumulative_returns=cumulative,
        drawdowns_df=drawdowns,
        annual_returns_df=annual_returns,
        correlation_matrix=correlation,
        worst_periods_df=worst_periods,
        cumulative_metrics_dict=cumulative_metrics,
        annual_metrics_dict=annual_metrics,
        monthly_returns_dict=monthly_returns,
        title=f"Strategy Performance Comparison ({backend})",
    )
    is_test_run = n_trials < 30
    file_suffix = "_test" if is_test_run else datetime.now().strftime("%Y-%m-%d")
    comparison_file = output_dir / f"performance_of_models_{backend}{file_suffix}.html"
    _save_html(comparison_html, comparison_file)

    if "close" not in final_price.columns and "adjusted_close" in final_price.columns:
        final_price_long = final_price.rename({"adjusted_close": "close"})
    else:
        final_price_long = final_price

    current_portfolio_freq = StrategyLearner.get_portfolio_at_month(combined_frequency)
    report_html_freq = PortfolioVisualizer.make_portfolio_report(
        portfolio=current_portfolio_freq,
        title=f"Aggregated Portfolio (Frequency Weighted) - {backend}",
        price_data=to_pandas(final_price_long),
        balance_sheet=to_pandas(balance_sheet),
        income_statement=to_pandas(income_statement),
        cash_flow=to_pandas(cash_flow),
        earnings=to_pandas(earnings),
        backend="polars",
    )
    freq_file = output_dir / f"portfolio_report_frequency_{backend}{file_suffix}.html"
    _save_html(report_html_freq, freq_file)

    current_portfolio_equal = StrategyLearner.get_portfolio_at_month(combined_equal)
    report_html_equal = PortfolioVisualizer.make_portfolio_report(
        portfolio=current_portfolio_equal,
        title=f"Aggregated Portfolio (Equal Weighted) - {backend}",
        price_data=to_pandas(final_price_long),
        balance_sheet=to_pandas(balance_sheet),
        income_statement=to_pandas(income_statement),
        cash_flow=to_pandas(cash_flow),
        earnings=to_pandas(earnings),
        backend="polars",
    )
    equal_file = output_dir / f"portfolio_report_equal_{backend}{file_suffix}.html"
    _save_html(report_html_equal, equal_file)

    return PipelineOutput(
        monthly_return=monthly_return,
        final_price_vs_index=final_price_vs_index,
        stocks_selections=stocks_selections,
        optuna_outputs=optuna_outputs,
        combined_equal=combined_equal,
        combined_frequency=combined_frequency,
        metrics=metrics,
        artifacts={
            "comparison_html": comparison_file,
            "portfolio_frequency_html": freq_file,
            "portfolio_equal_html": equal_file,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy strategy runner (polars backend).")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--first-date", type=str, default="2010-01")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory containing legacy input files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Path where HTML outputs will be written.")
    parser.add_argument("--checkpoints-dir", type=str, default="outputs/checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    out = run_pipeline(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        first_date=args.first_date,
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
    )
    print("Artifacts:")
    for k, v in out.artifacts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
