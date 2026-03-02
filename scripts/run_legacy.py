# %%
import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.data.processing import FundamentalProcessor, IndexDataManager, PricesDataPreprocessor
from alpharank.features.indicators import TechnicalIndicators
from alpharank.strategy.legacy import ModelEvaluator, StrategyLearner
from alpharank.visualization.plotting import PortfolioVisualizer


@dataclass
class PipelineOutput:
    monthly_return: pd.DataFrame
    final_price_vs_index: pd.DataFrame
    stocks_selections: pd.DataFrame
    optuna_outputs: Dict[str, Dict[str, pd.DataFrame]]
    combined_equal: Dict[str, pd.DataFrame]
    combined_frequency: Dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    artifacts: Dict[str, Path]


def _load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    # Keep pandas boundaries for compatibility.
    print("Loading data...")
    return {
        "final_price": pd.read_parquet(data_dir / "US_Finalprice.parquet"),
        "general": pd.read_parquet(data_dir / "US_General.parquet"),
        "income_statement": pd.read_parquet(data_dir / "US_Income_statement.parquet"),
        "balance_sheet": pd.read_parquet(data_dir / "US_Balance_sheet.parquet"),
        "cash_flow": pd.read_parquet(data_dir / "US_Cash_flow.parquet"),
        "earnings": pd.read_parquet(data_dir / "US_Earnings.parquet"),
        "us_historical_company": pd.read_csv(data_dir / "SP500_Constituents.csv"),
        "sp500_price": pd.read_parquet(data_dir / "SP500Price.parquet"),
    }


def _write_checkpoint(df: pd.DataFrame, checkpoints_dir: Optional[Path], name: str) -> None:
    if checkpoints_dir is None:
        return
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoints_dir / f"{name}.parquet"
    df.to_parquet(out, index=False)


def _save_html(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def _sort_monthly_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sort_cols = [c for c in ["year_month", "model", "ticker"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def run_pipeline(
    *,
    n_trials: int,
    n_jobs: int,
    first_date: pd.Period,
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
    for ticker in ticker_to_exclude:
        final_price = final_price[final_price["ticker"] != ticker]
        general = general[general["ticker"] != ticker]
        income_statement = income_statement[income_statement["ticker"] != ticker]
        balance_sheet = balance_sheet[balance_sheet["ticker"] != ticker]
        cash_flow = cash_flow[cash_flow["ticker"] != ticker]
        earnings = earnings[earnings["ticker"] != ticker]

    final_price["year_month"] = pd.to_datetime(final_price["date"]).dt.to_period("M")
    us_historical_company["ticker"] = us_historical_company["Ticker"].apply(
        lambda x: re.sub(r"\.", "-", x) if isinstance(x, str) else x
    )
    us_historical_company["ticker"] = us_historical_company["ticker"] + ".US"
    us_historical_company["year_month"] = pd.to_datetime(us_historical_company["Date"]).dt.to_period("M")

    index_data = IndexDataManager(
        daily_prices_df=sp500_price.copy(),
        components_df=us_historical_company.copy(),
        backend=backend,
    )

    monthly_return = PricesDataPreprocessor.calculate_monthly_returns(
        df=final_price.copy(),
        column_close="adjusted_close",
        column_date="date",
        backend=backend,
    )
    _write_checkpoint(monthly_return, checkpoints_dir, f"{backend}_monthly_return")

    print("Calculating prices vs index...")
    sp500_price = sp500_price.rename(columns={"close": "sp500_close"})
    final_price_vs_index = PricesDataPreprocessor.prices_vs_index(
        index=sp500_price.copy(),
        prices=final_price.copy(),
        column_close_index="sp500_close",
        column_close_prices="adjusted_close",
        backend=backend,
    )
    final_price_vs_index = PricesDataPreprocessor.compute_dr(
        df=final_price_vs_index,
        column_date="date",
        column_close="adjusted_close",
        backend=backend,
    )
    _write_checkpoint(
        final_price_vs_index[
            ["ticker", "date", "close_vs_index", "dr_vs_index", "dr"]
        ].copy(),
        checkpoints_dir,
        f"{backend}_final_price_vs_index",
    )

    print("Calculating ratios...")
    stocks_selections = FundamentalProcessor.calculate_pe_ratios(
        balance=balance_sheet,
        earnings=earnings,
        cashflow=cash_flow,
        income=income_statement,
        earning_choice="netincome_rolling",
        monthly_return=monthly_return.copy(),
        list_date_to_maximise=["filing_date_income", "filing_date_balance"],
        backend=backend,
    )
    _ = FundamentalProcessor.calculate_all_ratios(
        balance_sheet=balance_sheet.copy(),
        income_statement=income_statement.copy(),
        cash_flow=cash_flow.copy(),
        earnings=earnings.copy(),
        monthly_return=monthly_return.copy(),
        backend=backend,
    )

    stocks_selections = (
        stocks_selections[(stocks_selections["pe"] < 100) & (stocks_selections["pe"] > 0)]
        .dropna(subset=["pe", "market_cap"])
        .merge(
            us_historical_company[["year_month", "ticker"]],
            how="inner",
            left_on=["ticker", "year_month"],
            right_on=["ticker", "year_month"],
        )
    )
    _write_checkpoint(stocks_selections, checkpoints_dir, f"{backend}_stocks_selections")

    print("Running strategy learning (Optuna)...")
    optuna_output_1 = StrategyLearner.learning_process_optuna_full(
        prices=final_price_vs_index.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_selections.copy(),
        sector=general[["ticker", "Sector"]].copy(),
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
        prices=final_price_vs_index.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_selections.copy(),
        sector=general[["ticker", "Sector"]].copy(),
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
        prices=final_price_vs_index.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_selections.copy(),
        sector=general[["ticker", "Sector"]].copy(),
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
        prices=final_price_vs_index.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_selections.copy(),
        sector=general[["ticker", "Sector"]].copy(),
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
        "Legacy_Optuna_11": _sort_monthly_frame(optuna_output_1["aggregated"]),
        "Legacy_Optuna_12": _sort_monthly_frame(optuna_output_12["aggregated"]),
        "Legacy_Optuna_21": _sort_monthly_frame(optuna_output_21["aggregated"]),
        "Legacy_Optuna_22": _sort_monthly_frame(optuna_output_22["aggregated"]),
        "Combined_Equal": _sort_monthly_frame(combined_equal["aggregated"]),
        "Combined_Frequency": _sort_monthly_frame(combined_frequency["aggregated"]),
        "SP500": _sort_monthly_frame(index_data.monthly_returns),
    }
    metrics, cumulative, correlation, worst_periods, drawdowns, annual_returns, cumulative_metrics, annual_metrics, monthly_returns = ModelEvaluator.compare_models(
        models, start_year=first_date.year
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
        final_price_long = final_price.rename(columns={"adjusted_close": "close"})
    else:
        final_price_long = final_price

    current_portfolio_freq = StrategyLearner.get_portfolio_at_month(combined_frequency)
    report_html_freq = PortfolioVisualizer.make_portfolio_report(
        portfolio=current_portfolio_freq,
        title=f"Aggregated Portfolio (Frequency Weighted) - {backend}",
        price_data=final_price_long,
        balance_sheet=balance_sheet,
        income_statement=income_statement,
        cash_flow=cash_flow,
        earnings=earnings,
        backend="pandas",
    )
    freq_file = output_dir / f"portfolio_report_frequency_{backend}{file_suffix}.html"
    _save_html(report_html_freq, freq_file)

    current_portfolio_equal = StrategyLearner.get_portfolio_at_month(combined_equal)
    # Optional historical snapshot if available in the run window.
    try:
        _ = StrategyLearner.get_portfolio_at_month(combined_equal, month=pd.Period("2026-01", freq="M"))
    except Exception:
        pass
    report_html_equal = PortfolioVisualizer.make_portfolio_report(
        portfolio=current_portfolio_equal,
        title=f"Aggregated Portfolio (Equal Weighted) - {backend}",
        price_data=final_price_long,
        balance_sheet=balance_sheet,
        income_statement=income_statement,
        cash_flow=cash_flow,
        earnings=earnings,
        backend="pandas",
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
    first_date = pd.Period(args.first_date, freq="M")
    checkpoints_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    out = run_pipeline(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        first_date=first_date,
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
    )
    print("Artifacts:")
    for k, v in out.artifacts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
