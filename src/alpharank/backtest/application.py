from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd
import polars as pl

from alpharank.backtest.kpis import compute_backtest_kpis
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n
from alpharank.strategy.legacy import ModelEvaluator
from alpharank.visualization.plotting import PortfolioVisualizer


SelectionMode = Literal["top_n", "prediction_threshold"]


@dataclass(frozen=True)
class ApplicationBacktestConfig:
    name: str
    selection_mode: SelectionMode = "top_n"
    top_n: int | None = 20
    prediction_threshold: float | None = None
    max_price_staleness_months: int | None = None

    def __post_init__(self) -> None:
        if self.selection_mode == "top_n":
            if self.top_n is None or int(self.top_n) <= 0:
                raise ValueError("`top_n` must be a positive integer when selection_mode='top_n'.")
        elif self.selection_mode == "prediction_threshold":
            if self.prediction_threshold is None:
                raise ValueError(
                    "`prediction_threshold` must be provided when selection_mode='prediction_threshold'."
                )
        else:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode!r}")

        if self.max_price_staleness_months is not None and int(self.max_price_staleness_months) < 0:
            raise ValueError("`max_price_staleness_months` must be >= 0 when provided.")


@dataclass
class ApplicationBacktestResult:
    name: str
    config: ApplicationBacktestConfig
    eligible_predictions: pl.DataFrame
    selections: pl.DataFrame
    monthly_returns: pl.DataFrame
    kpis: pl.DataFrame


@dataclass
class BacktestComparisonResult:
    metrics: pd.DataFrame
    cumulative_returns: pd.DataFrame
    correlation_matrix: pd.DataFrame
    worst_periods: pd.DataFrame
    drawdowns: pd.DataFrame
    annual_returns: pd.DataFrame
    cumulative_metrics: Dict[str, pd.DataFrame]
    annual_metrics: Dict[str, pd.DataFrame]
    monthly_returns: Dict[str, pd.Series]
    html: str
    output_path: Path | None


def _month_index_expr(column: str) -> pl.Expr:
    return pl.col(column).dt.year() * pl.lit(12) + pl.col(column).dt.month()


def filter_predictions_by_price_staleness(
    predictions: pl.DataFrame,
    max_price_staleness_months: int | None,
) -> pl.DataFrame:
    if max_price_staleness_months is None or predictions.is_empty():
        return predictions

    required_cols = {"decision_month", "decision_asof_date"}
    missing = [col for col in required_cols if col not in predictions.columns]
    if missing:
        raise ValueError(
            "Cannot apply price staleness filter. Missing columns: " + ", ".join(sorted(missing))
        )

    staleness_limit = int(max_price_staleness_months)
    return (
        predictions.with_columns(
            (
                _month_index_expr("decision_month")
                - _month_index_expr("decision_asof_date")
            ).alias("_price_staleness_months")
        )
        .filter(
            pl.col("_price_staleness_months").is_not_null()
            & (pl.col("_price_staleness_months") <= pl.lit(staleness_limit))
        )
        .drop("_price_staleness_months")
    )


def select_predictions_above_threshold(predictions: pl.DataFrame, threshold: float) -> pl.DataFrame:
    if predictions.is_empty():
        return predictions.with_columns(pl.lit(None).alias("rank")).head(0)

    ranked = predictions.with_columns(
        pl.col("prediction").rank(method="ordinal", descending=True).over("year_month").alias("rank")
    )
    return ranked.filter(pl.col("prediction") > pl.lit(float(threshold))).sort(["year_month", "rank"])


def _complete_application_monthly_returns(
    eligible_predictions: pl.DataFrame,
    monthly_returns: pl.DataFrame,
) -> pl.DataFrame:
    if eligible_predictions.is_empty():
        return monthly_returns

    base_months = (
        eligible_predictions.group_by("holding_month")
        .agg(
            pl.col("decision_month").min().alias("decision_month"),
            pl.mean("benchmark_future_return").alias("benchmark_return"),
        )
        .sort("holding_month")
        .with_columns(pl.col("holding_month").alias("year_month"))
    )

    completed = base_months.join(
        monthly_returns.select(["holding_month", "portfolio_return", "hit_rate", "n_positions"])
        if not monthly_returns.is_empty()
        else pl.DataFrame(
            schema={
                "holding_month": pl.Date,
                "portfolio_return": pl.Float64,
                "hit_rate": pl.Float64,
                "n_positions": pl.Int64,
            }
        ),
        on="holding_month",
        how="left",
    ).with_columns(
        pl.col("portfolio_return").fill_null(0.0).alias("portfolio_return"),
        pl.col("benchmark_return").fill_null(0.0).alias("benchmark_return"),
        pl.col("hit_rate").fill_null(0.0).alias("hit_rate"),
        pl.col("n_positions").fill_null(0).cast(pl.Int64).alias("n_positions"),
    )

    return completed.with_columns(
        (pl.col("portfolio_return") - pl.col("benchmark_return")).alias("active_return")
    ).select(
        [
            "year_month",
            "decision_month",
            "holding_month",
            "portfolio_return",
            "benchmark_return",
            "active_return",
            "hit_rate",
            "n_positions",
        ]
    )


def run_application_backtest(
    predictions: pl.DataFrame,
    config: ApplicationBacktestConfig,
    *,
    risk_free_rate: float = 0.02,
) -> ApplicationBacktestResult:
    eligible_predictions = filter_predictions_by_price_staleness(
        predictions,
        max_price_staleness_months=config.max_price_staleness_months,
    )

    if config.selection_mode == "top_n":
        selections = select_top_n(eligible_predictions, top_n=int(config.top_n))
    else:
        selections = select_predictions_above_threshold(
            eligible_predictions,
            threshold=float(config.prediction_threshold),
        )

    monthly_returns = compute_monthly_portfolio_returns(
        selections.select(
            [
                "year_month",
                "decision_month",
                "holding_month",
                "future_return",
                "benchmark_future_return",
                "future_excess_return",
                "target_label",
                "prediction",
                "ticker",
            ]
        )
        if not selections.is_empty()
        else selections
    )
    monthly_returns = _complete_application_monthly_returns(eligible_predictions, monthly_returns)
    kpis = compute_backtest_kpis(monthly_returns=monthly_returns, risk_free_rate=risk_free_rate)

    return ApplicationBacktestResult(
        name=config.name,
        config=config,
        eligible_predictions=eligible_predictions,
        selections=selections,
        monthly_returns=monthly_returns,
        kpis=kpis,
    )


def _standardize_curve_frame(
    curve: ApplicationBacktestResult | pl.DataFrame | pd.DataFrame,
    *,
    return_column: str | None = None,
) -> pd.DataFrame:
    if isinstance(curve, ApplicationBacktestResult):
        frame = curve.monthly_returns
        default_return_column = "portfolio_return"
    else:
        frame = curve
        default_return_column = None

    if isinstance(frame, pl.DataFrame):
        pdf = frame.to_pandas()
    else:
        pdf = frame.copy()

    selected_return_column = return_column
    if selected_return_column is None:
        for candidate in [default_return_column, "portfolio_return", "monthly_return", "active_return"]:
            if candidate and candidate in pdf.columns:
                selected_return_column = candidate
                break
    if selected_return_column is None:
        raise ValueError("Could not infer return column for comparison.")

    if "year_month" not in pdf.columns:
        raise ValueError("Comparison curves must contain a `year_month` column.")

    out = pdf[["year_month", selected_return_column]].copy()
    out = out.rename(columns={selected_return_column: "monthly_return"})
    if "n_positions" in pdf.columns:
        out["n"] = pdf["n_positions"]
    return out


def compare_backtest_curves(
    curves: Dict[str, ApplicationBacktestResult | pl.DataFrame | pd.DataFrame],
    *,
    output_path: str | Path | None = None,
    title: str = "Backtest Strategy Comparison",
    start_year: int | None = None,
    end_year: int | None = None,
    risk_free_rate: float = 0.02,
    return_column: str | None = None,
) -> BacktestComparisonResult:
    models_data = {
        name: _standardize_curve_frame(curve, return_column=return_column)
        for name, curve in curves.items()
    }
    positions_dict = {}
    for name, frame in models_data.items():
        if "n" not in frame.columns or frame.empty:
            continue
        month_index = pd.to_datetime(frame["year_month"]).dt.to_period("M")
        positions_dict[name] = pd.Series(frame["n"].to_numpy(), index=month_index, name=name)
    empty_curves = [name for name, frame in models_data.items() if frame.empty]
    if empty_curves:
        raise ValueError(
            "Comparison curves have no monthly observations after filtering: " + ", ".join(sorted(empty_curves))
        )

    (
        metrics,
        cumulative,
        correlation,
        worst_periods,
        drawdowns,
        annual_returns,
        cumulative_metrics,
        annual_metrics,
        monthly_returns,
    ) = ModelEvaluator.compare_models(
        models_data,
        start_year=start_year,
        end_year=end_year,
        risk_free_rate=risk_free_rate,
    )

    html_report = PortfolioVisualizer.make_comparison_report(
        metrics_df=metrics,
        cumulative_returns=cumulative,
        drawdowns_df=drawdowns,
        annual_returns_df=annual_returns,
        correlation_matrix=correlation,
        worst_periods_df=worst_periods,
        cumulative_metrics_dict=cumulative_metrics,
        annual_metrics_dict=annual_metrics,
        monthly_returns_dict=monthly_returns,
        positions_dict=positions_dict,
        title=title,
    )

    resolved_output_path: Path | None = None
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(html_report, encoding="utf-8")

    return BacktestComparisonResult(
        metrics=metrics,
        cumulative_returns=cumulative,
        correlation_matrix=correlation,
        worst_periods=worst_periods,
        drawdowns=drawdowns,
        annual_returns=annual_returns,
        cumulative_metrics=cumulative_metrics,
        annual_metrics=annual_metrics,
        monthly_returns=monthly_returns,
        html=html_report,
        output_path=resolved_output_path,
    )
