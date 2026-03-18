from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from alpharank.backtest.kpis import compute_backtest_kpis
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n


SelectionMode = Literal["top_n", "prediction_threshold"]

TARGET_COMPARISON_METRICS = (
    "CAGR",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown",
    "Annualized Volatility",
)

PERCENT_METRICS = {
    "Total Return",
    "CAGR",
    "CAGR (3Y)",
    "CAGR (5Y)",
    "CAGR (10Y)",
    "Monthly Mean",
    "Monthly Volatility",
    "Annualized Volatility",
    "Max Drawdown",
    "Positive Periods %",
}

FLOAT_METRICS = {
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Number of Stocks (Avg)",
}


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
    metrics: pl.DataFrame
    cumulative_returns: pl.DataFrame
    correlation_matrix: pl.DataFrame
    worst_periods: pl.DataFrame
    drawdowns: pl.DataFrame
    annual_returns: pl.DataFrame
    cumulative_metrics: Dict[str, pl.DataFrame]
    annual_metrics: Dict[str, pl.DataFrame]
    monthly_returns: Dict[str, pl.DataFrame]
    html: str
    output_path: Path | None


def _month_index_expr(column: str) -> pl.Expr:
    return pl.col(column).dt.year() * pl.lit(12) + pl.col(column).dt.month()


def _normalize_year_month_expr(column: str) -> pl.Expr:
    utf8_col = pl.col(column).cast(pl.Utf8, strict=False)
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Date, strict=False),
            pl.col(column).cast(pl.Datetime, strict=False).cast(pl.Date),
            utf8_col.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            utf8_col.str.strptime(pl.Date, "%Y-%m", strict=False),
        ]
    )


def _format_month_label(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)[:7]


def _format_percent(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2%}"


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _format_int(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    try:
        if not np.isfinite(float(value)):
            return "N/A"
    except Exception:
        return "N/A"
    return str(int(round(float(value))))


def _plot_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


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
    curve: ApplicationBacktestResult | pl.DataFrame,
    *,
    return_column: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pl.DataFrame:
    if isinstance(curve, ApplicationBacktestResult):
        frame = curve.monthly_returns
        default_return_column = "portfolio_return"
    elif isinstance(curve, pl.DataFrame):
        frame = curve
        default_return_column = None
    else:
        raise TypeError("Comparison curves must be polars DataFrames or ApplicationBacktestResult objects.")

    selected_return_column = return_column
    if selected_return_column is None:
        for candidate in [default_return_column, "portfolio_return", "monthly_return", "active_return"]:
            if candidate and candidate in frame.columns:
                selected_return_column = candidate
                break
    if selected_return_column is None:
        raise ValueError("Could not infer return column for comparison.")
    if "year_month" not in frame.columns:
        raise ValueError("Comparison curves must contain a `year_month` column.")

    n_column = "n_positions" if "n_positions" in frame.columns else ("n" if "n" in frame.columns else None)

    standardized = frame.with_columns(
        _normalize_year_month_expr("year_month").alias("year_month"),
        pl.col(selected_return_column).cast(pl.Float64, strict=False).alias("monthly_return"),
        pl.col(n_column).cast(pl.Float64, strict=False).alias("n") if n_column else pl.lit(None, dtype=pl.Float64).alias("n"),
    ).select(["year_month", "monthly_return", "n"])

    standardized = standardized.drop_nulls(["year_month", "monthly_return"])
    if start_year is not None:
        standardized = standardized.filter(pl.col("year_month").dt.year() >= pl.lit(int(start_year)))
    if end_year is not None:
        standardized = standardized.filter(pl.col("year_month").dt.year() <= pl.lit(int(end_year)))

    standardized = standardized.group_by("year_month").agg(
        pl.mean("monthly_return").alias("monthly_return"),
        pl.mean("n").alias("n"),
    ).sort("year_month")
    return standardized


def _series_metrics(values: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> Dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {}

    n_periods = int(clean.size)
    total_years = max(n_periods / periods_per_year, 1 / periods_per_year)

    total_return = float(np.prod(1.0 + clean) - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / total_years) - 1.0)
    monthly_mean = float(np.mean(clean))
    monthly_vol = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    annualized_vol = float(monthly_vol * np.sqrt(periods_per_year))

    sharpe = np.nan
    if annualized_vol > 0:
        sharpe = float((cagr - risk_free_rate) / annualized_vol)

    downside = clean[clean < 0]
    downside_std = float(np.std(downside, ddof=1) * np.sqrt(periods_per_year)) if downside.size > 1 else 0.0
    sortino = np.nan
    if downside_std > 0:
        sortino = float((cagr - risk_free_rate) / downside_std)

    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    max_drawdown = float(np.min(drawdowns))
    calmar = np.nan
    if max_drawdown != 0:
        calmar = float(cagr / abs(max_drawdown))

    max_dd_duration = 0
    current_duration = 0
    for value in drawdowns:
        if value < 0:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Monthly Mean": monthly_mean,
        "Monthly Volatility": monthly_vol,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Max DD Duration": float(max_dd_duration),
        "Positive Periods %": float(np.mean(clean > 0)),
    }


def _window_cagr(values: np.ndarray, years: int) -> float | None:
    required_periods = 12 * years
    if values.size < required_periods:
        return None
    subset = values[-required_periods:]
    return float(np.prod(1.0 + subset) ** (1.0 / years) - 1.0)


def _build_metrics_table(
    models_data: Dict[str, pl.DataFrame],
    *,
    risk_free_rate: float,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, frame in models_data.items():
        returns = np.asarray(frame.get_column("monthly_return").to_list(), dtype=float)
        metrics = _series_metrics(returns, risk_free_rate=risk_free_rate)
        if not metrics:
            continue

        start_date = _format_month_label(frame.get_column("year_month").min())
        end_date = _format_month_label(frame.get_column("year_month").max())
        avg_positions: float | None = None
        if "n" in frame.columns and frame.height > 0:
            mean_positions = frame.get_column("n").mean()
            avg_positions = float(mean_positions) if mean_positions is not None else None

        row: dict[str, Any] = {
            "model": model_name,
            "Start Date": start_date,
            "End Date": end_date,
            "Number of Stocks (Avg)": avg_positions,
            **metrics,
            "CAGR (3Y)": _window_cagr(returns, 3),
            "CAGR (5Y)": _window_cagr(returns, 5),
            "CAGR (10Y)": _window_cagr(returns, 10),
        }
        rows.append(row)

    if not rows:
        return pl.DataFrame({"model": []})

    return pl.DataFrame(rows).sort("model")


def _aligned_returns_frame(models_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    calendar = (
        pl.concat([frame.select("year_month") for frame in models_data.values()], how="vertical")
        .unique()
        .sort("year_month")
    )

    aligned = calendar
    for model_name, frame in models_data.items():
        aligned = aligned.join(
            frame.select("year_month", pl.col("monthly_return").alias(model_name)),
            on="year_month",
            how="left",
        )
    return aligned.sort("year_month")


def _compute_cumulative_returns(aligned_returns: pl.DataFrame) -> pl.DataFrame:
    model_columns = [col for col in aligned_returns.columns if col != "year_month"]
    returns_matrix = (
        aligned_returns.select(model_columns)
        .with_columns(pl.all().fill_null(0.0))
        .to_numpy()
        .astype(float)
    )
    cumulative_matrix = np.cumprod(1.0 + returns_matrix, axis=0)
    return pl.DataFrame(
        {
            "year_month": aligned_returns.get_column("year_month").to_list(),
            **{model_columns[idx]: cumulative_matrix[:, idx].tolist() for idx in range(len(model_columns))},
        }
    )


def _compute_drawdowns(cumulative_returns: pl.DataFrame) -> pl.DataFrame:
    model_columns = [col for col in cumulative_returns.columns if col != "year_month"]
    cumulative_matrix = cumulative_returns.select(model_columns).to_numpy().astype(float)
    peaks = np.maximum.accumulate(cumulative_matrix, axis=0)
    drawdowns = cumulative_matrix / peaks - 1.0
    return pl.DataFrame(
        {
            "year_month": cumulative_returns.get_column("year_month").to_list(),
            **{model_columns[idx]: drawdowns[:, idx].tolist() for idx in range(len(model_columns))},
        }
    )


def _compute_annual_returns(models_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    annual_frames: list[pl.DataFrame] = []
    for model_name, frame in models_data.items():
        annual = frame.group_by(pl.col("year_month").dt.year().alias("year")).agg(
            ((pl.col("monthly_return") + 1.0).product() - 1.0).alias(model_name)
        )
        annual_frames.append(annual.sort("year"))

    if not annual_frames:
        return pl.DataFrame({"year": []})

    result = annual_frames[0]
    for frame in annual_frames[1:]:
        result = result.join(frame, on="year", how="full", coalesce=True)
    return result.sort("year")


def _compute_correlation_matrix(aligned_returns: pl.DataFrame) -> pl.DataFrame:
    model_columns = [col for col in aligned_returns.columns if col != "year_month"]
    rows: list[dict[str, Any]] = []
    columns_data = {
        name: np.asarray(aligned_returns.get_column(name).to_list(), dtype=float)
        for name in model_columns
    }
    for left_name in model_columns:
        row: dict[str, Any] = {"model": left_name}
        left_values = columns_data[left_name]
        for right_name in model_columns:
            right_values = columns_data[right_name]
            valid_mask = np.isfinite(left_values) & np.isfinite(right_values)
            if valid_mask.sum() < 2:
                row[right_name] = None
            elif left_name == right_name:
                row[right_name] = 1.0
            else:
                row[right_name] = float(np.corrcoef(left_values[valid_mask], right_values[valid_mask])[0, 1])
        rows.append(row)
    return pl.DataFrame(rows)


def _metric_grids(
    models_data: Dict[str, pl.DataFrame],
    *,
    risk_free_rate: float,
    exact_year: bool,
) -> Dict[str, pl.DataFrame]:
    years = sorted(
        {
            int(year)
            for frame in models_data.values()
            for year in frame.get_column("year_month").dt.year().to_list()
        }
    )
    grids: Dict[str, list[dict[str, Any]]] = {metric: [] for metric in TARGET_COMPARISON_METRICS}

    for year in years:
        per_metric_row: dict[str, dict[str, Any]] = {
            metric: {"year": year}
            for metric in TARGET_COMPARISON_METRICS
        }
        for model_name, frame in models_data.items():
            if exact_year:
                subset = frame.filter(pl.col("year_month").dt.year() == pl.lit(year))
            else:
                subset = frame.filter(pl.col("year_month").dt.year() >= pl.lit(year))
            returns = np.asarray(subset.get_column("monthly_return").to_list(), dtype=float)
            if returns.size < 3:
                for metric in TARGET_COMPARISON_METRICS:
                    per_metric_row[metric][model_name] = None
                continue
            metrics = _series_metrics(returns, risk_free_rate=risk_free_rate)
            for metric in TARGET_COMPARISON_METRICS:
                per_metric_row[metric][model_name] = metrics.get(metric)
        for metric in TARGET_COMPARISON_METRICS:
            grids[metric].append(per_metric_row[metric])

    return {metric: pl.DataFrame(rows).sort("year") if rows else pl.DataFrame({"year": []}) for metric, rows in grids.items()}


def _compute_worst_periods(models_data: Dict[str, pl.DataFrame], annual_returns: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, str]] = []
    annual_lookup = {
        model_name: annual_returns.select(["year", model_name]).drop_nulls(model_name)
        for model_name in annual_returns.columns
        if model_name != "year"
    }

    for model_name, frame in models_data.items():
        worst_month_row = frame.sort("monthly_return").row(0, named=True) if frame.height > 0 else None
        worst_month = "N/A"
        if worst_month_row is not None:
            worst_month = f"{_format_month_label(worst_month_row['year_month'])}: {_format_percent(float(worst_month_row['monthly_return']))}"

        worst_year = "N/A"
        annual_frame = annual_lookup.get(model_name)
        if annual_frame is not None and annual_frame.height > 0:
            annual_row = annual_frame.sort(model_name).row(0, named=True)
            worst_year = f"{int(annual_row['year'])}: {_format_percent(float(annual_row[model_name]))}"

        rows.append({"model": model_name, "Worst Month": worst_month, "Worst Year": worst_year})

    return pl.DataFrame(rows).sort("model") if rows else pl.DataFrame({"model": [], "Worst Month": [], "Worst Year": []})


def _display_metrics_table(metrics: pl.DataFrame) -> pl.DataFrame:
    if metrics.is_empty():
        return metrics

    rows: list[dict[str, str]] = []
    for row in metrics.to_dicts():
        display_row: dict[str, str] = {}
        for column, value in row.items():
            if column in {"model", "Start Date", "End Date"}:
                display_row[column] = "N/A" if value is None else str(value)
            elif column in PERCENT_METRICS:
                display_row[column] = _format_percent(None if value is None else float(value))
            elif column in FLOAT_METRICS:
                display_row[column] = _format_float(None if value is None else float(value))
            elif column == "Max DD Duration":
                display_row[column] = _format_int(value)
            else:
                display_row[column] = "N/A" if value is None else str(value)
        rows.append(display_row)
    return pl.DataFrame(rows)


def _table_html(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "<div class='text-muted'>No data available.</div>"

    header_html = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    body_rows = []
    for row in frame.to_dicts():
        cells = "".join(f"<td>{escape('N/A' if value is None else str(value))}</td>" for value in row.values())
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<div class='table-responsive'>"
        "<table class='table table-striped table-sm'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _plot_cumulative_returns(cumulative_returns: pl.DataFrame, date_range_str: str) -> str:
    fig = go.Figure()
    for column in cumulative_returns.columns:
        if column == "year_month":
            continue
        series = cumulative_returns.select(["year_month", column]).drop_nulls(column)
        if series.is_empty():
            continue
        values = np.asarray(series.get_column(column).to_list(), dtype=float)
        rebased = np.round((values / values[0]) * 100.0, 2)
        fig.add_trace(
            go.Scatter(
                x=series.get_column("year_month").to_list(),
                y=rebased.tolist(),
                mode="lines",
                name=column,
            )
        )
    fig.update_layout(
        title=f"Cumulative Returns (Log Scale) - Rebased to 100<br><sup>{date_range_str}</sup>",
        yaxis_type="log",
        yaxis_title="Value (Start=100)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )
    return _plot_html(fig)


def _plot_drawdowns(drawdowns: pl.DataFrame, date_range_str: str) -> str:
    fig = go.Figure()
    for column in drawdowns.columns:
        if column == "year_month":
            continue
        fig.add_trace(
            go.Scatter(
                x=drawdowns.get_column("year_month").to_list(),
                y=drawdowns.get_column(column).to_list(),
                mode="lines",
                name=column,
                fill="tozeroy",
            )
        )
    fig.update_layout(
        title=f"Drawdown Analysis<br><sup>{date_range_str}</sup>",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    return _plot_html(fig)


def _plot_positions(positions_dict: Dict[str, pl.DataFrame], date_range_str: str) -> str:
    fig = go.Figure()
    for model_name, frame in positions_dict.items():
        if frame.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=frame.get_column("year_month").to_list(),
                y=frame.get_column("n").to_list(),
                mode="lines",
                name=model_name,
            )
        )
    fig.update_layout(
        title=f"Number of Positions by Date<br><sup>{date_range_str}</sup>",
        yaxis_title="Positions",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    return _plot_html(fig)


def _plot_metric_heatmap(metric: str, cumulative_grid: pl.DataFrame, annual_grid: pl.DataFrame) -> str:
    models = [column for column in cumulative_grid.columns if column != "year"]
    if not models:
        return "<div class='text-muted'>No KPI data available.</div>"

    cumulative_years = cumulative_grid.get_column("year").to_list()
    annual_years = annual_grid.get_column("year").to_list()
    cumulative_matrix = np.asarray([cumulative_grid.get_column(model).to_list() for model in models], dtype=float)
    annual_matrix = np.asarray([annual_grid.get_column(model).to_list() for model in models], dtype=float)

    combined = np.concatenate([cumulative_matrix.flatten(), annual_matrix.flatten()])
    finite_values = combined[np.isfinite(combined)]
    zmin = float(finite_values.min()) if finite_values.size else None
    zmax = float(finite_values.max()) if finite_values.size else None

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{metric}: Cumulative from Start Year (Consistency)",
            f"{metric}: Discrete Annual Performance",
        ),
        vertical_spacing=0.16,
    )

    is_percent = metric not in {"Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"}
    texttemplate = "%{z:.1%}" if is_percent else "%{z:.2f}"
    colorscale = "Viridis" if metric == "Annualized Volatility" else "RdYlGn"

    fig.add_trace(
        go.Heatmap(
            z=cumulative_matrix,
            x=cumulative_years,
            y=models,
            texttemplate=texttemplate,
            textfont={"size": 11},
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=annual_matrix,
            x=annual_years,
            y=models,
            texttemplate=texttemplate,
            textfont={"size": 11},
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_xaxes(type="category", row=2, col=1)
    fig.update_layout(height=1050, title_text=f"Deep Dive: {metric}", template="plotly_white")
    return _plot_html(fig)


def _plot_monthly_returns_heatmap(model_name: str, monthly_returns: pl.DataFrame) -> str:
    frame = monthly_returns.with_columns(
        pl.col("year_month").dt.year().alias("year"),
        pl.col("year_month").dt.month().alias("month"),
    )
    years = sorted(frame.get_column("year").unique().to_list())
    year_to_index = {year: idx for idx, year in enumerate(years)}
    matrix = np.full((len(years), 12), np.nan)
    for row in frame.select(["year", "month", "monthly_return"]).to_dicts():
        matrix[year_to_index[int(row["year"])]][int(row["month"]) - 1] = float(row["monthly_return"])

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=years,
            texttemplate="%{z:.1%}",
            textfont={"size": 10},
            colorscale="RdYlGn",
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        title=f"{model_name} - Monthly Returns",
        height=max(400, len(years) * 40),
        template="plotly_white",
        yaxis=dict(dtick=1, type="category"),
    )
    return _plot_html(fig)


def _plot_risk_reward(metrics: pl.DataFrame, date_range_str: str) -> str:
    plot_rows = []
    for row in metrics.to_dicts():
        cagr = row.get("CAGR")
        vol = row.get("Annualized Volatility")
        if cagr is None or vol is None:
            continue
        if not np.isfinite(float(cagr)) or not np.isfinite(float(vol)):
            continue
        plot_rows.append(row)

    if not plot_rows:
        return "<div class='text-muted'>No risk/reward data available.</div>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[float(row["Annualized Volatility"]) for row in plot_rows],
            y=[float(row["CAGR"]) for row in plot_rows],
            mode="markers+text",
            text=[str(row["model"]) for row in plot_rows],
            textposition="top center",
            marker={"size": 12},
        )
    )
    fig.update_layout(
        title=f"Risk-Reward<br><sup>{date_range_str}</sup>",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=400,
    )
    return _plot_html(fig)


def _plot_correlation_matrix(correlation_matrix: pl.DataFrame, date_range_str: str) -> str:
    models = correlation_matrix.get_column("model").to_list()
    z = correlation_matrix.drop("model").to_numpy().astype(float)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=models,
            y=models,
            texttemplate="%{z:.2f}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        title=f"Correlation Matrix<br><sup>{date_range_str}</sup>",
        template="plotly_white",
        height=500,
    )
    return _plot_html(fig)


def _build_comparison_report(
    *,
    title: str,
    metrics: pl.DataFrame,
    cumulative_returns: pl.DataFrame,
    drawdowns: pl.DataFrame,
    annual_returns: pl.DataFrame,
    correlation_matrix: pl.DataFrame,
    worst_periods: pl.DataFrame,
    cumulative_metrics: Dict[str, pl.DataFrame],
    annual_metrics: Dict[str, pl.DataFrame],
    monthly_returns: Dict[str, pl.DataFrame],
    positions_dict: Dict[str, pl.DataFrame],
) -> str:
    try:
        start_date = _format_month_label(cumulative_returns.get_column("year_month").min())
        end_date = _format_month_label(cumulative_returns.get_column("year_month").max())
        date_range_str = f"Period: {start_date} to {end_date}"
    except Exception:
        date_range_str = "Period: N/A"

    metrics_html = _table_html(_display_metrics_table(metrics))
    worst_periods_html = _table_html(worst_periods)
    cumulative_html = _plot_cumulative_returns(cumulative_returns, date_range_str)
    drawdowns_html = _plot_drawdowns(drawdowns, date_range_str)
    positions_html = _plot_positions(positions_dict, date_range_str) if positions_dict else ""
    risk_reward_html = _plot_risk_reward(metrics, date_range_str)
    correlation_html = _plot_correlation_matrix(correlation_matrix, date_range_str)

    kpi_nav = ""
    kpi_content = ""
    first = True
    for metric in TARGET_COMPARISON_METRICS:
        cumulative_grid = cumulative_metrics.get(metric)
        annual_grid = annual_metrics.get(metric)
        if cumulative_grid is None or annual_grid is None:
            continue
        slug = "".join(character for character in metric if character.isalnum())
        active_class = "active" if first else ""
        show_class = "show active" if first else ""
        kpi_nav += f'<li class="nav-item"><a class="nav-link {active_class}" data-toggle="tab" href="#kpi-{slug}">{metric}</a></li>'
        kpi_content += (
            f'<div id="kpi-{slug}" class="tab-pane fade {show_class}">'
            f'<div class="chart-container">{_plot_metric_heatmap(metric, cumulative_grid, annual_grid)}</div>'
            "</div>"
        )
        first = False

    monthly_nav = ""
    monthly_content = ""
    first = True
    for model_name, frame in monthly_returns.items():
        if frame.is_empty():
            continue
        slug = "".join(character for character in model_name if character.isalnum())
        active_class = "active" if first else ""
        show_class = "show active" if first else ""
        monthly_nav += f'<li class="nav-item"><a class="nav-link {active_class}" data-toggle="tab" href="#mon-{slug}">{escape(model_name)}</a></li>'
        monthly_content += (
            f'<div id="mon-{slug}" class="tab-pane fade {show_class}">'
            f'<div class="chart-container">{_plot_monthly_returns_heatmap(model_name, frame)}</div>'
            "</div>"
        )
        first = False

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{escape(title)}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <style>
        body {{ background-color: #f8f9fa; padding: 20px; font-family: 'Segoe UI', sans-serif; }}
        .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none; border-radius: 10px; }}
        .card-header {{ background-color: #fff; border-bottom: 1px solid #eee; font-weight: bold; color: #333; }}
        .chart-container {{ padding: 15px; background: white; border-radius: 8px; }}
        .chart-container .plotly-graph-div {{ width: 100% !important; }}
        h2 {{ color: #2c3e50; margin-bottom: 30px; font-weight: 700; }}
        .nav-tabs .nav-link.active {{ background-color: #e9ecef; font-weight: bold; color: #007bff; }}
        .table-striped tbody tr:nth-of-type(odd) {{ background-color: rgba(0,0,0,.02); }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <center><h2>{escape(title)}</h2></center>

        <div class="card">
            <div class="card-header">Performance Metrics</div>
            <div class="card-body">{metrics_html}</div>
        </div>

        <div class="card">
            <div class="card-header">Worst Drawdown Periods</div>
            <div class="card-body">{worst_periods_html}</div>
        </div>

        <div class="card"><div class="card-body">{cumulative_html}</div></div>
        <div class="card"><div class="card-body">{drawdowns_html}</div></div>
        {f'<div class="card"><div class="card-body">{positions_html}</div></div>' if positions_html else ''}

        <div class="card">
            <div class="card-header">KPI Stability Analysis (Annual vs Cumulative)</div>
            <div class="card-body">
                <ul class="nav nav-tabs">{kpi_nav}</ul>
                <div class="tab-content pt-3">{kpi_content}</div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6"><div class="card"><div class="card-body">{risk_reward_html}</div></div></div>
            <div class="col-md-6"><div class="card"><div class="card-body">{correlation_html}</div></div></div>
        </div>

        <div class="card">
            <div class="card-header">Monthly Returns Deep Dive</div>
            <div class="card-body">
                <ul class="nav nav-tabs">{monthly_nav}</ul>
                <div class="tab-content pt-3">{monthly_content}</div>
            </div>
        </div>
    </div>
    <script>
        function resizeVisiblePlots(target) {{
            var scope = target || document;
            var plots = scope.querySelectorAll('.plotly-graph-div');
            plots.forEach(function(plot) {{
                if (window.Plotly && window.Plotly.Plots) {{
                    window.Plotly.Plots.resize(plot);
                }}
            }});
        }}
        $(document).on('shown.bs.tab', 'a[data-toggle="tab"]', function (event) {{
            var paneSelector = $(event.target).attr('href');
            if (!paneSelector) return;
            var pane = document.querySelector(paneSelector);
            if (pane) {{
                setTimeout(function() {{ resizeVisiblePlots(pane); }}, 50);
            }}
        }});
        $(window).on('load', function () {{
            setTimeout(function() {{ resizeVisiblePlots(document); }}, 50);
        }});
    </script>
</body>
</html>
"""


def compare_backtest_curves(
    curves: Dict[str, ApplicationBacktestResult | pl.DataFrame],
    *,
    output_path: str | Path | None = None,
    title: str = "Backtest Strategy Comparison",
    start_year: int | None = None,
    end_year: int | None = None,
    risk_free_rate: float = 0.02,
    return_column: str | None = None,
) -> BacktestComparisonResult:
    models_data = {
        name: _standardize_curve_frame(
            curve,
            return_column=return_column,
            start_year=start_year,
            end_year=end_year,
        )
        for name, curve in curves.items()
    }
    empty_curves = [name for name, frame in models_data.items() if frame.is_empty()]
    if empty_curves:
        raise ValueError(
            "Comparison curves have no monthly observations after filtering: " + ", ".join(sorted(empty_curves))
        )

    positions_dict = {
        name: frame.select(["year_month", "n"]).drop_nulls("n")
        for name, frame in models_data.items()
        if "n" in frame.columns and frame.select(pl.col("n").is_not_null().any()).item()
    }

    metrics = _build_metrics_table(models_data, risk_free_rate=risk_free_rate)
    aligned_returns = _aligned_returns_frame(models_data)
    cumulative_returns = _compute_cumulative_returns(aligned_returns)
    drawdowns = _compute_drawdowns(cumulative_returns)
    annual_returns = _compute_annual_returns(models_data)
    correlation_matrix = _compute_correlation_matrix(aligned_returns)
    cumulative_metrics = _metric_grids(models_data, risk_free_rate=risk_free_rate, exact_year=False)
    annual_metrics = _metric_grids(models_data, risk_free_rate=risk_free_rate, exact_year=True)
    worst_periods = _compute_worst_periods(models_data, annual_returns)
    monthly_returns = {
        name: frame.select(["year_month", "monthly_return"]).sort("year_month")
        for name, frame in models_data.items()
    }

    html_report = _build_comparison_report(
        title=title,
        metrics=metrics,
        cumulative_returns=cumulative_returns,
        drawdowns=drawdowns,
        annual_returns=annual_returns,
        correlation_matrix=correlation_matrix,
        worst_periods=worst_periods,
        cumulative_metrics=cumulative_metrics,
        annual_metrics=annual_metrics,
        monthly_returns=monthly_returns,
        positions_dict=positions_dict,
    )

    resolved_output_path: Path | None = None
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(html_report, encoding="utf-8")

    return BacktestComparisonResult(
        metrics=metrics,
        cumulative_returns=cumulative_returns,
        correlation_matrix=correlation_matrix,
        worst_periods=worst_periods,
        drawdowns=drawdowns,
        annual_returns=annual_returns,
        cumulative_metrics=cumulative_metrics,
        annual_metrics=annual_metrics,
        monthly_returns=monthly_returns,
        html=html_report,
        output_path=resolved_output_path,
    )
