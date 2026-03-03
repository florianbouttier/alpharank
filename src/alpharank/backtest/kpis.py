from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import polars as pl


FLOAT_DTYPES = {pl.Float32, pl.Float64}


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _compute_series_kpis(returns: np.ndarray, risk_free_rate: float) -> Dict[str, float]:
    clean = returns[np.isfinite(returns)]
    if clean.size == 0:
        clean = np.array([0.0], dtype=float)

    total_return = np.prod(1.0 + clean) - 1.0
    n_periods = clean.size
    n_years = max(n_periods / 12.0, 1.0 / 12.0)

    if 1.0 + total_return <= 0.0:
        cagr = -1.0
    else:
        cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0

    monthly_mean = np.mean(clean)
    monthly_std = np.std(clean, ddof=1) if clean.size > 1 else 0.0
    annualized_vol = monthly_std * np.sqrt(12.0)

    downside = clean[clean < 0.0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(12.0) if downside.size > 1 else 0.0

    sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol > 1e-12 else 0.0
    sortino = (cagr - risk_free_rate) / downside_std if downside_std > 1e-12 else 0.0

    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth / peaks) - 1.0
    max_drawdown = float(np.min(drawdowns)) if drawdowns.size else 0.0

    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else 0.0
    win_rate = float(np.mean(clean > 0.0))

    return {
        "total_return": _finite_or_zero(total_return),
        "cagr": _finite_or_zero(cagr),
        "avg_monthly_return": _finite_or_zero(monthly_mean),
        "annualized_volatility": _finite_or_zero(annualized_vol),
        "sharpe_ratio": _finite_or_zero(sharpe),
        "sortino_ratio": _finite_or_zero(sortino),
        "calmar_ratio": _finite_or_zero(calmar),
        "max_drawdown": _finite_or_zero(max_drawdown),
        "win_rate": _finite_or_zero(win_rate),
        "months": float(n_periods),
    }


def compute_backtest_kpis(monthly_returns: pl.DataFrame, risk_free_rate: float) -> pl.DataFrame:
    if monthly_returns.is_empty():
        empty = pl.DataFrame(
            {
                "strategy": ["Portfolio", "Benchmark", "Active"],
                "total_return": [0.0, 0.0, 0.0],
                "cagr": [0.0, 0.0, 0.0],
                "avg_monthly_return": [0.0, 0.0, 0.0],
                "annualized_volatility": [0.0, 0.0, 0.0],
                "sharpe_ratio": [0.0, 0.0, 0.0],
                "sortino_ratio": [0.0, 0.0, 0.0],
                "calmar_ratio": [0.0, 0.0, 0.0],
                "max_drawdown": [0.0, 0.0, 0.0],
                "win_rate": [0.0, 0.0, 0.0],
                "months": [0.0, 0.0, 0.0],
                "avg_hit_rate": [0.0, 0.0, 0.0],
                "avg_positions": [0.0, 0.0, 0.0],
            }
        )
        return empty

    portfolio = monthly_returns.get_column("portfolio_return").to_numpy()
    benchmark = monthly_returns.get_column("benchmark_return").to_numpy()
    active = monthly_returns.get_column("active_return").to_numpy()

    rows: List[Dict[str, float | str]] = []

    for strategy, series, rf in [
        ("Portfolio", portfolio, risk_free_rate),
        ("Benchmark", benchmark, risk_free_rate),
        ("Active", active, 0.0),
    ]:
        metrics = _compute_series_kpis(series.astype(float), risk_free_rate=rf)
        rows.append({"strategy": strategy, **metrics})

    kpis = pl.DataFrame(rows)

    avg_hit_rate = float(monthly_returns.get_column("hit_rate").mean() or 0.0)
    avg_positions = float(monthly_returns.get_column("n_positions").mean() or 0.0)

    kpis = kpis.with_columns(
        pl.when(pl.col("strategy") == pl.lit("Portfolio"))
        .then(pl.lit(avg_hit_rate))
        .otherwise(pl.lit(0.0))
        .alias("avg_hit_rate"),
        pl.when(pl.col("strategy") == pl.lit("Portfolio"))
        .then(pl.lit(avg_positions))
        .otherwise(pl.lit(0.0))
        .alias("avg_positions"),
    )

    return sanitize_numeric_frame(kpis)


def sanitize_numeric_frame(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for col_name, dtype in df.schema.items():
        if dtype in FLOAT_DTYPES:
            exprs.append(pl.col(col_name).fill_nan(0.0).fill_null(0.0).alias(col_name))
        elif dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            exprs.append(pl.col(col_name).fill_null(0).alias(col_name))

    if not exprs:
        return df

    return df.with_columns(exprs)


def assert_no_numeric_na(df: pl.DataFrame, context: str) -> None:
    issues: List[str] = []

    for col_name, dtype in df.schema.items():
        null_count = int(df.select(pl.col(col_name).is_null().sum()).item())
        if dtype in FLOAT_DTYPES:
            nan_count = int(df.select(pl.col(col_name).is_nan().sum()).item())
        else:
            nan_count = 0

        if null_count > 0 or nan_count > 0:
            issues.append(f"{col_name} (null={null_count}, nan={nan_count})")

    if issues:
        raise ValueError(f"Found NA/NaN in {context}: {', '.join(issues)}")
