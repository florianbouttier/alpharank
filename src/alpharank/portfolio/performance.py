from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import polars as pl


def performance_statistics(
    returns: np.ndarray | Iterable[float],
    *,
    risk_free_rate: float = 0.0,
    sharpe_convention: str = "arithmetic",
) -> dict[str, float]:
    """Canonical statistics with an explicit Sharpe convention.

    ``arithmetic`` preserves the historical multi-horizon research definition.
    ``legacy`` uses ``(CAGR - risk_free_rate) / annualized volatility`` and is
    the comparison convention for Legacy, Alpha, and SPY reports.
    """

    clean = np.asarray(list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_month_rate": float("nan"),
        }
    curve = np.cumprod(1.0 + clean)
    total = float(curve[-1] - 1.0)
    years = clean.size / 12.0
    cagr = float(curve[-1] ** (1.0 / years) - 1.0) if curve[-1] > 0 and years > 0 else -1.0
    monthly_std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    volatility = monthly_std * math.sqrt(12.0)
    if sharpe_convention == "arithmetic":
        sharpe = (
            float(np.mean(clean) / monthly_std * math.sqrt(12.0))
            if monthly_std > 0.0
            else float("nan")
        )
    elif sharpe_convention == "legacy":
        sharpe = (
            float((cagr - risk_free_rate) / volatility)
            if volatility > 0.0
            else float("nan")
        )
    else:
        raise ValueError(f"Unknown sharpe_convention={sharpe_convention!r}.")
    running_peak = np.maximum.accumulate(curve)
    max_drawdown = float(np.min(curve / running_peak - 1.0))
    return {
        "total_return": total,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "positive_month_rate": float(np.mean(clean > 0.0)),
    }


def advanced_performance_statistics(
    returns: np.ndarray | Iterable[float],
    *,
    benchmark_returns: np.ndarray | Iterable[float] | None = None,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """Canonical advanced statistics for a monthly strategy comparison."""

    values = np.asarray(
        list(returns) if not isinstance(returns, np.ndarray) else returns,
        dtype=float,
    )
    benchmark = None
    if benchmark_returns is not None:
        benchmark = np.asarray(
            list(benchmark_returns)
            if not isinstance(benchmark_returns, np.ndarray)
            else benchmark_returns,
            dtype=float,
        )
        if benchmark.size != values.size:
            raise ValueError("returns and benchmark_returns must have the same length")
        valid = np.isfinite(values) & np.isfinite(benchmark)
        benchmark = benchmark[valid]
    else:
        valid = np.isfinite(values)
    values = values[valid]
    base = performance_statistics(
        values,
        risk_free_rate=risk_free_rate,
        sharpe_convention="legacy",
    )
    if values.size == 0:
        return {
            **base,
            **{
                key: float("nan")
                for key in (
                    "sortino",
                    "calmar",
                    "information_ratio",
                    "beta",
                    "alpha",
                    "correlation",
                    "benchmark_hit_rate",
                    "var_95",
                    "cvar_95",
                    "omega",
                    "up_capture",
                    "down_capture",
                )
            },
        }

    downside = math.sqrt(float(np.mean(np.minimum(values, 0.0) ** 2)) * 12.0)
    sortino = (
        (base["cagr"] - risk_free_rate) / downside
        if downside > 0.0
        else float("nan")
    )
    calmar = (
        base["cagr"] / abs(base["max_drawdown"])
        if base["max_drawdown"] < 0.0
        else float("nan")
    )
    var_95 = float(np.quantile(values, 0.05))
    tail = values[values <= var_95]
    gains = float(values[values > 0.0].sum())
    losses = abs(float(values[values < 0.0].sum()))
    result = {
        **base,
        "sortino": float(sortino),
        "calmar": float(calmar),
        "var_95": var_95,
        "cvar_95": float(np.mean(tail)) if tail.size else float("nan"),
        "omega": gains / losses if losses > 0.0 else float("nan"),
        "information_ratio": float("nan"),
        "beta": float("nan"),
        "alpha": float("nan"),
        "correlation": float("nan"),
        "benchmark_hit_rate": float("nan"),
        "up_capture": float("nan"),
        "down_capture": float("nan"),
    }
    if benchmark is None or values.size < 2:
        return result

    active = values - benchmark
    tracking_error = float(np.std(active, ddof=1) * math.sqrt(12.0))
    benchmark_variance = float(np.var(benchmark, ddof=1))
    covariance = float(np.cov(values, benchmark, ddof=1)[0, 1])
    beta = covariance / benchmark_variance if benchmark_variance > 0.0 else float("nan")
    monthly_risk_free = (1.0 + risk_free_rate) ** (1.0 / 12.0) - 1.0
    alpha = (
        12.0
        * (
            (float(np.mean(values)) - monthly_risk_free)
            - beta * (float(np.mean(benchmark)) - monthly_risk_free)
        )
        if math.isfinite(beta)
        else float("nan")
    )
    up = benchmark > 0.0
    down = benchmark < 0.0
    result.update(
        information_ratio=(
            12.0 * float(np.mean(active)) / tracking_error
            if tracking_error > 0.0
            else float("nan")
        ),
        beta=beta,
        alpha=alpha,
        correlation=float(np.corrcoef(values, benchmark)[0, 1]),
        benchmark_hit_rate=float(np.mean(values > benchmark)),
        up_capture=(
            float(np.mean(values[up]) / np.mean(benchmark[up]))
            if np.any(up) and np.mean(benchmark[up]) != 0.0
            else float("nan")
        ),
        down_capture=(
            float(np.mean(values[down]) / np.mean(benchmark[down]))
            if np.any(down) and np.mean(benchmark[down]) != 0.0
            else float("nan")
        ),
    )
    return result


def annual_returns(
    returns: np.ndarray | Iterable[float],
    *,
    holding_months: Iterable[date | datetime | np.datetime64],
    full_years_only: bool = False,
) -> pl.DataFrame:
    values = np.asarray(list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float)
    months = np.asarray(list(holding_months))
    if values.size != months.size:
        raise ValueError("returns and holding_months must have the same length")
    valid = np.isfinite(values) & ~np.isnat(months.astype("datetime64[ns]"))
    values = values[valid]
    months = months[valid].astype("datetime64[M]")
    rows: list[dict[str, float | int | bool]] = []
    if values.size:
        years = months.astype("datetime64[Y]").astype(int) + 1970
        month_numbers = months.astype(int) - months.astype("datetime64[Y]").astype(int) * 12 + 1
        for year in np.unique(years):
            in_year = years == year
            complete = set(month_numbers[in_year].tolist()) == set(range(1, 13))
            if full_years_only and not complete:
                continue
            rows.append(
                {
                    "year": int(year),
                    "months": int(np.sum(in_year)),
                    "is_full_calendar_year": complete,
                    "annual_return": float(np.prod(1.0 + values[in_year]) - 1.0),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "months": pl.Int64,
            "is_full_calendar_year": pl.Boolean,
            "annual_return": pl.Float64,
        },
    )


def legacy_report_statistics(
    returns: np.ndarray | Iterable[float],
    *,
    holding_months: Iterable[date | datetime | np.datetime64],
    risk_free_rate: float = 0.02,
) -> dict[str, float | int]:
    values = np.asarray(list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float)
    months = list(holding_months)
    if values.size != len(months):
        raise ValueError("returns and holding_months must have the same length")
    base = performance_statistics(
        values,
        risk_free_rate=risk_free_rate,
        sharpe_convention="legacy",
    )
    yearly = annual_returns(values, holding_months=months, full_years_only=True)
    if yearly.is_empty():
        worst_year = -1
        worst_return = float("nan")
    else:
        worst = yearly.sort("annual_return").row(0, named=True)
        worst_year = int(worst["year"])
        worst_return = float(worst["annual_return"])
    return {
        **base,
        "risk_free_rate": float(risk_free_rate),
        "worst_full_calendar_year": worst_year,
        "worst_full_calendar_year_return": worst_return,
        "full_calendar_years": yearly.height,
    }
