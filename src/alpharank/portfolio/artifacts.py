from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.portfolio.contracts import validate_holdings, validate_monthly_returns
from alpharank.portfolio.performance import annual_returns, legacy_report_statistics


def write_common_portfolio_artifacts(
    *,
    output_dir: Path,
    holdings: pl.DataFrame,
    monthly_returns: pl.DataFrame,
    prefix: str = "portfolio_common",
    risk_free_rate: float = 0.02,
) -> dict[str, Path]:
    """Persist the shared audit contract used by Legacy and boosting reports."""

    validate_holdings(holdings)
    validate_monthly_returns(monthly_returns)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "holdings": output_dir / f"{prefix}_holdings.parquet",
        "monthly_parquet": output_dir / f"{prefix}_monthly.parquet",
        "monthly_csv": output_dir / f"{prefix}_monthly.csv",
        "annual_csv": output_dir / f"{prefix}_annual.csv",
        "performance_csv": output_dir / f"{prefix}_performance.csv",
        "calendar_json": output_dir / f"{prefix}_calendar.json",
    }
    holdings.sort(["strategy", "decision_month", "ticker"]).write_parquet(paths["holdings"])
    monthly = monthly_returns.sort(["strategy", "holding_month"])
    monthly.write_parquet(paths["monthly_parquet"])
    monthly.write_csv(paths["monthly_csv"])

    annual_parts: list[pl.DataFrame] = []
    performance_rows: list[dict[str, Any]] = []
    calendar_rows: list[dict[str, Any]] = []
    for strategy_frame in monthly.partition_by("strategy", maintain_order=True):
        strategy = str(strategy_frame["strategy"][0])
        yearly = annual_returns(
            strategy_frame["net_return"].to_numpy(),
            holding_months=strategy_frame["holding_month"].to_list(),
        ).with_columns(pl.lit(strategy).alias("strategy"))
        annual_parts.append(yearly.select("strategy", *[c for c in yearly.columns if c != "strategy"]))
        performance_rows.append(
            {
                "strategy": strategy,
                "start_holding_month": strategy_frame["holding_month"].min(),
                "end_holding_month": strategy_frame["holding_month"].max(),
                "months": strategy_frame.height,
                **legacy_report_statistics(
                    strategy_frame["net_return"].to_numpy(),
                    holding_months=strategy_frame["holding_month"].to_list(),
                    risk_free_rate=risk_free_rate,
                ),
            }
        )
        calendar_rows.append(
            {
                "strategy": strategy,
                "start_decision_month": str(strategy_frame["decision_month"].min()),
                "end_decision_month": str(strategy_frame["decision_month"].max()),
                "start_holding_month": str(strategy_frame["holding_month"].min()),
                "end_holding_month": str(strategy_frame["holding_month"].max()),
                "months": strategy_frame.height,
            }
        )
    pl.concat(annual_parts, how="diagonal_relaxed").write_csv(paths["annual_csv"])
    pl.DataFrame(performance_rows).write_csv(paths["performance_csv"])
    paths["calendar_json"].write_text(
        json.dumps(
            {
                "timing_contract": "decision_month=t; holding_month=t+1",
                "return_contract": "gross - turnover*cost_bps/10000 = net",
                "sharpe_convention": "(CAGR - risk_free_rate) / annualized_volatility",
                "risk_free_rate": risk_free_rate,
                "strategies": calendar_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
