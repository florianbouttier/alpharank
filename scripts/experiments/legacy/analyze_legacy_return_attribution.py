#!/usr/bin/env python3
"""Build additive ticker and monthly attribution for a Legacy backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month(value: str) -> pd.Period:
    return pd.Period(value, freq="M")


def _cagr(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("Cannot calculate CAGR from an empty return series.")
    wealth = float(np.prod(1.0 + returns.to_numpy(dtype=float)))
    if wealth <= 0:
        raise ValueError(f"Non-positive terminal wealth: {wealth}")
    return wealth ** (12.0 / len(returns)) - 1.0


def _monthly_adjusted_returns(path: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    query = (
        pl.scan_parquet(path)
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .filter(pl.col("adjusted_close").is_not_null())
    )
    if tickers is not None:
        query = query.filter(pl.col("ticker").is_in(tickers))
    group_keys = ["ticker", "year_month"] if tickers is not None else ["year_month"]
    sort_keys = ["ticker", "date"] if tickers is not None else ["date"]
    result = (
        query.with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
        .sort(sort_keys)
        .group_by(group_keys, maintain_order=True)
        .agg(
            pl.col("adjusted_close").last().alias("last_close"),
            pl.col("date").last().alias("last_date"),
        )
        .sort(group_keys)
    )
    if tickers is not None:
        result = result.with_columns(
            pl.col("last_close").pct_change().over("ticker").alias("return")
        )
    else:
        result = result.with_columns(pl.col("last_close").pct_change().alias("return"))
    frame = result.collect().to_pandas()
    frame["year_month"] = pd.to_datetime(frame["year_month"]).dt.to_period("M")
    return frame


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int) -> str:
    view = frame.loc[:, columns].head(limit).copy()
    headers = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in view.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([headers, separator, *rows])


def build_attribution(
    *,
    run_dir: Path,
    snapshot_dir: Path,
    output_dir: Path,
    portfolio_model: str,
    through_month: pd.Period,
    refresh_month: pd.Period | None,
) -> dict[str, object]:
    detailed_path = run_dir / "legacy_detailed_returns_polars.parquet"
    aggregated_path = run_dir / "legacy_aggregated_returns_polars.parquet"
    stock_prices_path = snapshot_dir / "US_Finalprice.parquet"
    benchmark_prices_path = snapshot_dir / "SP500Price.parquet"
    for path in [detailed_path, aggregated_path, stock_prices_path, benchmark_prices_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    detailed = (
        pl.read_parquet(detailed_path)
        .filter(pl.col("portfolio_model") == portfolio_model)
        .select("year_month", "ticker", "dr", "weight_normalized", "n_models")
        .to_pandas()
    )
    detailed["year_month"] = pd.to_datetime(detailed["year_month"]).dt.to_period("M")
    detailed = detailed[detailed["year_month"] <= through_month].copy()
    if detailed.empty:
        raise ValueError(f"No {portfolio_model} rows through {through_month}.")

    refreshed_rows = 0
    if refresh_month is not None:
        refresh_mask = detailed["year_month"] == refresh_month
        refresh_tickers = sorted(detailed.loc[refresh_mask, "ticker"].unique().tolist())
        if not refresh_tickers:
            raise ValueError(f"No holdings found for refresh month {refresh_month}.")
        refreshed = _monthly_adjusted_returns(stock_prices_path, refresh_tickers)
        refreshed = refreshed[refreshed["year_month"] == refresh_month][["ticker", "return"]]
        return_map = dict(zip(refreshed["ticker"], refreshed["return"], strict=True))
        missing = sorted(set(refresh_tickers) - set(return_map))
        if missing:
            raise ValueError(f"Missing refreshed returns for {refresh_month}: {missing}")
        detailed.loc[refresh_mask, "dr"] = detailed.loc[refresh_mask, "ticker"].map(return_map)
        refreshed_rows = int(refresh_mask.sum())

    if (detailed["dr"].dropna() <= -1.0).any():
        invalid = detailed[detailed["dr"] <= -1.0][["year_month", "ticker", "dr"]]
        raise ValueError(f"Returns at or below -100%:\n{invalid.to_string(index=False)}")

    # The Legacy aggregate excludes holdings with a missing realized return and
    # renormalizes the remaining weights for that month.
    detailed["valid_return_weight"] = detailed["weight_normalized"].where(
        detailed["dr"].notna(), 0.0
    )
    valid_weight_sums = detailed.groupby("year_month")["valid_return_weight"].transform("sum")
    if (valid_weight_sums <= 0).any():
        raise ValueError("At least one month has no holding with a valid realized return.")
    detailed["effective_weight"] = detailed["valid_return_weight"] / valid_weight_sums

    weight_sums = detailed.groupby("year_month")["effective_weight"].sum()
    max_weight_error = float((weight_sums - 1.0).abs().max())
    if max_weight_error > 1e-9:
        raise ValueError(f"Monthly weights do not sum to one; max error={max_weight_error}")

    detailed["simple_contribution"] = (
        detailed["dr"].fillna(0.0) * detailed["effective_weight"]
    )
    detailed["selected_weight_cash_contribution"] = (
        detailed["dr"].fillna(0.0) * detailed["weight_normalized"]
    )
    monthly = (
        detailed.groupby("year_month", as_index=True)
        .agg(
            portfolio_return=("simple_contribution", "sum"),
            cash_for_missing_returns_return=(
                "selected_weight_cash_contribution",
                "sum",
            ),
            holdings=("ticker", "size"),
            max_weight=("effective_weight", "max"),
        )
        .sort_index()
    )
    monthly = monthly.loc[:through_month]
    n_months = len(monthly)

    stored = (
        pl.read_parquet(aggregated_path)
        .filter(pl.col("portfolio_model") == portfolio_model)
        .select("year_month", "monthly_return")
        .to_pandas()
    )
    stored["year_month"] = pd.to_datetime(stored["year_month"]).dt.to_period("M")
    stored_series = stored.set_index("year_month")["monthly_return"].sort_index()
    validation_months = monthly.index
    if refresh_month is not None:
        validation_months = validation_months[validation_months != refresh_month]
    common = validation_months.intersection(stored_series.index)
    max_aggregate_error = float(
        (monthly.loc[common, "portfolio_return"] - stored_series.loc[common]).abs().max()
    )
    if max_aggregate_error > 1e-10:
        raise ValueError(f"Detailed returns do not reproduce aggregate; max error={max_aggregate_error}")

    benchmark = _monthly_adjusted_returns(benchmark_prices_path)
    benchmark_series = benchmark.set_index("year_month")["return"].sort_index()
    missing_benchmark = monthly.index.difference(benchmark_series.dropna().index)
    if len(missing_benchmark):
        raise ValueError(f"Missing adjusted SPY returns: {missing_benchmark.tolist()}")
    monthly["spy_adjusted_return"] = benchmark_series.loc[monthly.index]
    monthly["active_return"] = monthly["portfolio_return"] - monthly["spy_adjusted_return"]

    overall_cagr = _cagr(monthly["portfolio_return"])
    cash_for_missing_returns_cagr = _cagr(
        monthly["cash_for_missing_returns_return"]
    )
    spy_cagr = _cagr(monthly["spy_adjusted_return"])
    annualized_log_return = 12.0 / n_months * float(
        np.log1p(monthly["portfolio_return"]).sum()
    )
    monthly["log_return"] = np.log1p(monthly["portfolio_return"])
    monthly["annualized_log_contribution_pp"] = 100.0 * 12.0 / n_months * monthly["log_return"]
    monthly["wealth_before"] = (1.0 + monthly["portfolio_return"]).cumprod().shift(1).fillna(1.0)
    monthly["wealth_after"] = monthly["wealth_before"] * (1.0 + monthly["portfolio_return"])
    monthly["cash_cagr_impact_pp"] = [
        100.0
        * (
            overall_cagr
            - _cagr(monthly["portfolio_return"].where(monthly.index != month, 0.0))
        )
        for month in monthly.index
    ]

    factors = monthly["log_return"] / monthly["portfolio_return"]
    factors = factors.where(monthly["portfolio_return"].abs() > 1e-15, 1.0)
    detailed["log_contribution"] = detailed["simple_contribution"] * detailed["year_month"].map(factors)

    ticker_month_export = detailed[
        [
            "year_month",
            "ticker",
            "dr",
            "weight_normalized",
            "effective_weight",
            "simple_contribution",
            "log_contribution",
        ]
    ].copy()
    ticker_month_export["year_month"] = ticker_month_export["year_month"].astype(str)
    ticker_month_export["stock_return_pct"] = 100.0 * ticker_month_export.pop("dr")
    ticker_month_export["selected_weight_pct"] = (
        100.0 * ticker_month_export.pop("weight_normalized")
    )
    ticker_month_export["effective_weight_pct"] = (
        100.0 * ticker_month_export.pop("effective_weight")
    )
    ticker_month_export["simple_contribution_pp"] = (
        100.0 * ticker_month_export.pop("simple_contribution")
    )
    ticker_month_export["annualized_log_contribution_pp"] = (
        100.0 * 12.0 / n_months * ticker_month_export.pop("log_contribution")
    )

    ticker_rows: list[dict[str, object]] = []
    for ticker, group in detailed.groupby("ticker", sort=True):
        monthly_contribution = group.groupby("year_month")["simple_contribution"].sum()
        counterfactual = monthly["portfolio_return"].subtract(monthly_contribution, fill_value=0.0)
        ticker_rows.append(
            {
                "ticker": ticker,
                "months_held": int(group["year_month"].nunique()),
                "months_with_valid_return": int(group.loc[group["dr"].notna(), "year_month"].nunique()),
                "missing_return_months": int(group["dr"].isna().sum()),
                "first_month": str(group["year_month"].min()),
                "last_month": str(group["year_month"].max()),
                "average_selected_weight_when_held_pct": 100.0
                * float(group["weight_normalized"].mean()),
                "average_effective_weight_when_held_pct": 100.0
                * float(group["effective_weight"].mean()),
                "simple_contribution_sum_pp": 100.0 * float(group["simple_contribution"].sum()),
                "annualized_log_contribution_pp": 100.0
                * 12.0
                / n_months
                * float(group["log_contribution"].sum()),
                "cash_cagr_impact_pp": 100.0 * (overall_cagr - _cagr(counterfactual)),
                "min_stock_month_return_pct": 100.0 * float(group["dr"].min()),
                "max_stock_month_return_pct": 100.0 * float(group["dr"].max()),
                "max_abs_month_contribution_pp": 100.0
                * float(group["simple_contribution"].abs().max()),
                "months_abs_return_over_50pct": int((group["dr"].abs() > 0.5).sum()),
                "months_abs_return_over_100pct": int((group["dr"].abs() > 1.0).sum()),
            }
        )
    ticker_attribution = pd.DataFrame(ticker_rows).sort_values(
        ["annualized_log_contribution_pp", "ticker"], ascending=[False, True]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_path = output_dir / "ticker_contributions.csv"
    ticker_month_path = output_dir / "ticker_month_contributions.csv"
    monthly_path = output_dir / "monthly_contributions.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "README.md"
    ticker_attribution.to_csv(ticker_path, index=False)
    ticker_month_export.sort_values(["year_month", "ticker"]).to_csv(
        ticker_month_path, index=False
    )
    monthly_export = monthly.reset_index()
    monthly_export["year_month"] = monthly_export["year_month"].astype(str)
    monthly_export.to_csv(monthly_path, index=False)

    summary: dict[str, object] = {
        "portfolio_model": portfolio_model,
        "run_dir": str(run_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "start_month": str(monthly.index.min()),
        "through_month": str(monthly.index.max()),
        "months": n_months,
        "ticker_count": int(ticker_attribution.shape[0]),
        "positive_ticker_count": int(
            (ticker_attribution["annualized_log_contribution_pp"] > 0.0).sum()
        ),
        "negative_ticker_count": int(
            (ticker_attribution["annualized_log_contribution_pp"] < 0.0).sum()
        ),
        "tickers_with_monthly_return_over_50pct": int(
            (ticker_attribution["months_abs_return_over_50pct"] > 0).sum()
        ),
        "ticker_month_returns_over_100pct": int(
            ticker_attribution["months_abs_return_over_100pct"].sum()
        ),
        "missing_return_ticker_months": int(detailed["dr"].isna().sum()),
        "portfolio_cagr": overall_cagr,
        "cash_for_missing_returns_cagr": cash_for_missing_returns_cagr,
        "missing_return_renormalization_cagr_lift_pp": 100.0
        * (overall_cagr - cash_for_missing_returns_cagr),
        "spy_adjusted_cagr": spy_cagr,
        "annualized_log_return": annualized_log_return,
        "terminal_wealth": float((1.0 + monthly["portfolio_return"]).prod()),
        "max_weight_sum_error": max_weight_error,
        "max_detailed_vs_aggregate_error": max_aggregate_error,
        "refreshed_month": str(refresh_month) if refresh_month is not None else None,
        "refreshed_holding_rows": refreshed_rows,
        "annualized_ticker_log_contribution_sum": float(
            ticker_attribution["annualized_log_contribution_pp"].sum() / 100.0
        ),
        "annualized_month_log_contribution_sum": float(
            monthly["annualized_log_contribution_pp"].sum() / 100.0
        ),
        "top_5_ticker_share_of_net_log_return": float(
            ticker_attribution.head(5)["annualized_log_contribution_pp"].sum()
            / ticker_attribution["annualized_log_contribution_pp"].sum()
        ),
        "top_20_ticker_share_of_net_log_return": float(
            ticker_attribution.head(20)["annualized_log_contribution_pp"].sum()
            / ticker_attribution["annualized_log_contribution_pp"].sum()
        ),
        "input_sha256": {
            "detailed_returns": _sha256(detailed_path),
            "aggregated_returns": _sha256(aggregated_path),
            "stock_prices": _sha256(stock_prices_path),
            "benchmark_prices": _sha256(benchmark_prices_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top = ticker_attribution.head(15)
    bottom = ticker_attribution.sort_values(
        ["annualized_log_contribution_pp", "ticker"], ascending=[True, True]
    ).head(15)
    best_months = monthly_export.sort_values("annualized_log_contribution_pp", ascending=False)
    worst_months = monthly_export.sort_values("annualized_log_contribution_pp", ascending=True)
    report = f"""# Legacy Return Attribution

- Model: `{portfolio_model}`
- Period: `{summary['start_month']}` to `{summary['through_month']}` ({n_months} months)
- Portfolio CAGR: `{overall_cagr:.4%}`
- SPY adjusted CAGR: `{spy_cagr:.4%}`
- Annualized additive log return: `{annualized_log_return:.4%}`
- Terminal wealth multiple: `{summary['terminal_wealth']:.6f}`
- Tickers held: `{summary['ticker_count']}`
- Positive / negative ticker contributions: `{summary['positive_ticker_count']}` / `{summary['negative_ticker_count']}`
- Top 5 / top 20 share of net log return: `{summary['top_5_ticker_share_of_net_log_return']:.2%}` / `{summary['top_20_ticker_share_of_net_log_return']:.2%}`
- Tickers with at least one monthly return above 50% in absolute value: `{summary['tickers_with_monthly_return_over_50pct']}`
- Ticker-month returns above 100% in absolute value: `{summary['ticker_month_returns_over_100pct']}`
- Missing ticker-month returns: `{summary['missing_return_ticker_months']}`
- CAGR with missing returns held as cash: `{summary['cash_for_missing_returns_cagr']:.4%}` (Legacy renormalization lift: `{summary['missing_return_renormalization_cagr_lift_pp']:.4f}` pp)
- Refreshed month: `{summary['refreshed_month']}` ({refreshed_rows} holdings)

The additive ticker and month columns allocate `log(1 + portfolio_return)`.
Their sums equal the annualized log return exactly; exponentiating that sum
recovers the compounded CAGR. `cash_cagr_impact_pp` is a non-additive marginal
counterfactual that replaces the relevant ticker or month with cash.

User-facing reports: `html/index.html`, `html/ticker_attribution.html`,
`html/monthly_attribution.html`, and `html/preprocessing_impact.html`.

## Top Ticker Contributions

{_markdown_table(top, ['ticker', 'months_held', 'annualized_log_contribution_pp', 'cash_cagr_impact_pp', 'max_stock_month_return_pct'], 15)}

## Bottom Ticker Contributions

{_markdown_table(bottom, ['ticker', 'months_held', 'annualized_log_contribution_pp', 'cash_cagr_impact_pp', 'min_stock_month_return_pct'], 15)}

## Best Months

{_markdown_table(best_months, ['year_month', 'portfolio_return', 'spy_adjusted_return', 'annualized_log_contribution_pp', 'cash_cagr_impact_pp'], 15)}

## Worst Months

{_markdown_table(worst_months, ['year_month', 'portfolio_return', 'spy_adjusted_return', 'annualized_log_contribution_pp', 'cash_cagr_impact_pp'], 15)}
"""
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portfolio-model", default="Combined_Frequency")
    parser.add_argument("--through-month", type=_month, required=True)
    parser.add_argument("--refresh-month", type=_month)
    args = parser.parse_args()
    summary = build_attribution(
        run_dir=args.run_dir,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        portfolio_model=args.portfolio_model,
        through_month=args.through_month,
        refresh_month=args.refresh_month,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
