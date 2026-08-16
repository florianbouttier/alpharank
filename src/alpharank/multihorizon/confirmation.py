from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from scipy.stats import kurtosis, norm, skew

from alpharank.portfolio.performance import performance_statistics


@dataclass(frozen=True)
class BootstrapResult:
    comparator: str
    observed_annualized_mean_difference: float
    observed_sharpe_difference: float
    annualized_mean_ci_low: float
    annualized_mean_ci_high: float
    sharpe_difference_ci_low: float
    sharpe_difference_ci_high: float
    probability_mean_difference_le_zero: float
    probability_sharpe_difference_le_zero: float


def add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def moving_block_indices(
    sample_size: int,
    *,
    block_months: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size < 1:
        raise ValueError("sample_size must be positive.")
    if block_months < 1:
        raise ValueError("block_months must be positive.")
    blocks = math.ceil(sample_size / block_months)
    starts = rng.integers(0, sample_size, size=blocks)
    offsets = np.arange(block_months)
    return np.concatenate(
        [(start + offsets) % sample_size for start in starts]
    )[:sample_size]


def paired_block_bootstrap(
    monthly: pl.DataFrame,
    *,
    comparator_columns: Mapping[str, str],
    samples: int = 10_000,
    block_months: int = 12,
    seed: int = 42,
) -> pl.DataFrame:
    model = monthly["net_return"].to_numpy().astype(float)
    if len(model) < block_months * 2:
        raise ValueError("At least two temporal blocks are required.")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for comparator, column in comparator_columns.items():
        reference = monthly[column].to_numpy().astype(float)
        observed_mean = float(np.mean(model - reference) * 12.0)
        observed_sharpe = (
            performance_statistics(model)["sharpe"]
            - performance_statistics(reference)["sharpe"]
        )
        mean_samples = np.empty(samples, dtype=float)
        sharpe_samples = np.empty(samples, dtype=float)
        for sample in range(samples):
            indices = moving_block_indices(
                len(model),
                block_months=block_months,
                rng=rng,
            )
            sampled_model = model[indices]
            sampled_reference = reference[indices]
            mean_samples[sample] = float(
                np.mean(sampled_model - sampled_reference) * 12.0
            )
            sharpe_samples[sample] = float(
                performance_statistics(sampled_model)["sharpe"]
                - performance_statistics(sampled_reference)["sharpe"]
            )
        mean_low, mean_high = np.quantile(mean_samples, [0.025, 0.975])
        sharpe_low, sharpe_high = np.quantile(sharpe_samples, [0.025, 0.975])
        rows.append(
            {
                "comparator": comparator,
                "observed_annualized_mean_difference": observed_mean,
                "observed_sharpe_difference": observed_sharpe,
                "annualized_mean_ci_low": float(mean_low),
                "annualized_mean_ci_high": float(mean_high),
                "sharpe_difference_ci_low": float(sharpe_low),
                "sharpe_difference_ci_high": float(sharpe_high),
                "probability_mean_difference_le_zero": float(
                    np.mean(mean_samples <= 0.0)
                ),
                "probability_sharpe_difference_le_zero": float(
                    np.mean(sharpe_samples <= 0.0)
                ),
                "bootstrap_samples": samples,
                "block_months": block_months,
                "seed": seed,
            }
        )
    return pl.DataFrame(rows)


def deflated_sharpe_statistics(
    returns: Sequence[float] | np.ndarray,
    *,
    trials: int,
) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or trials < 1:
        raise ValueError("Deflated Sharpe requires at least three returns and one trial.")
    monthly_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    skewness = float(skew(values, bias=False))
    pearson_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    euler_gamma = 0.5772156649015329
    null_standard_error = 1.0 / math.sqrt(len(values) - 1)
    expected_maximum = null_standard_error * (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
        + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * math.e))
    )
    sampling_variance = (
        1.0
        - skewness * monthly_sharpe
        + ((pearson_kurtosis - 1.0) / 4.0) * monthly_sharpe**2
    ) / (len(values) - 1)
    z_score = (monthly_sharpe - expected_maximum) / math.sqrt(
        max(sampling_variance, 1e-12)
    )
    return {
        "observations": len(values),
        "trials": trials,
        "observed_monthly_sharpe": monthly_sharpe,
        "observed_annualized_sharpe": monthly_sharpe * math.sqrt(12.0),
        "expected_maximum_monthly_sharpe_under_null": expected_maximum,
        "expected_maximum_annualized_sharpe_under_null": (
            expected_maximum * math.sqrt(12.0)
        ),
        "return_skewness": skewness,
        "return_pearson_kurtosis": pearson_kurtosis,
        "deflated_sharpe_z": z_score,
        "deflated_sharpe_probability": float(norm.cdf(z_score)),
    }


def cost_sensitivity(
    monthly: pl.DataFrame,
    *,
    cost_bps_values: Sequence[float],
) -> pl.DataFrame:
    rows: list[dict] = []
    for cost_bps in cost_bps_values:
        net = (
            monthly["gross_return"].to_numpy()
            - monthly["turnover"].to_numpy() * float(cost_bps) / 10_000.0
        )
        rows.append(
            {
                "cost_bps": float(cost_bps),
                **{
                    f"model_{key}": value
                    for key, value in performance_statistics(net).items()
                },
            }
        )
    return pl.DataFrame(rows)


def yearly_stability(monthly: pl.DataFrame) -> pl.DataFrame:
    return (
        monthly.with_columns(pl.col("decision_month").dt.year().alias("year"))
        .group_by("year")
        .agg(
            ((1.0 + pl.col("net_return")).product() - 1.0).alias("model_return"),
            ((1.0 + pl.col("benchmark_return")).product() - 1.0).alias(
                "benchmark_return"
            ),
            ((1.0 + pl.col("legacy_return")).product() - 1.0).alias(
                "legacy_return"
            ),
            pl.col("net_return").mean().alias("model_average_month"),
            pl.col("net_return").std().alias("model_monthly_volatility"),
            (pl.col("net_return") > pl.col("legacy_return"))
            .mean()
            .alias("monthly_legacy_beat_rate"),
            pl.len().alias("months"),
        )
        .sort("year")
    )


def meta_walk_forward_selection(
    run_dirs: Mapping[str, Path],
    *,
    horizons: Sequence[int] = (1, 3, 6, 12),
    methods: Sequence[str] = ("classification", "regression", "ranking"),
    top_n_values: Sequence[int] = (5, 10, 20),
    lookback_months: int = 36,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    candidates: dict[str, pl.DataFrame] = {}
    for feature_mode, run_dir in run_dirs.items():
        for method in methods:
            for horizon in horizons:
                path = run_dir / f"{method}_h{horizon:02d}" / "trading_monthly.csv"
                if not path.exists():
                    continue
                monthly = pl.read_csv(path, try_parse_dates=True)
                for top_n in top_n_values:
                    subset = monthly.filter(pl.col("top_n") == top_n).sort(
                        "decision_month"
                    )
                    if not subset.is_empty():
                        candidate_id = (
                            f"{feature_mode}__{method}__h{horizon:02d}__top{top_n:02d}"
                        )
                        candidates[candidate_id] = subset
    if not candidates:
        raise ValueError("No meta-selection candidates were found.")
    common_end = min(frame["decision_month"].max() for frame in candidates.values())
    first_start = max(frame["decision_month"].min() for frame in candidates.values())
    first_selection = date(first_start.year + 1, 1, 1)
    while add_months(first_selection, -lookback_months) < first_start:
        first_selection = date(first_selection.year + 1, 1, 1)

    choice_rows: list[dict] = []
    monthly_parts: list[pl.DataFrame] = []
    selection_date = first_selection
    while selection_date <= common_end:
        history_start = add_months(selection_date, -lookback_months)
        scored: list[tuple[float, str, int]] = []
        for candidate_id, frame in candidates.items():
            history = frame.filter(
                (pl.col("decision_month") >= history_start)
                & (pl.col("decision_month") < selection_date)
            )
            if history.height != lookback_months:
                continue
            sharpe = float(
                performance_statistics(history["net_return"].to_numpy())["sharpe"]
            )
            if np.isfinite(sharpe):
                scored.append((sharpe, candidate_id, history.height))
        if not scored:
            selection_date = date(selection_date.year + 1, 1, 1)
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_sharpe, selected_id, history_count = scored[0]
        evaluation_end = min(date(selection_date.year, 12, 1), common_end)
        evaluation = candidates[selected_id].filter(
            (pl.col("decision_month") >= selection_date)
            & (pl.col("decision_month") <= evaluation_end)
        )
        if not evaluation.is_empty():
            feature_mode, method, horizon_part, top_n_part = selected_id.split("__")
            choice_rows.append(
                {
                    "selection_date": selection_date,
                    "history_start": history_start,
                    "history_end": add_months(selection_date, -1),
                    "lookback_months": history_count,
                    "candidate_count": len(scored),
                    "selected_candidate": selected_id,
                    "selected_feature_mode": feature_mode,
                    "selected_method": method,
                    "selected_horizon": int(horizon_part.removeprefix("h")),
                    "selected_top_n": int(top_n_part.removeprefix("top")),
                    "historical_net_sharpe": selected_sharpe,
                    "evaluation_start": evaluation["decision_month"].min(),
                    "evaluation_end": evaluation["decision_month"].max(),
                    "evaluation_months": evaluation.height,
                }
            )
            monthly_parts.append(
                evaluation.with_columns(
                    pl.lit(selected_id).alias("selected_candidate"),
                    pl.lit(selection_date).alias("selection_date"),
                )
            )
        selection_date = date(selection_date.year + 1, 1, 1)
    if not monthly_parts:
        raise ValueError("The meta-selection protocol produced no evaluation months.")
    monthly = pl.concat(monthly_parts).sort("decision_month")
    summary = pl.DataFrame(
        [
            {
                "strategy": "meta_walk_forward",
                **{
                    f"model_{key}": value
                    for key, value in performance_statistics(
                        monthly["net_return"].to_numpy()
                    ).items()
                },
                **{
                    f"benchmark_{key}": value
                    for key, value in performance_statistics(
                        monthly["benchmark_return"].to_numpy()
                    ).items()
                },
                **{
                    f"legacy_{key}": value
                    for key, value in performance_statistics(
                        monthly["legacy_return"].to_numpy()
                    ).items()
                },
                "start_decision_month": monthly["decision_month"].min(),
                "end_decision_month": monthly["decision_month"].max(),
                "months": monthly.height,
                "lookback_months": lookback_months,
                "candidate_universe_size": len(candidates),
            }
        ]
    )
    return pl.DataFrame(choice_rows), monthly, summary


def holdings_and_concentration(
    predictions: pl.DataFrame,
    *,
    general_path: Path,
    top_n: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    general = (
        pl.read_parquet(general_path)
        .select(
            pl.col("ticker").cast(pl.Utf8),
            pl.coalesce("GicSector", "Sector").fill_null("Unknown").alias("sector"),
            pl.coalesce("GicIndustry", "Industry").fill_null("Unknown").alias(
                "industry"
            ),
        )
        .unique("ticker", keep="last")
    )
    selected_parts: list[pl.DataFrame] = []
    for month in predictions.partition_by("decision_month", maintain_order=True):
        selected_parts.append(
            month.sort(
                ["score", "ticker"],
                descending=[True, False],
            )
            .head(top_n)
            .with_row_index("selection_rank", offset=1)
        )
    holdings = (
        pl.concat(selected_parts)
        .join(general, on="ticker", how="left")
        .with_columns(
            pl.col("sector").fill_null("Unknown"),
            pl.col("industry").fill_null("Unknown"),
            pl.lit(1.0 / top_n).alias("portfolio_weight"),
        )
        .sort(["decision_month", "selection_rank"])
    )
    months = holdings["decision_month"].n_unique()
    ticker = (
        holdings.group_by("ticker")
        .agg(
            pl.len().alias("selected_months"),
            pl.col("portfolio_weight").sum().alias("cumulative_slot_weight"),
            pl.col("score").mean().alias("average_score"),
            pl.col("selection_rank").mean().alias("average_rank"),
            pl.col("sector").first().alias("sector"),
            pl.col("industry").first().alias("industry"),
        )
        .with_columns(
            (pl.col("selected_months") / months).alias("month_selection_rate"),
            (
                pl.col("cumulative_slot_weight")
                / pl.col("cumulative_slot_weight").sum()
            ).alias("share_of_all_portfolio_slots"),
        )
        .sort("selected_months", descending=True)
    )
    sector_monthly = (
        holdings.group_by("decision_month", "sector")
        .agg(pl.col("portfolio_weight").sum().alias("weight"))
        .sort(["decision_month", "weight"], descending=[False, True])
    )
    sector = (
        sector_monthly.group_by("sector")
        .agg(
            (pl.col("weight").sum() / months).alias("average_monthly_weight"),
            pl.col("weight").max().alias("maximum_monthly_weight"),
            (pl.len() / months).alias("active_month_rate"),
        )
        .sort("average_monthly_weight", descending=True)
    )
    concentration = pl.DataFrame(
        [
            {
                "test_months": months,
                "portfolio_slots": holdings.height,
                "unique_tickers": holdings["ticker"].n_unique(),
                "ticker_slot_hhi": float(
                    (ticker["share_of_all_portfolio_slots"] ** 2).sum()
                ),
                "top_5_ticker_slot_share": float(
                    ticker.head(5)["share_of_all_portfolio_slots"].sum()
                ),
                "top_10_ticker_slot_share": float(
                    ticker.head(10)["share_of_all_portfolio_slots"].sum()
                ),
                "average_monthly_max_sector_weight": float(
                    sector_monthly.group_by("decision_month")
                    .agg(pl.col("weight").max().alias("max_weight"))["max_weight"]
                    .mean()
                ),
                "maximum_sector_weight_any_month": float(
                    sector_monthly["weight"].max()
                ),
            }
        ]
    )
    return holdings, ticker, sector, concentration
