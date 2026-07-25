from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from alpharank.backtest.data_loading import find_existing_column
from alpharank.backtest.mlcraft_adapter import ensure_mlcraft_importable
from alpharank.multihorizon.metrics import _expected_calibration_error, _safe_metric


DAILY_RISK_TRADING_DAYS = 252.0


@dataclass
class FittedRiskBooster:
    model: Any
    task_type: str
    features: tuple[str, ...]
    calibrator: IsotonicRegression | None = None
    target_bounds: tuple[float, float] | None = None

    def predict_raw_score(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "classification":
            return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        transformed = np.asarray(self.model.predict(X), dtype=float)
        return np.exp(transformed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw_score(X)
        if self.task_type == "classification" and self.calibrator is not None:
            return np.asarray(self.calibrator.predict(raw), dtype=float)
        return raw


def _daily_risk_statistics(final_price: pl.DataFrame) -> pl.DataFrame:
    close_column = find_existing_column(
        final_price,
        ("adjusted_close", "close", "adj_close"),
    )
    if close_column is None:
        raise ValueError("No supported close column was found.")
    daily = (
        final_price.select(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("date").cast(pl.Date, strict=False),
            pl.col(close_column).cast(pl.Float64).alias("_close"),
        )
        .drop_nulls()
        .sort(["ticker", "date"])
        .with_columns(
            pl.col("_close").shift(1).over("ticker").alias("_previous_close"),
            pl.col("date").shift(1).over("ticker").alias("_previous_date"),
        )
        .with_columns(
            pl.when(
                (pl.col("date") - pl.col("_previous_date"))
                .dt.total_days()
                .is_between(1, 7)
                & (pl.col("_previous_close") > 0.0)
            )
            .then(pl.col("_close") / pl.col("_previous_close") - 1.0)
            .otherwise(None)
            .alias("_daily_return"),
            pl.col("date").dt.truncate("1mo").alias("_risk_month"),
        )
        .filter(pl.col("_daily_return").is_finite())
    )
    return (
        daily.group_by(["ticker", "_risk_month"])
        .agg(
            pl.len().alias("_daily_count"),
            pl.col("_daily_return").sum().alias("_daily_sum"),
            (pl.col("_daily_return") ** 2).sum().alias("_daily_sum_sq"),
            pl.when(pl.col("_daily_return") < 0.0)
            .then(pl.col("_daily_return") ** 2)
            .otherwise(0.0)
            .sum()
            .alias("_daily_downside_sum_sq"),
        )
        .sort(["ticker", "_risk_month"])
    )


def add_daily_forward_risk_targets(
    frame: pl.DataFrame,
    *,
    final_price: pl.DataFrame,
    horizons: Sequence[int] = (1, 3, 6),
    minimum_daily_observations_per_month: int = 10,
) -> pl.DataFrame:
    """Add causal forward realized-volatility and downside targets.

    Each decision at month ``t`` uses daily returns strictly from calendar
    months ``t+1`` through ``t+h``. Every target month must contain at least
    ``minimum_daily_observations_per_month`` valid daily returns.
    """

    monthly = _daily_risk_statistics(final_price)
    result = frame
    for horizon in sorted({int(value) for value in horizons}):
        joined = result
        temporary_columns: list[str] = []
        count_columns: list[str] = []
        sum_columns: list[str] = []
        square_columns: list[str] = []
        downside_columns: list[str] = []
        for step in range(1, horizon + 1):
            suffix = f"_{horizon}m_{step}"
            count = f"_daily_count{suffix}"
            total = f"_daily_sum{suffix}"
            square = f"_daily_sum_sq{suffix}"
            downside = f"_daily_downside_sum_sq{suffix}"
            target_month = f"_risk_month{suffix}"
            lookup = monthly.rename(
                {
                    "_risk_month": target_month,
                    "_daily_count": count,
                    "_daily_sum": total,
                    "_daily_sum_sq": square,
                    "_daily_downside_sum_sq": downside,
                }
            )
            joined = (
                joined.with_columns(
                    pl.col("decision_month")
                    .dt.offset_by(f"{step}mo")
                    .alias(target_month)
                )
                .join(lookup, on=["ticker", target_month], how="left")
            )
            temporary_columns.extend((target_month, count, total, square, downside))
            count_columns.append(count)
            sum_columns.append(total)
            square_columns.append(square)
            downside_columns.append(downside)

        complete = pl.all_horizontal(
            [
                pl.col(column) >= minimum_daily_observations_per_month
                for column in count_columns
            ]
        )
        observation_count = pl.sum_horizontal(count_columns)
        daily_sum = pl.sum_horizontal(sum_columns)
        daily_sum_sq = pl.sum_horizontal(square_columns)
        downside_sum_sq = pl.sum_horizontal(downside_columns)
        sample_variance = (
            daily_sum_sq - daily_sum**2 / observation_count
        ) / (observation_count - 1.0)
        volatility_column = f"future_realized_volatility_{horizon}m"
        downside_column = f"future_daily_downside_{horizon}m"
        result = (
            joined.with_columns(
                pl.when(complete & (observation_count > 1))
                .then(
                    pl.max_horizontal(sample_variance, pl.lit(0.0))
                    .sqrt()
                    * math.sqrt(DAILY_RISK_TRADING_DAYS)
                )
                .otherwise(None)
                .alias(volatility_column),
                pl.when(complete & (observation_count > 0))
                .then(
                    (downside_sum_sq / observation_count).sqrt()
                    * math.sqrt(DAILY_RISK_TRADING_DAYS)
                )
                .otherwise(None)
                .alias(downside_column),
                pl.when(complete).then(observation_count).otherwise(None).alias(
                    f"future_daily_observations_{horizon}m"
                ),
            )
            .drop(temporary_columns)
            .with_columns(
                (
                    pl.col(volatility_column).rank(method="average").over(
                        "decision_month"
                    )
                    / pl.col(volatility_column).count().over("decision_month")
                ).alias(f"future_realized_volatility_rank_{horizon}m"),
                (
                    pl.col(downside_column).rank(method="average").over(
                        "decision_month"
                    )
                    / pl.col(downside_column).count().over("decision_month")
                ).alias(f"future_daily_downside_rank_{horizon}m"),
            )
        )
    return result


def fit_risk_booster(
    *,
    task_type: str,
    target_column: str,
    train_frame: pl.DataFrame,
    validation_frame: pl.DataFrame,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    features: tuple[str, ...],
    seed: int,
    num_boost_round: int,
    positive_threshold: float = 0.80,
) -> FittedRiskBooster:
    if task_type not in {"regression", "classification"}:
        raise ValueError(f"Unsupported risk task_type={task_type!r}.")
    ensure_mlcraft_importable()
    from mlcraft.core.task import TaskSpec
    from mlcraft.models.factory import ModelFactory

    train_target = train_frame[target_column].to_numpy().astype(float)
    validation_target = validation_frame[target_column].to_numpy().astype(float)
    bounds = None
    if task_type == "classification":
        y_train = (train_target >= positive_threshold).astype(np.int8)
        y_validation = (validation_target >= positive_threshold).astype(np.int8)
    else:
        bounds = tuple(np.nanquantile(train_target, [0.01, 0.99]).tolist())
        y_train = np.log(
            np.clip(train_target, bounds[0], bounds[1]) + 1e-8
        )
        y_validation = np.log(
            np.clip(validation_target, bounds[0], bounds[1]) + 1e-8
        )
    model = ModelFactory.create(
        "xgboost",
        task_spec=TaskSpec(task_type=task_type),
        model_params={
            "eta": 0.04,
            "max_depth": 5,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "lambda": 5.0,
            "alpha": 0.2,
            "nthread": 4,
        },
        fit_params={
            "num_boost_round": int(num_boost_round),
            "early_stopping_rounds": 25,
        },
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
    calibrator = None
    if task_type == "classification" and np.unique(y_validation).size == 2:
        raw = model.predict_proba(X_validation)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            raw,
            y_validation,
        )
    return FittedRiskBooster(
        model=model,
        task_type=task_type,
        features=features,
        calibrator=calibrator,
        target_bounds=bounds,
    )


def score_risk_predictions(
    predictions: pl.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
    task_type: str,
    probability_column: str | None = None,
    positive_threshold: float = 0.80,
) -> dict[str, float]:
    target = predictions[target_column].to_numpy().astype(float)
    score = predictions[prediction_column].to_numpy().astype(float)
    monthly_spearman = [
        _safe_metric(
            lambda a, b: spearmanr(a, b).statistic,
            month[target_column].to_numpy(),
            month[prediction_column].to_numpy(),
        )
        for month in predictions.partition_by("decision_month", maintain_order=True)
    ]
    finite_spearman = [value for value in monthly_spearman if np.isfinite(value)]
    metrics = {
        "monthly_spearman": float(np.mean(finite_spearman))
        if finite_spearman
        else float("nan"),
    }
    if task_type == "regression":
        rmse = _safe_metric(
            lambda a, b: mean_squared_error(a, b) ** 0.5,
            target,
            score,
        )
        target_std = float(np.std(target))
        metrics.update(
            rmse=rmse,
            normalized_rmse=rmse / target_std
            if target_std > 0.0
            else float("nan"),
            mae=_safe_metric(mean_absolute_error, target, score),
            r2=_safe_metric(r2_score, target, score),
            target_mean=float(np.mean(target)),
            target_std=target_std,
        )
        return metrics

    label = (target >= positive_threshold).astype(np.int8)
    probability = (
        predictions[probability_column].to_numpy().astype(float)
        if probability_column
        else score
    )
    prevalence = float(label.mean())
    pr_auc = _safe_metric(average_precision_score, label, score)
    metrics.update(
        roc_auc=_safe_metric(roc_auc_score, label, score),
        pr_auc_average_precision=pr_auc,
        pr_auc_lift_vs_prevalence=pr_auc / prevalence
        if prevalence > 0.0
        else float("nan"),
        brier=_safe_metric(brier_score_loss, label, probability),
        log_loss=_safe_metric(
            log_loss,
            label,
            np.clip(probability, 1e-6, 1.0 - 1e-6),
        ),
        expected_calibration_error=_expected_calibration_error(
            label,
            probability,
        ),
        positive_rate=prevalence,
    )
    return metrics


def capped_inverse_risk_weights(
    risk: Sequence[float],
    *,
    maximum_weight: float,
    floor_quantile: float = 0.20,
) -> np.ndarray:
    values = np.asarray(risk, dtype=float)
    if values.size == 0:
        return values
    if maximum_weight * values.size < 1.0 - 1e-12:
        raise ValueError("maximum_weight is infeasible for the number of assets.")
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size == 0:
        base = np.ones(values.size, dtype=float)
    else:
        floor = max(
            float(np.quantile(finite_positive, floor_quantile)),
            1e-8,
        )
        clean = np.where(
            np.isfinite(values) & (values > 0.0),
            np.maximum(values, floor),
            np.nanmedian(finite_positive),
        )
        base = 1.0 / clean
    weights = np.zeros(values.size, dtype=float)
    active = np.ones(values.size, dtype=bool)
    remaining = 1.0
    while np.any(active):
        proposed = remaining * base[active] / base[active].sum()
        active_indices = np.flatnonzero(active)
        capped = proposed > maximum_weight + 1e-12
        if not np.any(capped):
            weights[active_indices] = proposed
            break
        capped_indices = active_indices[capped]
        weights[capped_indices] = maximum_weight
        remaining -= maximum_weight * len(capped_indices)
        active[capped_indices] = False
    return weights / weights.sum()


def constrained_inverse_risk_weights(
    risk: Sequence[float],
    sectors: Sequence[str],
    *,
    maximum_weight: float,
    maximum_sector_weight: float,
    floor_quantile: float = 0.20,
) -> np.ndarray:
    values = np.asarray(risk, dtype=float)
    sector_values = np.asarray(sectors, dtype=object)
    if values.size != sector_values.size:
        raise ValueError("risk and sectors must have the same length.")
    if values.size == 0:
        return values
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size:
        floor = max(
            float(np.quantile(finite_positive, floor_quantile)),
            1e-8,
        )
        clean = np.where(
            np.isfinite(values) & (values > 0.0),
            np.maximum(values, floor),
            np.nanmedian(finite_positive),
        )
        base = 1.0 / clean
    else:
        base = np.ones(values.size, dtype=float)
    weights = np.zeros(values.size, dtype=float)
    remaining = 1.0
    unique_sectors = list(dict.fromkeys(sector_values.tolist()))
    for _ in range(values.size + len(unique_sectors) + 2):
        stock_capacity = maximum_weight - weights
        sector_capacity = {
            sector: maximum_sector_weight
            - float(weights[sector_values == sector].sum())
            for sector in unique_sectors
        }
        active = np.asarray(
            [
                stock_capacity[index] > 1e-12
                and sector_capacity[sector_values[index]] > 1e-12
                for index in range(values.size)
            ]
        )
        if remaining <= 1e-12:
            break
        if not np.any(active):
            raise ValueError("Portfolio constraints are infeasible.")
        proportions = np.zeros(values.size, dtype=float)
        proportions[active] = base[active] / base[active].sum()
        allocation = remaining
        allocation = min(
            allocation,
            *[
                stock_capacity[index] / proportions[index]
                for index in np.flatnonzero(active)
                if proportions[index] > 0.0
            ],
        )
        for sector in unique_sectors:
            sector_share = float(
                proportions[sector_values == sector].sum()
            )
            if sector_share > 0.0:
                allocation = min(
                    allocation,
                    sector_capacity[sector] / sector_share,
                )
        if allocation <= 1e-12:
            raise ValueError("Portfolio constraints cannot absorb remaining weight.")
        weights += allocation * proportions
        remaining -= allocation
    if remaining > 1e-8:
        raise ValueError("Portfolio constraints are infeasible.")
    return weights / weights.sum()


def build_risk_weighted_backtest(
    predictions: pl.DataFrame,
    *,
    general: pl.DataFrame,
    strategy: str,
    top_n: int = 5,
    risk_column: str | None = None,
    maximum_weight: float = 0.30,
    maximum_names_per_sector: int | None = None,
    maximum_sector_weight: float | None = None,
    transaction_cost_bps: float = 10.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a monthly top-alpha portfolio with optional risk sizing.

    The risk prediction never changes the alpha order. The optional sector
    rule only skips lower-ranked names once a sector has reached its name cap.
    """

    sector_map = (
        general.select(
            pl.col("ticker").cast(pl.Utf8),
            pl.coalesce("GicSector", "Sector").fill_null("Unknown").alias(
                "sector"
            ),
        )
        .unique("ticker", keep="last")
    )
    panel = predictions.join(sector_map, on="ticker", how="left").with_columns(
        pl.col("sector").fill_null("Unknown")
    )
    previous_weights: dict[str, float] = {}
    monthly_rows: list[dict] = []
    holding_parts: list[pl.DataFrame] = []
    for month in panel.partition_by("decision_month", maintain_order=True):
        ordered = month.sort("score", descending=True)
        selected_indices: list[int] = []
        sector_counts: dict[str, int] = {}
        for index, sector in enumerate(ordered["sector"].to_list()):
            if (
                maximum_names_per_sector is not None
                and sector_counts.get(sector, 0) >= maximum_names_per_sector
            ):
                continue
            selected_indices.append(index)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected_indices) == top_n:
                break
        if len(selected_indices) < top_n:
            if maximum_names_per_sector is not None:
                raise ValueError(
                    "Not enough names to satisfy maximum_names_per_sector."
                )
            selected_indices = list(range(min(top_n, ordered.height)))
        selected = ordered[selected_indices]
        if selected.is_empty():
            continue
        if risk_column is None:
            weights = np.full(selected.height, 1.0 / selected.height)
        elif maximum_sector_weight is not None:
            weights = constrained_inverse_risk_weights(
                selected[risk_column].to_numpy(),
                selected["sector"].to_list(),
                maximum_weight=maximum_weight,
                maximum_sector_weight=maximum_sector_weight,
            )
        else:
            weights = capped_inverse_risk_weights(
                selected[risk_column].to_numpy(),
                maximum_weight=maximum_weight,
            )
        selected = selected.with_columns(
            pl.Series("portfolio_weight", weights),
            pl.int_range(1, selected.height + 1, eager=True).alias(
                "selection_rank"
            ),
            pl.lit(strategy).alias("strategy"),
        )
        tickers = selected["ticker"].to_list()
        current_weights = dict(zip(tickers, weights, strict=True))
        names = set(previous_weights) | set(current_weights)
        turnover = (
            0.5
            * sum(
                abs(
                    current_weights.get(name, 0.0)
                    - previous_weights.get(name, 0.0)
                )
                for name in names
            )
            if previous_weights
            else 1.0
        )
        gross_return = float(
            np.dot(
                weights,
                selected["future_return_1m"].to_numpy().astype(float),
            )
        )
        cost = turnover * transaction_cost_bps / 10_000.0
        sector_weights = (
            selected.group_by("sector")
            .agg(pl.col("portfolio_weight").sum().alias("weight"))
            .sort("weight", descending=True)
        )
        monthly_rows.append(
            {
                "strategy": strategy,
                "decision_month": selected["decision_month"][0],
                "holding_month": selected["decision_month"][0],
                "n_positions": selected.height,
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "benchmark_return": float(
                    selected["benchmark_future_return_1m"][0]
                ),
                "turnover": turnover,
                "transaction_cost": cost,
                "maximum_position_weight": float(np.max(weights)),
                "maximum_sector_weight": float(sector_weights["weight"].max()),
                "sector_count": sector_weights.height,
            }
        )
        holding_parts.append(selected)
        previous_weights = current_weights
    monthly = (
        pl.DataFrame(monthly_rows)
        .with_columns(
            pl.col("decision_month").dt.offset_by("1mo").alias("holding_month")
        )
        .sort("decision_month")
    )
    holdings = (
        pl.concat(holding_parts)
        .with_columns(
            pl.col("decision_month").dt.offset_by("1mo").alias("holding_month")
        )
        .sort(["decision_month", "selection_rank"])
    )
    return monthly, holdings
