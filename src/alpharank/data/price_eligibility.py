"""Shared point-in-time monthly price-quality and liquidity eligibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import polars as pl


STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY_ID = "monthly_price_eligibility_v1"


@dataclass(frozen=True)
class MonthlyPriceEligibilityPolicy:
    policy_id: str
    minimum_observations: int
    minimum_median_dollar_volume: float
    maximum_ohlc_violation_rate: float

    def __post_init__(self) -> None:
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive.")
        if self.minimum_median_dollar_volume < 0.0:
            raise ValueError("minimum_median_dollar_volume cannot be negative.")
        if not 0.0 <= self.maximum_ohlc_violation_rate <= 1.0:
            raise ValueError(
                "maximum_ohlc_violation_rate must be between zero and one."
            )

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY = MonthlyPriceEligibilityPolicy(
    policy_id=STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY_ID,
    minimum_observations=10,
    minimum_median_dollar_volume=1_000_000.0,
    maximum_ohlc_violation_rate=0.05,
)


def monthly_price_eligibility_policy(
    *,
    policy_id: str,
    minimum_observations: int,
    minimum_median_dollar_volume: float,
    maximum_ohlc_violation_rate: float,
) -> MonthlyPriceEligibilityPolicy:
    policy = MonthlyPriceEligibilityPolicy(
        policy_id=str(policy_id),
        minimum_observations=int(minimum_observations),
        minimum_median_dollar_volume=float(minimum_median_dollar_volume),
        maximum_ohlc_violation_rate=float(maximum_ohlc_violation_rate),
    )
    if (
        policy.policy_id == STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY_ID
        and policy != STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY
    ):
        raise ValueError(
            f"{STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY_ID} requires exactly "
            "10 observations, USD 1,000,000 median dollar volume, and a 5% "
            "maximum OHLC violation rate. Use policy_id='custom' for overrides."
        )
    return policy


def price_eligibility_policy_from_manifest(
    manifest: Mapping[str, Any],
) -> MonthlyPriceEligibilityPolicy:
    for section_name in ("run_config", "config"):
        section = manifest.get(section_name)
        if not isinstance(section, Mapping):
            continue
        required = {
            "price_eligibility_policy_id",
            "minimum_monthly_price_observations",
            "minimum_monthly_median_dollar_volume",
            "maximum_monthly_ohlc_violation_rate",
        }
        if required <= set(section):
            return monthly_price_eligibility_policy(
                policy_id=str(section["price_eligibility_policy_id"]),
                minimum_observations=int(
                    section["minimum_monthly_price_observations"]
                ),
                minimum_median_dollar_volume=float(
                    section["minimum_monthly_median_dollar_volume"]
                ),
                maximum_ohlc_violation_rate=float(
                    section["maximum_monthly_ohlc_violation_rate"]
                ),
            )
    raise ValueError("Manifest does not expose a complete price eligibility policy.")


def build_monthly_price_eligibility(
    final_price: pl.DataFrame,
    *,
    policy: MonthlyPriceEligibilityPolicy,
) -> pl.DataFrame:
    """Build causal ticker-month eligibility using only rows observed in that month."""

    price_column = next(
        (
            column
            for column in ("close", "adjusted_close", "adj_close")
            if column in final_price.columns
        ),
        None,
    )
    if price_column is None:
        raise ValueError("No supported price column was found.")

    required_ohlc = {"open", "high", "low", "close"}
    if required_ohlc <= set(final_price.columns):
        ohlc_violation = (
            (pl.col("high") < pl.max_horizontal("open", "close", "low"))
            | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
            | (pl.col("high") < pl.col("low"))
            | (pl.col("open") <= 0.0)
            | (pl.col("close") <= 0.0)
        ).fill_null(True)
    else:
        ohlc_violation = pl.lit(False)

    volume = (
        pl.col("volume").cast(pl.Float64)
        if "volume" in final_price.columns
        else pl.lit(None).cast(pl.Float64)
    )
    return (
        final_price.select(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("date").cast(pl.Date, strict=False).alias("date"),
            pl.col(price_column).cast(pl.Float64).alias("_price"),
            volume.alias("_volume"),
            ohlc_violation.alias("_ohlc_violation"),
        )
        .with_columns(pl.col("date").dt.truncate("1mo").alias("decision_month"))
        .group_by(["ticker", "decision_month"])
        .agg(
            pl.col("_price").is_not_null().sum().alias("price_observations"),
            (pl.col("_price") * pl.col("_volume"))
            .drop_nulls()
            .median()
            .alias("median_dollar_volume"),
            pl.col("_ohlc_violation").mean().alias("ohlc_violation_rate"),
        )
        .with_columns(
            (
                (pl.col("price_observations") >= policy.minimum_observations)
                & (
                    pl.col("median_dollar_volume").fill_null(0.0)
                    >= policy.minimum_median_dollar_volume
                )
                & (
                    pl.col("ohlc_violation_rate").fill_null(1.0)
                    <= policy.maximum_ohlc_violation_rate
                )
            ).alias("price_eligible")
        )
        .sort(["decision_month", "ticker"])
    )
