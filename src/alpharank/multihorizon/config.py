from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Tuple

from alpharank.data.ticker_integrity import load_ticker_exclusion_registry
from alpharank.data.price_eligibility import (
    STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY,
    monthly_price_eligibility_policy,
)


DEFAULT_HISTORICAL_EXCLUDED_TICKERS = (
    load_ticker_exclusion_registry().excluded_tickers
)

LATEST_COMMON_COMPARISON_PROFILE_NAME = "legacy_ema_latest_common_v1"
LATEST_COMMON_COMPARISON_PROFILE = {
    "horizons": (6,),
    "methods": ("classification",),
    "start_month": "2005-01",
    "min_train_months": 62,
    "validation_months": 6,
    "test_months": 12,
    "step_months": 12,
    "include_partial_test_window": True,
    "n_trials": 0,
    "num_boost_round": 100,
    "shap_sample_per_fold": 0,
    "feature_mode": "legacy_winners_pit_ema_only",
    "price_eligibility_policy_id": (
        STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.policy_id
    ),
    "minimum_monthly_price_observations": (
        STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_observations
    ),
    "minimum_monthly_median_dollar_volume": (
        STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_median_dollar_volume
    ),
    "maximum_monthly_ohlc_violation_rate": (
        STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.maximum_ohlc_violation_rate
    ),
    "random_seed": 42,
}


def validate_latest_common_comparison_profile(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the versioned methodology contract used by the public comparison."""

    mismatches: dict[str, dict[str, object]] = {}
    for key, expected in LATEST_COMMON_COMPARISON_PROFILE.items():
        observed = config.get(key)
        if isinstance(expected, tuple) and isinstance(observed, list):
            observed = tuple(observed)
        if observed != expected:
            mismatches[key] = {"expected": expected, "observed": observed}
    if not config.get("score_only_end_month"):
        mismatches["score_only_end_month"] = {
            "expected": "a YYYY-MM decision cutoff",
            "observed": config.get("score_only_end_month"),
        }
    return {
        "name": LATEST_COMMON_COMPARISON_PROFILE_NAME,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


@dataclass(frozen=True)
class MultiHorizonConfig:
    """Configuration for the multi-horizon screening and validation pipeline."""

    data_dir: Path
    legacy_detailed_returns_path: Path
    legacy_monthly_returns_path: Path
    output_dir: Path = Path("outputs")
    run_dir: Path | None = None
    methodology_manifest: Path | None = None
    run_profile: str | None = None
    horizons: Tuple[int, ...] = (1, 3, 6, 12, 24, 36)
    methods: Tuple[str, ...] = ("classification", "regression", "ranking", "teacher")
    start_month: str = "2000-01"
    min_train_months: int = 72
    validation_months: int = 24
    test_months: int = 12
    step_months: int = 12
    include_partial_test_window: bool = False
    score_only_end_month: str | None = None
    max_windows: int | None = None
    missing_feature_threshold: float = 0.35
    target_clip_quantiles: Tuple[float, float] = (0.01, 0.99)
    positive_quantile: float = 0.90
    top_n_values: Tuple[int, ...] = (5, 10, 20)
    n_trials: int = 0
    inner_cpcv_groups: int = 4
    inner_test_groups: int = 1
    shap_sample_per_fold: int = 200
    shap_top_features: int = 30
    save_research_frame: bool = False
    feature_mode: str = "broad"
    excluded_tickers: Tuple[str, ...] = DEFAULT_HISTORICAL_EXCLUDED_TICKERS
    price_eligibility_policy_id: str = "custom"
    minimum_monthly_price_observations: int = 1
    minimum_monthly_median_dollar_volume: float = 0.0
    maximum_monthly_ohlc_violation_rate: float = 1.0
    random_seed: int = 42
    num_boost_round: int = 160
    verbose: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(sorted({int(value) for value in self.horizons}))
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("horizons must contain positive month counts.")
        object.__setattr__(self, "horizons", horizons)
        allowed_methods = {"classification", "regression", "ranking", "teacher"}
        methods = tuple(dict.fromkeys(str(value).lower() for value in self.methods))
        unknown = sorted(set(methods) - allowed_methods)
        if unknown:
            raise ValueError(f"Unsupported methods: {unknown}")
        object.__setattr__(self, "methods", methods)
        if self.min_train_months < 36:
            raise ValueError("min_train_months must be at least 36.")
        if self.validation_months < 6:
            raise ValueError("validation_months must be at least 6.")
        if self.test_months < 1 or self.step_months < 1:
            raise ValueError("test_months and step_months must be positive.")
        if self.score_only_end_month is not None:
            try:
                normalized = date.fromisoformat(
                    f"{self.score_only_end_month[:7]}-01"
                ).strftime("%Y-%m")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "score_only_end_month must use YYYY-MM."
                ) from exc
            if normalized != self.score_only_end_month:
                raise ValueError("score_only_end_month must use YYYY-MM.")
        if self.shap_sample_per_fold < 0:
            raise ValueError("shap_sample_per_fold cannot be negative; use 0 for all rows.")
        if not 0.0 < self.positive_quantile < 1.0:
            raise ValueError("positive_quantile must be strictly between 0 and 1.")
        allowed_feature_modes = {
            "broad",
            "legacy_winners_pit_ema_only",
            "legacy_winners_pit_ema_plus",
            "legacy_active_oracle",
        }
        normalized_feature_mode = str(self.feature_mode).lower()
        if normalized_feature_mode not in allowed_feature_modes:
            raise ValueError(
                f"Unsupported feature_mode={self.feature_mode!r}; "
                f"expected one of {sorted(allowed_feature_modes)}."
            )
        object.__setattr__(self, "feature_mode", normalized_feature_mode)
        if self.minimum_monthly_price_observations < 1:
            raise ValueError("minimum_monthly_price_observations must be positive.")
        if self.minimum_monthly_median_dollar_volume < 0.0:
            raise ValueError(
                "minimum_monthly_median_dollar_volume cannot be negative."
            )
        if not 0.0 <= self.maximum_monthly_ohlc_violation_rate <= 1.0:
            raise ValueError(
                "maximum_monthly_ohlc_violation_rate must be between 0 and 1."
            )
        monthly_price_eligibility_policy(
            policy_id=self.price_eligibility_policy_id,
            minimum_observations=self.minimum_monthly_price_observations,
            minimum_median_dollar_volume=(
                self.minimum_monthly_median_dollar_volume
            ),
            maximum_ohlc_violation_rate=(
                self.maximum_monthly_ohlc_violation_rate
            ),
        )
