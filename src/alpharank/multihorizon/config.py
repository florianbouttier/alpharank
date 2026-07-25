from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class MultiHorizonConfig:
    """Configuration for the multi-horizon screening and validation pipeline."""

    data_dir: Path
    legacy_detailed_returns_path: Path
    legacy_monthly_returns_path: Path
    output_dir: Path = Path("outputs")
    run_dir: Path | None = None
    horizons: Tuple[int, ...] = (1, 3, 6, 12, 24, 36)
    methods: Tuple[str, ...] = ("classification", "regression", "ranking", "teacher")
    start_month: str = "2000-01"
    min_train_months: int = 72
    validation_months: int = 24
    test_months: int = 12
    step_months: int = 12
    include_partial_test_window: bool = False
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
    excluded_tickers: Tuple[str, ...] = ("SII.US", "CBE.US", "TIE.US", "CPWR.US")
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
