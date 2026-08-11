#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.data.ticker_integrity import load_ticker_exclusion_registry
from alpharank.multihorizon import MultiHorizonConfig, run_multihorizon_research
from alpharank.multihorizon.config import (
    LATEST_COMMON_COMPARISON_PROFILE,
    LATEST_COMMON_COMPARISON_PROFILE_NAME,
)


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-aware multi-horizon boosting research.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--legacy-detailed", type=Path, required=True)
    parser.add_argument("--legacy-monthly", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--latest-common-comparison-profile",
        action="store_true",
        help=(
            "Apply the versioned causal Legacy-EMA comparison settings. "
            "Requires --score-only-end-month."
        ),
    )
    parser.add_argument("--horizons", type=_integers, default=(1, 3, 6, 12, 24, 36))
    parser.add_argument(
        "--methods",
        type=_strings,
        default=("classification", "regression", "ranking", "teacher"),
    )
    parser.add_argument("--start-month", default="2000-01")
    parser.add_argument("--min-train-months", type=int, default=72)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--include-partial-test-window", action="store_true")
    parser.add_argument(
        "--score-only-end-month",
        help=(
            "Final decision month (YYYY-MM) to score using the causal outer "
            "fold even when its learning target is not mature. Model metrics "
            "remain restricted to mature targets."
        ),
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--n-trials", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument(
        "--shap-sample-per-fold",
        type=int,
        default=200,
        help="Maximum SHAP rows per test fold; use 0 to explain every test row.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=(
            "broad",
            "legacy_winners_pit_ema_only",
            "legacy_winners_pit_ema_plus",
            "legacy_active_oracle",
        ),
        default="broad",
    )
    parser.add_argument(
        "--excluded-tickers",
        type=_strings,
        default=load_ticker_exclusion_registry().excluded_tickers,
        help="Comma-separated tickers excluded for documented price integrity.",
    )
    parser.add_argument(
        "--minimum-monthly-price-observations",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--minimum-monthly-median-dollar-volume",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-monthly-ohlc-violation-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument("--save-research-frame", action="store_true")
    args = parser.parse_args()
    run_profile = None
    if args.latest_common_comparison_profile:
        if not args.score_only_end_month:
            parser.error(
                "--latest-common-comparison-profile requires "
                "--score-only-end-month YYYY-MM"
            )
        for key, value in LATEST_COMMON_COMPARISON_PROFILE.items():
            setattr(args, key, value)
        run_profile = LATEST_COMMON_COMPARISON_PROFILE_NAME
    config = MultiHorizonConfig(
        data_dir=args.data_dir,
        legacy_detailed_returns_path=args.legacy_detailed,
        legacy_monthly_returns_path=args.legacy_monthly,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        run_profile=run_profile,
        horizons=args.horizons,
        methods=args.methods,
        start_month=args.start_month,
        min_train_months=args.min_train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        include_partial_test_window=args.include_partial_test_window,
        score_only_end_month=args.score_only_end_month,
        max_windows=args.max_windows,
        n_trials=args.n_trials,
        num_boost_round=args.num_boost_round,
        shap_sample_per_fold=args.shap_sample_per_fold,
        feature_mode=args.feature_mode,
        excluded_tickers=args.excluded_tickers,
        minimum_monthly_price_observations=(
            args.minimum_monthly_price_observations
        ),
        minimum_monthly_median_dollar_volume=(
            args.minimum_monthly_median_dollar_volume
        ),
        maximum_monthly_ohlc_violation_rate=(
            args.maximum_monthly_ohlc_violation_rate
        ),
        save_research_frame=args.save_research_frame,
    )
    run_dir = run_multihorizon_research(config)
    print(run_dir)


if __name__ == "__main__":
    main()
