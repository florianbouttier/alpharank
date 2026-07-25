#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.multihorizon import MultiHorizonConfig, run_multihorizon_research


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
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--n-trials", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--shap-sample-per-fold", type=int, default=200)
    parser.add_argument("--save-research-frame", action="store_true")
    args = parser.parse_args()
    config = MultiHorizonConfig(
        data_dir=args.data_dir,
        legacy_detailed_returns_path=args.legacy_detailed,
        legacy_monthly_returns_path=args.legacy_monthly,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        horizons=args.horizons,
        methods=args.methods,
        start_month=args.start_month,
        min_train_months=args.min_train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        max_windows=args.max_windows,
        n_trials=args.n_trials,
        num_boost_round=args.num_boost_round,
        shap_sample_per_fold=args.shap_sample_per_fold,
        save_research_frame=args.save_research_frame,
    )
    run_dir = run_multihorizon_research(config)
    print(run_dir)


if __name__ == "__main__":
    main()
