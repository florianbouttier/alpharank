# %%
import argparse
import contextlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from alpharank.data.contracts.ticker_integrity import (
    DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
)
from alpharank.data.price_eligibility import (
    STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY,
)
from alpharank.observability import configure_run_logging
from alpharank.production.legacy_pipeline import (
    INPUT_PACKAGE_FILENAMES as INPUT_PACKAGE_FILENAMES,
)
from alpharank.production.legacy_pipeline import (
    _copy_snapshot_file as _copy_snapshot_file,
)
from alpharank.production.legacy_pipeline import (
    _input_files as _input_files,
)
from alpharank.production.legacy_pipeline import (
    _manifest_extra_context as _manifest_extra_context,
)
from alpharank.production.legacy_pipeline import (
    _resolve_open_source_output_by_run_id as _resolve_open_source_output_by_run_id,
)
from alpharank.production.legacy_pipeline import (
    _simulate_historical_legacy_common as _simulate_historical_legacy_common,
)
from alpharank.production.legacy_pipeline import (
    _snapshot_input_package as _snapshot_input_package,
)
from alpharank.production.legacy_pipeline import (
    normalize_year_month_to_timestamp as normalize_year_month_to_timestamp,
)
from alpharank.production.legacy_pipeline import (
    run_pipeline,
)
from alpharank.strategy.legacy_valuation import NO_SEC_FUNDAMENTALS_POLICY_ID


def main(
    *,
    n_trials: int = 30,
    n_jobs: int = 1,
    first_date: str = "2010-01",
    data_dir: str | Path | None = None,
    open_source_run_id: str | None = None,
    output_dir: str | Path | None = None,
    checkpoints_dir: str | Path | None = "outputs/checkpoints",
    final_price_path: str | Path | None = None,
    sp500_price_path: str | Path | None = None,
    methodology_manifest: str | Path | None = None,
    ticker_exclusion_registry: str | Path | None = DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
    price_eligibility_policy_id: str = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.policy_id,
    minimum_monthly_price_observations: int = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_observations,
    minimum_monthly_median_dollar_volume: float = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_median_dollar_volume,
    maximum_monthly_ohlc_violation_rate: float = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.maximum_ohlc_violation_rate,
    fundamental_eligibility_policy_id: str = NO_SEC_FUNDAMENTALS_POLICY_ID,
) -> None:
    checkpoints_dir = (
        Path(checkpoints_dir).expanduser().resolve() if checkpoints_dir is not None else None
    )
    data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    final_price_path = Path(final_price_path).expanduser().resolve() if final_price_path else None
    sp500_price_path = Path(sp500_price_path).expanduser().resolve() if sp500_price_path else None
    methodology_manifest = (
        Path(methodology_manifest).expanduser().resolve() if methodology_manifest else None
    )
    ticker_exclusion_registry = (
        Path(ticker_exclusion_registry).expanduser().resolve()
        if ticker_exclusion_registry
        else None
    )

    out = run_pipeline(
        n_trials=n_trials,
        n_jobs=n_jobs,
        first_date=first_date,
        data_dir=data_dir,
        open_source_run_id=open_source_run_id,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        final_price_path=final_price_path,
        sp500_price_path=sp500_price_path,
        methodology_manifest=methodology_manifest,
        ticker_exclusion_registry=ticker_exclusion_registry,
        price_eligibility_policy_id=price_eligibility_policy_id,
        minimum_monthly_price_observations=minimum_monthly_price_observations,
        minimum_monthly_median_dollar_volume=(
            minimum_monthly_median_dollar_volume
        ),
        maximum_monthly_ohlc_violation_rate=(
            maximum_monthly_ohlc_violation_rate
        ),
        fundamental_eligibility_policy_id=fundamental_eligibility_policy_id,
    )
    print("Artifacts:")
    for k, v in out.artifacts.items():
        print(f"  {k}: {v}")


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "default"


def _default_log_stem(*, data_dir: str | None, open_source_run_id: str | None) -> str:
    if open_source_run_id:
        return f"run_legacy_open_source_{_slug(open_source_run_id)}"
    if data_dir and "open_source" in data_dir:
        return "run_legacy_open_source"
    return "run_legacy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the legacy monthly portfolio pipeline.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--first-date", default="2010-01")
    parser.add_argument("--data-dir")
    parser.add_argument("--open-source-run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Skip optional diagnostic checkpoints; canonical run artifacts remain unchanged.",
    )
    parser.add_argument("--final-price-path")
    parser.add_argument("--sp500-price-path")
    parser.add_argument(
        "--methodology-manifest",
        help=(
            "Sealed causal-v2 manifest; requires --data-dir to be that package's "
            "input_snapshot and emits next-open/cost-scenario artifacts."
        ),
    )
    ticker_registry_group = parser.add_mutually_exclusive_group()
    ticker_registry_group.add_argument(
        "--ticker-exclusion-registry",
        default=str(DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY),
        help="Versioned full-trajectory exclusion registry (enabled by default).",
    )
    ticker_registry_group.add_argument(
        "--no-ticker-exclusion-registry",
        action="store_true",
        help="Disable the registry for an explicitly labelled compatibility replay.",
    )
    parser.add_argument(
        "--price-eligibility-policy-id",
        default=STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.policy_id,
    )
    parser.add_argument(
        "--minimum-monthly-price-observations",
        type=int,
        default=STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_observations,
    )
    parser.add_argument(
        "--minimum-monthly-median-dollar-volume",
        type=float,
        default=STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_median_dollar_volume,
    )
    parser.add_argument(
        "--maximum-monthly-ohlc-violation-rate",
        type=float,
        default=STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.maximum_ohlc_violation_rate,
    )
    parser.add_argument(
        "--fundamental-eligibility-policy-id",
        choices=("legacy_pe_market_cap_v1", "no_sec_fundamentals_v1"),
        default=NO_SEC_FUNDAMENTALS_POLICY_ID,
    )
    parser.add_argument("--log-dir", default="logs/legacy_runs")
    parser.add_argument("--no-log", action="store_true", help="Disable automatic CLI log capture.")
    return parser.parse_args()


def _run_cli() -> None:
    args = _parse_args()
    kwargs = {
        "n_trials": args.n_trials,
        "n_jobs": args.n_jobs,
        "first_date": args.first_date,
        "data_dir": args.data_dir,
        "open_source_run_id": args.open_source_run_id,
        "output_dir": args.output_dir,
        "checkpoints_dir": None if args.no_checkpoints else args.checkpoints_dir,
        "final_price_path": args.final_price_path,
        "sp500_price_path": args.sp500_price_path,
        "methodology_manifest": args.methodology_manifest,
        "ticker_exclusion_registry": (
            None
            if args.no_ticker_exclusion_registry
            else args.ticker_exclusion_registry
        ),
        "price_eligibility_policy_id": args.price_eligibility_policy_id,
        "minimum_monthly_price_observations": args.minimum_monthly_price_observations,
        "minimum_monthly_median_dollar_volume": (
            args.minimum_monthly_median_dollar_volume
        ),
        "maximum_monthly_ohlc_violation_rate": (
            args.maximum_monthly_ohlc_violation_rate
        ),
        "fundamental_eligibility_policy_id": args.fundamental_eligibility_policy_id,
    }
    if args.no_log:
        configure_run_logging()
        main(**kwargs)
        return

    project_root = Path(__file__).parent.parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _default_log_stem(data_dir=args.data_dir, open_source_run_id=args.open_source_run_id)
    log_path = log_dir / f"{stem}_{timestamp}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = _Tee(sys.stdout, log_file)
        stderr = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            configure_run_logging()
            print(f"Log file: {log_path}")
            print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Arguments: {kwargs}")
            main(**kwargs)
            print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    _run_cli()
