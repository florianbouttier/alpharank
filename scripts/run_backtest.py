from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from alpharank.backtest import BacktestConfig


def _missing_runtime_modules() -> list[str]:
    required_modules = {
        "alpharank": "the AlphaRank package itself",
        "polars": "the polars dataframe engine used by the backtest pipeline",
        "pyarrow": "the parquet/arrow runtime used alongside polars",
    }
    return [name for name in required_modules if importlib.util.find_spec(name) is None]


def _runtime_error_message(missing_modules: list[str]) -> str:
    missing = ", ".join(missing_modules)
    python_cmd = Path(sys.executable).name or "python"
    lines = [
        f"Missing runtime dependency in the current Python environment: {missing}.",
        "",
        "This script expects AlphaRank to be installed in the same environment as the Jupyter kernel.",
        "From the repo root, install the project and its dependencies with:",
        f"  {python_cmd} -m pip install -e .",
        "",
        "If you use the recommended conda environment instead:",
        "  bash scripts/setup_conda_env.sh alpharank",
        "  conda activate alpharank",
        "  python -m ipykernel install --user --name alpharank --display-name \"Python (alpharank)\"",
        "",
        "Then switch your notebook kernel to \"Python (alpharank)\" and rerun the cell.",
    ]
    return "\n".join(lines)


def _load_backtest_api():
    missing_modules = _missing_runtime_modules()
    if missing_modules:
        raise RuntimeError(_runtime_error_message(missing_modules))
    try:
        from alpharank.backtest import (
            BacktestConfig,
            FundamentalFeatureConfig,
            TechnicalFeatureConfig,
            run_boosting_backtest,
        )
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown module"
        raise RuntimeError(_runtime_error_message([missing_name])) from exc
    return BacktestConfig, FundamentalFeatureConfig, TechnicalFeatureConfig, run_boosting_backtest


def default_config() -> "BacktestConfig":
    BacktestConfig, FundamentalFeatureConfig, TechnicalFeatureConfig, _ = _load_backtest_api()
    return BacktestConfig(
        data_dir=PROJECT_ROOT / "data",
        output_dir=PROJECT_ROOT / "outputs",
        start_month="2000-01",
        n_folds=7,
        top_n=30,
        outperformance_threshold=0.15,
        min_train_months=24,
        missing_feature_threshold=0.05,
        n_optuna_trials=20,
        optuna_lambda_gap=5,
        optuna_startup_trials=20,
        risk_free_rate=0.02,
        random_seed=42,
        verbose=True,
        show_optuna_progress=True,
        optuna_progress_every=10,
        technical_feature_config=TechnicalFeatureConfig(
            roc_windows=(1, 3, 6, 12),
            ema_pairs=((2, 6), (3, 6), (3, 12), (6, 12), (6, 18), (12, 24)),
            price_to_ema_spans=(3, 6, 12, 24),
            rsi_windows=(3, 6, 12, 24),
            rsi_ratio_pairs=((3, 12), (6, 24)),
            bollinger_windows=(6, 12),
            stochastic_windows=((6, 3), (12, 3)),
            range_windows=(6, 12),
            volatility_windows=(3, 6, 12),
            volatility_ratio_pairs=((3, 12), (6, 12)),
        ),
        fundamental_feature_config=FundamentalFeatureConfig(
            quarterly_growth_lags=(1, 4, 12),
        ),
    )


def main() -> None:
    _, _, _, run_boosting_backtest = _load_backtest_api()
    print("[Main] Running overperformance backtest vs benchmark...")
    artifacts = run_boosting_backtest(default_config())

    print("\n=== Backtest Completed ===")
    print(f"Modeling rows: {artifacts.model_frame.height}")
    print(f"Predictions rows: {artifacts.predictions.height}")
    print(f"Selections rows: {artifacts.selections.height}")
    print(f"Completed folds: {artifacts.fold_metrics.height}")
    print(f"Features used: {len(artifacts.features_used)}")
    print(f"Report: {artifacts.output_paths['report_html']}")
    print(f"Backtest audit report: {artifacts.output_paths['backtest_audit_report']}")
    if "shap_global_report" in artifacts.output_paths:
        print(f"SHAP report: {artifacts.output_paths['shap_global_report']}")
    print(f"Fold index: {artifacts.output_paths['fold_index']}")
    print(f"Debug predictions (long): {artifacts.output_paths['debug_predictions_long']}")
    print(f"Debug predictions (full): {artifacts.output_paths['debug_predictions_full']}")


if __name__ == "__main__":
    main()
