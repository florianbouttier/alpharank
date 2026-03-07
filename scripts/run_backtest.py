from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpharank.backtest import BacktestConfig, run_boosting_backtest


def default_config() -> BacktestConfig:
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
    )


def main() -> None:
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
