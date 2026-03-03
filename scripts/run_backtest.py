from __future__ import annotations

from pathlib import Path

from alpharank.backtest import BacktestConfig, BacktestArtifacts, run_boosting_backtest


def default_config() -> BacktestConfig:
    project_root = Path(__file__).parent.parent

    return BacktestConfig(
        data_dir=project_root / "data",
        output_dir=project_root / "outputs",
        start_month="2006-01",
        n_folds=10,
        top_n=20,
        prediction_threshold=0.02,
        min_train_months=24,
        missing_feature_threshold=0.35,
        n_optuna_trials=40,
        optuna_lambda_gap=0.2,
        optuna_startup_trials=12,
        risk_free_rate=0.02,
        random_seed=42,
        verbose=True,
        show_optuna_progress=True,
        optuna_progress_every=1,
    )


def run_backtest(config: BacktestConfig | None = None) -> BacktestArtifacts:
    cfg = config if config is not None else default_config()
    return run_boosting_backtest(cfg)


def main() -> None:
    artifacts = run_backtest()

    print("\n=== Backtest Completed ===")
    print(f"Modeling rows: {artifacts.model_frame.height}")
    print(f"Predictions rows: {artifacts.predictions.height}")
    print(f"Selections rows: {artifacts.selections.height}")
    print(f"Completed folds: {artifacts.fold_metrics.height}")
    print(f"Features used: {len(artifacts.features_used)}")
    if artifacts.dropped_features:
        print(f"Dropped sparse features ({len(artifacts.dropped_features)}): {artifacts.dropped_features}")

    print("\n=== Fold Metrics ===")
    print(artifacts.fold_metrics)

    print("\n=== KPIs (No NA) ===")
    print(artifacts.kpis)

    print("\n=== Output Paths ===")
    for key, path in artifacts.output_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
