from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _import_backtest_api():
    try:
        from alpharank.backtest import (
            BacktestArtifacts,
            BacktestConfig,
            BacktestPhaseArtifacts,
            LearningArtifacts,
            run_backtest_phase,
            run_boosting_backtest,
            run_learning_phase,
        )
        return (
            BacktestArtifacts,
            BacktestConfig,
            BacktestPhaseArtifacts,
            LearningArtifacts,
            run_backtest_phase,
            run_boosting_backtest,
            run_learning_phase,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "alpharank":
            raise

        project_root = Path(__file__).resolve().parent.parent
        print(
            "[Bootstrap] Package 'alpharank' absent pour cet interpreteur.\n"
            f"[Bootstrap] python={sys.executable}\n"
            "[Bootstrap] Tentative d'installation locale: pip install -e ."
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(project_root)],
                check=True,
            )
        except Exception as install_exc:
            raise SystemExit(
                "Echec bootstrap package 'alpharank'.\n"
                f"Interpreteur: {sys.executable}\n"
                "Commande a lancer manuellement:\n"
                f"  {sys.executable} -m pip install -e {project_root}"
            ) from install_exc

        from alpharank.backtest import (
            BacktestArtifacts,
            BacktestConfig,
            BacktestPhaseArtifacts,
            LearningArtifacts,
            run_backtest_phase,
            run_boosting_backtest,
            run_learning_phase,
        )
        return (
            BacktestArtifacts,
            BacktestConfig,
            BacktestPhaseArtifacts,
            LearningArtifacts,
            run_backtest_phase,
            run_boosting_backtest,
            run_learning_phase,
        )


(
    BacktestArtifacts,
    BacktestConfig,
    BacktestPhaseArtifacts,
    LearningArtifacts,
    run_backtest_phase,
    run_boosting_backtest,
    run_learning_phase,
) = _import_backtest_api()


def default_config() -> BacktestConfig:
    project_root = Path(__file__).parent.parent

    return BacktestConfig(
        data_dir=project_root / "data",
        output_dir=project_root / "outputs",
        start_month="2006-01",
        n_folds=10,
        top_n=20,
        prediction_threshold=0.05,
        min_train_months=24,
        missing_feature_threshold=0.35,
        n_optuna_trials=40,
        optuna_lambda_gap=3,
        optuna_startup_trials=30,
        risk_free_rate=0.02,
        random_seed=42,
        verbose=True,
        show_optuna_progress=True,
        optuna_progress_every=1,
    )


def run_learning(config: BacktestConfig | None = None) -> LearningArtifacts:
    cfg = config if config is not None else default_config()
    return run_learning_phase(cfg)


def run_backtest_from_learning(
    learning: LearningArtifacts,
    config: BacktestConfig | None = None,
) -> BacktestPhaseArtifacts:
    cfg = config if config is not None else default_config()
    return run_backtest_phase(cfg, learning)


def run_backtest(config: BacktestConfig | None = None) -> BacktestArtifacts:
    cfg = config if config is not None else default_config()
    return run_boosting_backtest(cfg)


def main() -> None:
    print("[Main] Running full pipeline (learning + backtest + report)...")
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
    print("\n=== Split KPIs (Train / Validation / Test) ===")
    print(artifacts.split_kpis)

    print("\n=== Output Paths ===")
    for key, path in artifacts.output_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
