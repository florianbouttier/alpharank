from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.explainability import ShapFoldExplanation, generate_global_shap_report_pdf
from alpharank.backtest.reporting import (
    build_backtest_fold_focus_table,
    build_optimization_focus_table,
    save_learning_curve,
    save_lift_curve,
    save_optuna_visualizations,
    write_backtest_audit_report,
    write_html_report,
)


def test_save_learning_curve_supports_auc_and_logloss(tmp_path: Path) -> None:
    path = tmp_path / "learning_curve.png"
    evals_result = {
        "validation_0": {
            "auc": [0.55, 0.63, 0.71],
            "logloss": [0.69, 0.61, 0.57],
        },
        "validation_1": {
            "auc": [0.54, 0.60, 0.66],
            "logloss": [0.70, 0.65, 0.62],
        },
    }

    out = save_learning_curve(evals_result=evals_result, path=path, fold_label="fold_01")

    assert out == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_optuna_visualizations_excludes_requested_plots(tmp_path: Path, monkeypatch) -> None:
    fake_optuna = ModuleType("optuna")
    fake_viz = ModuleType("optuna.visualization")
    created = []

    def _plotter(_study):
        class _Figure:
            def write_html(self, path: str, include_plotlyjs: str, full_html: bool) -> None:
                created.append(Path(path).name)
                Path(path).write_text("ok", encoding="utf-8")

        return _Figure()

    for fn_name in [
        "plot_optimization_history",
        "plot_slice",
        "plot_param_importances",
        "plot_timeline",
        "plot_terminator_improvement",
        "plot_parallel_coordinate",
        "plot_contour",
        "plot_edf",
        "plot_intermediate_values",
        "plot_rank",
    ]:
        setattr(fake_viz, fn_name, _plotter)

    fake_optuna.visualization = fake_viz
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setitem(sys.modules, "optuna.visualization", fake_viz)

    paths = save_optuna_visualizations(study=object(), out_dir=tmp_path, fold_label="fold_01")

    assert "optuna_optimization_history" in paths
    assert "optuna_slice" in paths
    assert "optuna_param_importances" in paths
    assert "optuna_timeline" in paths
    assert "optuna_terminator_improvement" in paths
    assert "optuna_parallel_coordinate" not in paths
    assert "optuna_contour" not in paths
    assert "optuna_edf" not in paths
    assert "optuna_intermediate_values" not in paths
    assert "optuna_rank" not in paths
    assert all("parallel_coordinate" not in name for name in created)
    assert all("contour" not in name for name in created)


def test_write_html_report_omits_per_fold_section(tmp_path: Path) -> None:
    report_path = tmp_path / "training_report.html"
    table = pl.DataFrame({"metric": ["a"], "value": [1.0]})
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"fake")
    optuna_path = tmp_path / "fold_01_optuna_slice.html"
    optuna_path.write_text("<html></html>", encoding="utf-8")
    lift_path = tmp_path / "fold_01_validation_lift_curve.png"
    lift_path.write_bytes(b"fake")

    write_html_report(
        title="Test report",
        output_path=report_path,
        backtest_kpis=table,
        split_kpis=table,
        fold_metrics=table,
        best_params=table,
        global_assets={"chart": image_path},
        fold_assets=[
            {
                "__label__": "fold_01",
                "learning_curve": image_path,
                "optuna_slice": optuna_path,
                "lift_validation": lift_path,
            }
        ],
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Optimization Metric Focus" in content
    assert "train_val_gap_abs" in content
    assert "test_penalty" in content
    assert "Global Visualizations" in content
    assert "Per-Fold Analysis" not in content
    assert "Optuna Visualizations" in content
    assert "optuna_slice" in content
    assert "Lift Curves" in content
    assert "lift_validation" in content


def test_build_optimization_focus_table_exposes_penalty_impact() -> None:
    fold_metrics = pl.DataFrame(
        {
            "fold": [1],
            "train_auc": [0.90],
            "val_auc": [0.70],
            "test_auc": [0.80],
            "objective_score_val": [0.30],
            "objective_score": [0.30],
        }
    )

    out = build_optimization_focus_table(fold_metrics)
    row = out.to_dicts()[0]

    assert row["train_val_gap_abs"] == pytest.approx(0.20)
    assert row["train_test_gap_abs"] == pytest.approx(0.10)
    assert row["val_penalty"] == pytest.approx(0.40)
    assert row["test_penalty"] == pytest.approx(0.50)
    assert row["implied_lambda_val"] == pytest.approx(2.0)
    assert row["implied_lambda_test"] == pytest.approx(5.0)


def test_build_backtest_fold_focus_table_keeps_test_period_context() -> None:
    fold_backtest_kpis = pl.DataFrame(
        {
            "fold": [1, 1, 1],
            "strategy": ["Portfolio", "Benchmark", "Active"],
            "total_return": [0.12, 0.03, 0.09],
            "cagr": [0.12, 0.03, 0.09],
            "avg_monthly_return": [0.06, 0.015, 0.045],
            "annualized_volatility": [0.10, 0.08, 0.12],
            "sharpe_ratio": [1.2, 0.3, 0.8],
            "sortino_ratio": [1.4, 0.4, 0.9],
            "calmar_ratio": [2.0, 0.5, 1.1],
            "max_drawdown": [-0.06, -0.04, -0.08],
            "win_rate": [0.5, 0.5, 0.5],
            "months": [2.0, 2.0, 2.0],
            "avg_hit_rate": [0.55, 0.0, 0.0],
            "avg_positions": [10.0, 0.0, 0.0],
        }
    )
    fold_index = pl.DataFrame(
        {
            "fold": [1],
            "test_month_start": ["2020-02-01"],
            "test_month_end": ["2020-03-01"],
            "test_rows": [20],
            "status": ["completed"],
            "skip_reason": [None],
        }
    )

    out = build_backtest_fold_focus_table(fold_backtest_kpis, fold_index)

    assert out.get_column("strategy").to_list() == ["Portfolio", "Benchmark", "Active"]
    assert out.get_column("test_month_start").to_list() == ["2020-02-01", "2020-02-01", "2020-02-01"]


def test_save_lift_curve(tmp_path: Path) -> None:
    path = tmp_path / "lift_curve.png"
    out = save_lift_curve(
        y_true=np.array([1, 0, 1, 0, 1, 0], dtype=float),
        y_score=np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], dtype=float),
        n_buckets=3,
        path=path,
        fold_label="fold_01",
        split_label="Validation",
    )

    assert out == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_generate_global_shap_report_pdf(tmp_path: Path, monkeypatch) -> None:
    fake_shap = ModuleType("shap")

    def summary_plot(shap_values, X, feature_names=None, show=False, max_display=None, plot_type=None):
        del show, max_display, plot_type
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        labels = feature_names or [str(idx) for idx in range(len(mean_abs))]
        plt.barh(labels[::-1], mean_abs[::-1])

    def dependence_plot(feature, shap_values, X, feature_names=None, show=False, interaction_index="auto"):
        del show, interaction_index
        labels = feature_names or [str(idx) for idx in range(X.shape[1])]
        feature_idx = labels.index(feature) if isinstance(feature, str) else int(feature)
        plt.scatter(X[:, feature_idx], shap_values[:, feature_idx], s=10, alpha=0.5)

    fake_shap.summary_plot = summary_plot
    fake_shap.dependence_plot = dependence_plot
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    rng = np.random.default_rng(42)
    feature_names = ["value", "momentum", "quality"]
    explanations = [
        ShapFoldExplanation(
            fold_label=f"fold_{idx:02d}",
            feature_names=feature_names,
            X_sample=rng.normal(size=(24, 3)),
            shap_values=rng.normal(scale=0.2, size=(24, 3)),
            interaction_sample=rng.normal(size=(12, 3)),
            interaction_values=rng.normal(scale=0.05, size=(12, 3, 3)),
            mean_abs_shap=np.array([0.2, 0.15, 0.1]),
        )
        for idx in range(1, 3)
    ]

    out = generate_global_shap_report_pdf(
        explanations=explanations,
        out_path=tmp_path / "shap_global_report.pdf",
        max_features=3,
    )

    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_backtest_audit_report(tmp_path: Path) -> None:
    monthly_returns = pl.DataFrame(
        {
            "year_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "decision_month": [date(2019, 12, 1), date(2020, 1, 1)],
            "holding_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "portfolio_return": [0.04, -0.01],
            "benchmark_return": [0.01, -0.02],
            "active_return": [0.03, 0.01],
            "hit_rate": [0.6, 0.4],
            "n_positions": [10, 10],
        }
    )
    fold_monthly_returns = pl.DataFrame(
        {
            "fold": [1, 1],
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "holding_month": [date(2020, 2, 1), date(2020, 3, 1)],
            "year_month": [date(2020, 2, 1), date(2020, 3, 1)],
            "portfolio_return": [0.05, -0.01],
            "benchmark_return": [0.01, -0.02],
            "hit_rate": [0.7, 0.5],
            "n_positions": [10, 10],
            "active_return": [0.04, 0.01],
        }
    )
    backtest_kpis = pl.DataFrame(
        {
            "strategy": ["Portfolio", "Benchmark", "Active"],
            "total_return": [0.0395, -0.0102, 0.0501],
            "cagr": [0.26, -0.06, 0.34],
            "avg_monthly_return": [0.015, -0.005, 0.02],
            "annualized_volatility": [0.12, 0.10, 0.15],
            "sharpe_ratio": [1.8, -0.4, 1.2],
            "sortino_ratio": [2.1, -0.5, 1.5],
            "calmar_ratio": [2.0, -0.3, 1.6],
            "max_drawdown": [-0.02, -0.03, -0.01],
            "win_rate": [0.5, 0.5, 0.5],
            "months": [2.0, 2.0, 2.0],
            "avg_hit_rate": [0.5, 0.0, 0.0],
            "avg_positions": [10.0, 0.0, 0.0],
        }
    )
    fold_backtest_kpis = pl.DataFrame(
        {
            "fold": [1, 1, 1],
            "strategy": ["Portfolio", "Benchmark", "Active"],
            "total_return": [0.0395, -0.0102, 0.0501],
            "cagr": [0.26, -0.06, 0.34],
            "avg_monthly_return": [0.02, -0.005, 0.025],
            "annualized_volatility": [0.12, 0.10, 0.15],
            "sharpe_ratio": [1.8, -0.4, 1.2],
            "sortino_ratio": [2.1, -0.5, 1.5],
            "calmar_ratio": [2.0, -0.3, 1.6],
            "max_drawdown": [-0.02, -0.03, -0.01],
            "win_rate": [0.5, 0.5, 0.5],
            "months": [2.0, 2.0, 2.0],
            "avg_hit_rate": [0.6, 0.0, 0.0],
            "avg_positions": [10.0, 0.0, 0.0],
        }
    )
    selections = pl.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US"],
            "year_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "holding_month": [date(2020, 2, 1), date(2020, 3, 1)],
            "prediction": [0.8, 0.3],
            "rank": [1, 2],
            "fold": [1, 1],
            "future_return": [0.05, -0.01],
            "benchmark_future_return": [0.01, -0.02],
            "future_excess_return": [0.04, 0.01],
            "future_relative_return": [0.0396, 0.0102],
            "holding_period_complete": [True, True],
        }
    )
    debug_predictions_long = selections.with_columns(
        pl.Series("target_label", [1, 1], dtype=pl.Int8),
        pl.Series("prediction_rank_in_month", [1, 2], dtype=pl.Int64),
        pl.Series("selected_top_n", [True, False], dtype=pl.Boolean),
        pl.Series("objective_score", [0.7, 0.7], dtype=pl.Float64),
        pl.Series("objective_score_val", [0.65, 0.65], dtype=pl.Float64),
        pl.Series("status", ["completed", "completed"]),
        pl.Series("skip_reason", [None, None]),
        pl.Series("train_month_start", ["2019-01-01", "2019-01-01"]),
        pl.Series("train_month_end", ["2019-12-01", "2019-12-01"]),
        pl.Series("val_month_start", ["2020-01-01", "2020-01-01"]),
        pl.Series("val_month_end", ["2020-01-01", "2020-01-01"]),
        pl.Series("test_month_start", ["2020-02-01", "2020-02-01"]),
        pl.Series("test_month_end", ["2020-02-01", "2020-02-01"]),
        pl.Series("train_positive_rate", [0.4, 0.4], dtype=pl.Float64),
        pl.Series("val_positive_rate", [0.5, 0.5], dtype=pl.Float64),
        pl.Series("test_positive_rate", [0.6, 0.6], dtype=pl.Float64),
        pl.Series("is_scored", [True, True], dtype=pl.Boolean),
        pl.Series("decision_asof_date", [date(2020, 1, 31), date(2020, 2, 29)]),
        pl.Series("holding_asof_date", [date(2020, 2, 29), date(2020, 3, 31)]),
        pl.Series("benchmark_holding_asof_date", [date(2020, 2, 29), date(2020, 3, 31)]),
        pl.Series("holding_period_complete", [True, True], dtype=pl.Boolean),
        pl.Series("monthly_return", [0.01, 0.02], dtype=pl.Float64),
    )
    fold_index = pl.DataFrame(
        {
            "fold": [1],
            "status": ["completed"],
            "skip_reason": [None],
            "train_month_start": ["2019-01-01"],
            "train_month_end": ["2019-12-01"],
            "val_month_start": ["2020-01-01"],
            "val_month_end": ["2020-01-01"],
            "test_month_start": ["2020-02-01"],
            "test_month_end": ["2020-02-01"],
            "train_rows": [100],
            "val_rows": [20],
            "test_rows": [20],
            "train_positive_rate": [0.4],
            "val_positive_rate": [0.5],
            "test_positive_rate": [0.6],
        }
    )

    report_path = tmp_path / "backtest_audit_report.html"
    out = write_backtest_audit_report(
        output_path=report_path,
        monthly_returns=monthly_returns,
        fold_monthly_returns=fold_monthly_returns,
        backtest_kpis=backtest_kpis,
        fold_backtest_kpis=fold_backtest_kpis,
        selections=selections,
        debug_predictions_long=debug_predictions_long,
        fold_index=fold_index,
        linked_artifacts={"debug_predictions_long": tmp_path / "debug_predictions_long.parquet"},
    )

    content = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "Backtest Report" in content
    assert "Test Fold KPI Focus" in content
    assert "Fold-by-Fold Test Period Breakdown" in content
    assert "Fold 01" in content
    assert "Prediction vs Future Excess Return" in content
    assert "debug_predictions_long.parquet" in content
