from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.explainability import ShapFoldExplanation, generate_global_shap_report_pdf
from alpharank.backtest.reporting import (
    save_learning_curve,
    save_optuna_visualizations,
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
            }
        ],
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Global Visualizations" in content
    assert "Per-Fold Analysis" not in content
    assert "Optuna Visualizations" in content
    assert "optuna_slice" in content


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
