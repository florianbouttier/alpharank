from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_shap_plots(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    predictions: np.ndarray,
    out_dir: Path,
    fold_label: str,
    max_samples: int,
    max_features: int,
) -> Dict[str, Path]:
    _ensure_dir(out_dir)

    output: Dict[str, Path] = {}
    if X_test.size == 0:
        return output

    sample_size = min(int(max_samples), X_test.shape[0])
    sample_idx = np.linspace(0, X_test.shape[0] - 1, sample_size).astype(int)
    X_sample = X_test[sample_idx]

    try:
        import shap
    except Exception:
        return output

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        beeswarm_path = out_dir / f"{fold_label}_shap_beeswarm.png"
        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=max_features,
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=150)
        plt.close()
        output["beeswarm"] = beeswarm_path

        # Individual explanation: custom bar chart for top prediction.
        top_idx = int(np.argmax(predictions)) if predictions.size else 0
        if 0 <= top_idx < X_test.shape[0]:
            sample_contrib = np.asarray(explainer.shap_values(X_test[[top_idx]]))
            if sample_contrib.ndim == 3:
                sample_contrib = sample_contrib[-1]
            if isinstance(sample_contrib, list):
                sample_contrib = np.asarray(sample_contrib[-1])
            sample_contrib = np.asarray(sample_contrib).reshape(-1)

            abs_idx = np.argsort(np.abs(sample_contrib))[::-1][: max_features]
            labels = [feature_names[i] for i in abs_idx]
            values = sample_contrib[abs_idx]

            indiv_path = out_dir / f"{fold_label}_shap_individual.png"
            plt.figure(figsize=(10, 6))
            colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
            plt.barh(labels[::-1], values[::-1], color=colors[::-1])
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title(f"{fold_label} SHAP Individual Contributions")
            plt.xlabel("SHAP value")
            plt.tight_layout()
            plt.savefig(indiv_path, dpi=150)
            plt.close()
            output["individual"] = indiv_path
    except Exception:
        return output

    return output
