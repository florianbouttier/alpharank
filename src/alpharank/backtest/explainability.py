from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class ShapFoldExplanation:
    fold_label: str
    feature_names: List[str]
    X_sample: np.ndarray
    shap_values: np.ndarray
    interaction_sample: np.ndarray | None
    interaction_values: np.ndarray | None
    mean_abs_shap: np.ndarray


@dataclass
class ShapArtifacts:
    paths: Dict[str, Path]
    explanation: ShapFoldExplanation | None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sample_indices(n_rows: int, max_samples: int) -> np.ndarray:
    sample_size = min(int(max_samples), n_rows)
    if sample_size <= 0:
        return np.array([], dtype=int)
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=int)
    return np.linspace(0, n_rows - 1, sample_size).astype(int)


def _normalize_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[-1]
    return np.asarray(values, dtype=float)


def _normalize_interaction_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 4:
        values = values[-1]
    return np.asarray(values, dtype=float)


def _save_current_figure(pdf: PdfPages) -> None:
    fig = plt.gcf()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")


def _top_feature_indices(mean_abs_shap: np.ndarray, max_features: int) -> np.ndarray:
    if mean_abs_shap.size == 0:
        return np.array([], dtype=int)
    limit = min(int(max_features), mean_abs_shap.size)
    return np.argsort(mean_abs_shap)[::-1][:limit]


def _interaction_ranking(interaction_values: np.ndarray) -> List[Dict[str, float]]:
    mean_abs = np.mean(np.abs(interaction_values), axis=0)
    mean_abs = 0.5 * (mean_abs + mean_abs.T)
    np.fill_diagonal(mean_abs, 0.0)

    rows: List[Dict[str, float]] = []
    for i in range(mean_abs.shape[0]):
        for j in range(i + 1, mean_abs.shape[1]):
            rows.append({"i": float(i), "j": float(j), "strength": float(mean_abs[i, j])})

    rows.sort(key=lambda row: row["strength"], reverse=True)
    return rows


def _significant_interactions(
    interaction_values: np.ndarray,
    max_interactions: int,
) -> List[Dict[str, float]]:
    ranked = _interaction_ranking(interaction_values)
    if not ranked:
        return []

    max_strength = float(ranked[0]["strength"])
    if max_strength <= 0.0:
        return []

    threshold = max(max_strength * 0.15, np.percentile([row["strength"] for row in ranked], 90))
    significant = [row for row in ranked if row["strength"] >= threshold]
    if not significant:
        significant = ranked[:1]
    return significant[: max(1, int(max_interactions))]


def collect_shap_explanation(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    fold_label: str,
    max_samples: int,
    interaction_max_samples: int,
) -> ShapArtifacts:
    _ensure_dir(out_dir)

    if X_test.size == 0:
        return ShapArtifacts(paths={}, explanation=None)

    try:
        import shap
    except Exception:
        return ShapArtifacts(paths={}, explanation=None)

    sample_idx = _sample_indices(X_test.shape[0], max_samples=max_samples)
    if sample_idx.size == 0:
        return ShapArtifacts(paths={}, explanation=None)

    X_sample = X_test[sample_idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = _normalize_shap_values(explainer.shap_values(X_sample))
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        interaction_values = None
        interaction_sample = None
        interaction_idx = _sample_indices(X_sample.shape[0], max_samples=interaction_max_samples)
        if interaction_idx.size > 0:
            interaction_sample = X_sample[interaction_idx]
            try:
                interaction_values = _normalize_interaction_values(
                    explainer.shap_interaction_values(interaction_sample)
                )
            except Exception:
                interaction_values = None
                interaction_sample = None

        explanation = ShapFoldExplanation(
            fold_label=fold_label,
            feature_names=list(feature_names),
            X_sample=X_sample,
            shap_values=shap_values,
            interaction_sample=interaction_sample,
            interaction_values=interaction_values,
            mean_abs_shap=mean_abs_shap,
        )
        return ShapArtifacts(paths={}, explanation=explanation)
    except Exception:
        return ShapArtifacts(paths={}, explanation=None)


def _plot_text_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=18, fontweight="bold", va="top")
    ax.text(0.02, 0.88, "\n".join(lines), fontsize=11, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_global_beeswarm(pdf: PdfPages, shap_values: np.ndarray, X: np.ndarray, feature_names: List[str], max_features: int) -> None:
    import shap

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=max_features,
    )
    plt.title("Global SHAP Beeswarm (all folds)")
    _save_current_figure(pdf)


def _plot_global_importance_bar(
    pdf: PdfPages,
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_features: int,
) -> None:
    import shap

    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=max_features,
    )
    plt.title("Global SHAP Importance (mean |SHAP|)")
    _save_current_figure(pdf)


def _plot_dependence_pages(
    pdf: PdfPages,
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_features: int,
) -> None:
    import shap

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    for idx in _top_feature_indices(mean_abs, max_features=min(6, max_features)):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_names[idx],
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            interaction_index="auto",
        )
        plt.title(f"SHAP Dependence: {feature_names[idx]}")
        _save_current_figure(pdf)


def _plot_per_fold_beeswarms(
    pdf: PdfPages,
    explanations: List[ShapFoldExplanation],
    max_features: int,
) -> None:
    import shap

    for explanation in explanations:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            explanation.shap_values,
            explanation.X_sample,
            feature_names=explanation.feature_names,
            show=False,
            max_display=min(15, max_features),
        )
        plt.title(f"{explanation.fold_label} SHAP Beeswarm")
        _save_current_figure(pdf)


def _plot_per_fold_exhaustive_dependence_pages(
    pdf: PdfPages,
    explanations: List[ShapFoldExplanation],
) -> None:
    import shap

    for explanation in explanations:
        _plot_text_page(
            pdf,
            title=f"{explanation.fold_label} SHAP 1D Dependence",
            lines=[
                f"fold={explanation.fold_label}",
                f"samples={explanation.X_sample.shape[0]}",
                f"features={len(explanation.feature_names)}",
                "",
                "The following pages contain all 1D SHAP dependence plots for this fold.",
            ],
        )
        for feature_name in explanation.feature_names:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_name,
                explanation.shap_values,
                explanation.X_sample,
                feature_names=explanation.feature_names,
                show=False,
                interaction_index=None,
            )
            plt.title(f"{explanation.fold_label} SHAP Dependence: {feature_name}")
            _save_current_figure(pdf)


def _plot_interaction_heatmap(
    pdf: PdfPages,
    interaction_values: np.ndarray,
    feature_names: List[str],
    max_features: int,
) -> List[Dict[str, float]]:
    top_pairs = _significant_interactions(interaction_values, max_interactions=min(8, max_features))
    if not top_pairs:
        return []

    selected = sorted(
        {
            int(row["i"])
            for row in top_pairs
        }
        | {
            int(row["j"])
            for row in top_pairs
        }
    )
    selected = selected[: min(len(selected), 10)]

    mean_abs = np.mean(np.abs(interaction_values), axis=0)
    mean_abs = 0.5 * (mean_abs + mean_abs.T)
    matrix = mean_abs[np.ix_(selected, selected)]
    labels = [feature_names[idx] for idx in selected]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Mean |SHAP interaction| heatmap")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return top_pairs


def _plot_interaction_ranking(
    pdf: PdfPages,
    top_pairs: List[Dict[str, float]],
    feature_names: List[str],
) -> None:
    labels = [f"{feature_names[int(row['i'])]} x {feature_names[int(row['j'])]}" for row in top_pairs]
    values = [float(row["strength"]) for row in top_pairs]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels[::-1], values[::-1], color="#2a9d8f")
    ax.set_xlabel("Mean |interaction SHAP|")
    ax.set_title("Top significant SHAP interactions")
    ax.grid(alpha=0.2, axis="x")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_interaction_dependence_pages(
    pdf: PdfPages,
    interaction_values: np.ndarray,
    interaction_sample: np.ndarray,
    feature_names: List[str],
    top_pairs: List[Dict[str, float]],
) -> None:
    for row in top_pairs:
        i = int(row["i"])
        j = int(row["j"])
        x = interaction_sample[:, i]
        color = interaction_sample[:, j]
        y = interaction_values[:, i, j]

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(x, y, c=color, cmap="coolwarm", alpha=0.75, s=24)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(f"Interaction SHAP: {feature_names[i]} x {feature_names[j]}")
        ax.set_title(f"Interaction dependence: {feature_names[i]} x {feature_names[j]}")
        ax.grid(alpha=0.2)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(feature_names[j])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def generate_global_shap_report_pdf(
    explanations: List[ShapFoldExplanation],
    out_path: Path,
    max_features: int,
) -> Path | None:
    explanations = [explanation for explanation in explanations if explanation is not None]
    if not explanations:
        return None

    feature_names = explanations[0].feature_names
    if any(explanation.feature_names != feature_names for explanation in explanations):
        return None

    _ensure_dir(out_path.parent)

    all_X = np.concatenate([explanation.X_sample for explanation in explanations], axis=0)
    all_shap = np.concatenate([explanation.shap_values for explanation in explanations], axis=0)
    fold_lines = [
        f"- {explanation.fold_label}: samples={explanation.X_sample.shape[0]}"
        for explanation in explanations
    ]

    interaction_ready = [
        explanation
        for explanation in explanations
        if explanation.interaction_values is not None and explanation.interaction_sample is not None
    ]

    with PdfPages(out_path) as pdf:
        summary_lines = [
            f"folds={len(explanations)}",
            f"global_samples={all_X.shape[0]}",
            f"features={len(feature_names)}",
            f"interaction_folds={len(interaction_ready)}",
            "",
            "Included folds:",
            *fold_lines,
        ]
        _plot_text_page(pdf, title="Global SHAP Analysis", lines=summary_lines)
        _plot_global_beeswarm(pdf, shap_values=all_shap, X=all_X, feature_names=feature_names, max_features=max_features)
        _plot_global_importance_bar(pdf, shap_values=all_shap, X=all_X, feature_names=feature_names, max_features=max_features)
        _plot_dependence_pages(pdf, shap_values=all_shap, X=all_X, feature_names=feature_names, max_features=max_features)
        _plot_per_fold_beeswarms(pdf, explanations=explanations, max_features=max_features)
        _plot_per_fold_exhaustive_dependence_pages(pdf, explanations=explanations)

        if interaction_ready:
            interaction_sample = np.concatenate(
                [explanation.interaction_sample for explanation in interaction_ready if explanation.interaction_sample is not None],
                axis=0,
            )
            interaction_values = np.concatenate(
                [explanation.interaction_values for explanation in interaction_ready if explanation.interaction_values is not None],
                axis=0,
            )
            top_pairs = _plot_interaction_heatmap(
                pdf,
                interaction_values=interaction_values,
                feature_names=feature_names,
                max_features=max_features,
            )
            if top_pairs:
                _plot_interaction_ranking(pdf, top_pairs=top_pairs, feature_names=feature_names)
                _plot_interaction_dependence_pages(
                    pdf,
                    interaction_values=interaction_values,
                    interaction_sample=interaction_sample,
                    feature_names=feature_names,
                    top_pairs=top_pairs,
                )
            else:
                _plot_text_page(
                    pdf,
                    title="SHAP Interactions",
                    lines=["No dominant second-order interactions were detected on the sampled folds."],
                )

    return out_path
