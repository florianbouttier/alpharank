# %%
import numpy as np
import pandas as pd
from typing import Dict

def lift_curve(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """
    Simplifié: on découpe y_true en n_bins quantiles.
    Pour chaque bin:
      - bornes (q_min, q_max)
      - nombre d'observations
      - moyenne y_true
      - moyenne y_pred
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape, "Shapes must match."

    # Gestion cas dégénéré (tous identiques)
    if np.allclose(y_true.min(), y_true.max()):
        edges = np.linspace(y_true.min(), y_true.max() + 1e-12, n_bins + 1)
    else:
        edges = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
        # Assurer strictement croissant
        edges = np.unique(edges)
        if len(edges) - 1 < n_bins:
            # Réduit le nombre effectif de bins si peu de valeurs distinctes
            pass

    bins_idx = np.digitize(y_true, edges[1:-1], right=False)
    records = []
    for b in range(len(edges) - 1):
        mask = bins_idx == b
        if not np.any(mask):
            continue
        yt_bin = y_true[mask]
        yp_bin = y_pred[mask]
        records.append({
            "bin": b + 1,
            "q_min": float(edges[b]),
            "q_max": float(edges[b + 1]),
            "count": int(mask.sum()),
            "y_true_mean": float(yt_bin.mean()),
            "y_pred_mean": float(yp_bin.mean())
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["bin_fraction"] = df["count"] / df["count"].sum()
    return df

def plot_lift_curve(lift_df: pd.DataFrame,
                    y_true=None,
                    y_pred=None,
                    label_true="y_true",
                    label_pred="y_pred",
                    jitter_alpha: float = 0.08,
                    jitter_size: float = 1.0,
                    max_bins_violin: int = 30,
                    show_means: bool = True,
                    mean_linewidth: float = 2.0,
                    mean_marker_size: float = 6.5):
    """
    Distribution des prédictions par quantile de y_true + moyennes très visibles.
    - Violin (semi-transparent) + léger jitter.
    - Lignes des moyennes (prédiction vs vrai) mises en avant (épaisseur + zorder).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if lift_df.empty:
        return None

    n_bins = len(lift_df)
    use_distribution = (y_true is not None) and (y_pred is not None) and (n_bins <= max_bins_violin)
    seaborn_available = False
    if use_distribution:
        try:
            import seaborn as sns
            seaborn_available = True
        except ImportError:
            use_distribution = False

    if not use_distribution or not seaborn_available:
        fig, ax = plt.subplots(figsize=(6, 4))
        if show_means:
            ax.plot(lift_df["y_true_mean"], lift_df["y_pred_mean"],
                    marker="o", markersize=mean_marker_size,
                    linestyle="-", linewidth=mean_linewidth,
                    color="crimson", label="Moyennes (quantiles)")
            all_vals = np.concatenate([lift_df["y_true_mean"].values,
                                       lift_df["y_pred_mean"].values])
            vmin, vmax = np.min(all_vals), np.max(all_vals)
            if vmin == vmax:
                vmax = vmin + 1e-6
            ax.plot([vmin, vmax], [vmin, vmax],
                    color="black", linestyle="--", linewidth=1.2, label="Identité")
            ax.legend()
        ax.set_title("Calibration (fallback)")
        ax.set_xlabel(f"Moyenne {label_true} (quantile)")
        ax.set_ylabel(f"Moyenne {label_pred}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    # Distribution complète
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    assert y_true_arr.shape == y_pred_arr.shape, "Shapes must match."
    if not {"q_min", "q_max"}.issubset(lift_df.columns):
        return None

    last_qmax = lift_df["q_max"].iloc[-1]
    edges = np.concatenate([lift_df["q_min"].values, [last_qmax]])
    _, unique_idx = np.unique(edges, return_index=True)
    edges = edges[np.sort(unique_idx)]
    if len(edges) < 2:
        edges = np.array([np.min(edges), np.min(edges) + 1e-12])

    bins_idx = np.digitize(y_true_arr, edges[1:-1], right=False)
    plot_df = pd.DataFrame({"bin": bins_idx + 1, "y_pred": y_pred_arr})

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(min(1.0 * n_bins + 2, 11), 5))
    sns.violinplot(
        data=plot_df,
        x="bin",
        y="y_pred",
        ax=ax,
        inner=None,
        color="#1f77b4",
        linewidth=0.5,
        alpha=0.28
    )
    sns.stripplot(
        data=plot_df,
        x="bin",
        y="y_pred",
        ax=ax,
        color="black",
        alpha=jitter_alpha,
        size=jitter_size,
        jitter=0.25
    )

    handles = []
    if show_means:
        # Moyenne prédite
        h_pred, = ax.plot(
            lift_df["bin"],
            lift_df["y_pred_mean"],
            color="crimson",
            marker="o",
            markersize=mean_marker_size,
            linestyle="-",
            linewidth=mean_linewidth,
            label=f"Moyenne {label_pred}",
            zorder=5
        )
        # Moyenne vraie
        h_true, = ax.plot(
            lift_df["bin"],
            lift_df["y_true_mean"],
            color="black",
            marker="D",
            markersize=mean_marker_size * 0.85,
            linestyle="--",
            linewidth=mean_linewidth * 0.9,
            label=f"Moyenne {label_true}",
            zorder=6
        )
        # Lien visuel: segments verticaux (erreur)
        for _, r in lift_df.iterrows():
            ax.vlines(
                r["bin"],
                r["y_true_mean"],
                r["y_pred_mean"],
                colors="gray",
                linewidth=0.9,
                alpha=0.65,
                zorder=4
            )
        handles.extend([h_pred, h_true])

    ax.set_xlabel(f"Quantiles de {label_true}")
    ax.set_ylabel(f"{label_pred}")
    ax.set_title("Distribution + Moyennes par quantile")
    if handles:
        ax.legend(handles=handles, frameon=True, loc="best")
    ax.grid(axis="y", alpha=0.25)

    for _, row in lift_df.iterrows():
        ax.text(
            row["bin"] - 1,
            ax.get_ylim()[1],
            f"n={int(row['count'])}",
            rotation=90,
            fontsize=7,
            va="top",
            ha="center",
            alpha=0.5
        )

    fig.tight_layout()
    return fig

def overprediction_penalty(y_true, y_pred, alpha: float = 1.0) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    diff = np.maximum(y_pred - y_true, 0.0)
    return float(alpha * np.mean(diff**2))

def evaluate_regressor(y_true,
                       y_pred,
                       plot_lift: bool = False,
                       n_lift_bins: int = 10,
                       alpha_over: float = 1.0) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape
    resid = y_pred - y_true
    over_diff = np.maximum(resid, 0.0)

    out: Dict[str, float] = {
        "mae": float(np.mean(np.abs(resid))),
        "mse": float(np.mean(resid**2)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "r2": float(1 - np.sum(resid**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12)),
        "spread": float(y_pred.max() - y_pred.min()),
        "std": float(np.std(y_pred)),
        "std_ratio": float(np.std(y_pred))/float(np.std(y_true)),
        "mean_overprediction": float(np.mean(over_diff)),
        "overprediction_penalty": overprediction_penalty(y_true, y_pred, alpha_over),
    }

    # Nouvelle courbe calibration (ex- lift)
    calib_df = lift_curve(y_true, y_pred, n_bins=n_lift_bins)
    calib_df_inverse = lift_curve(y_pred,y_true, n_bins=n_lift_bins)
    out["last_quantile"] = float(calib_df_inverse["y_pred_mean"].iloc[-1])
    if not calib_df.empty:
        diff_bin = calib_df["y_pred_mean"] - calib_df["y_true_mean"]
        out["calibration_mae"] = float(np.mean(np.abs(diff_bin)))
        out["last_error"] = 1/float(np.abs(diff_bin.iloc[-1]))
        
        if calib_df["y_true_mean"].nunique() > 1 and calib_df["y_pred_mean"].nunique() > 1:
            out["calibration_corr"] = float(np.corrcoef(calib_df["y_true_mean"], calib_df["y_pred_mean"])[0, 1])
        else:
            out["calibration_corr"] = float("nan")
    else:
        out["calibration_mae"] = float("nan")
        out["calibration_corr"] = float("nan")

    if plot_lift:
        fig = plot_lift_curve(calib_df, y_true=y_true, y_pred=y_pred)
        out["_calibration_df"] = calib_df
        out["_calibration_figure"] = fig
    return out

def portfolio_score(topn_realized: pd.Series,
                    y_true_topn: np.ndarray,
                    y_pred_topn: np.ndarray,
                    alpha_over: float = 1.0) -> float:
    realized_mean = float(np.mean(topn_realized))
    pen = overprediction_penalty(y_true_topn, y_pred_topn, alpha_over)
    return realized_mean - pen
