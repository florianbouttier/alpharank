from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def shap_row_indexes(*, row_count: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size > 0 and row_count > sample_size:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(row_count, sample_size, replace=False))
    return np.arange(row_count)


def compute_shap_sample(
    *,
    fitted,
    X: np.ndarray,
    source: pl.DataFrame,
    fold: int,
    method: str,
    horizon: int,
    sample_size: int,
    seed: int,
) -> pl.DataFrame:
    import shap

    indexes = shap_row_indexes(
        row_count=X.shape[0], sample_size=sample_size, seed=seed
    )
    values = np.asarray(shap.TreeExplainer(fitted.model.model_).shap_values(X[indexes]))
    if values.ndim == 3:
        values = values[:, :, -1]
    rows = source[indexes].select("decision_month", "ticker").with_columns(
        pl.lit(fold).alias("fold"),
        pl.lit(method).alias("method"),
        pl.lit(horizon).alias("horizon"),
    )
    feature_values = pl.DataFrame(
        {f"value__{name}": X[indexes, index] for index, name in enumerate(fitted.features)}
    )
    shap_values = pl.DataFrame(
        {f"shap__{name}": values[:, index] for index, name in enumerate(fitted.features)}
    )
    return rows.hstack(feature_values).hstack(shap_values)


def write_shap_outputs(samples: pl.DataFrame, output_dir: Path, *, top_features: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.write_parquet(output_dir / "shap_samples.parquet")
    shap_columns = [column for column in samples.columns if column.startswith("shap__")]
    importance = (
        samples.select([pl.col(column).abs().mean().alias(column.removeprefix("shap__")) for column in shap_columns])
        .transpose(include_header=True, header_name="feature", column_names=["mean_abs_shap"])
        .sort("mean_abs_shap", descending=True)
    )
    importance.write_csv(output_dir / "shap_importance.csv")
    direction_rows: list[dict] = []
    for shap_column in shap_columns:
        feature = shap_column.removeprefix("shap__")
        value_column = f"value__{feature}"
        if value_column not in samples.columns:
            continue
        correlation = samples.select(pl.corr(value_column, shap_column)).item()
        direction_rows.append(
            {
                "feature": feature,
                "value_shap_correlation": correlation,
                "direction": (
                    "higher increases score"
                    if correlation is not None and correlation > 0.1
                    else "higher decreases score"
                    if correlation is not None and correlation < -0.1
                    else "non-monotonic or weak"
                ),
            }
        )
    if direction_rows:
        (
            pl.DataFrame(direction_rows)
            .join(importance, on="feature", how="left")
            .sort("mean_abs_shap", descending=True)
            .write_csv(output_dir / "shap_direction.csv")
        )
    try:
        import matplotlib.pyplot as plt

        shown = importance.head(top_features).sort("mean_abs_shap")
        plt.figure(figsize=(9, max(5, 0.27 * shown.height)))
        plt.barh(shown["feature"], shown["mean_abs_shap"])
        plt.xlabel("Mean absolute SHAP")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance.png", dpi=160)
        plt.close()
    except Exception:
        pass
