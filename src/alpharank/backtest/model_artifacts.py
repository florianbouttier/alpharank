"""Replayable per-fold boosting model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from alpharank.multihorizon.preprocessing import FoldPreprocessor


@dataclass(frozen=True)
class SerializedFoldPredictor:
    features: tuple[str, ...]
    preprocessor: FoldPreprocessor
    backend: str
    model_path: Path | None
    constant_probability: float | None
    best_num_iterations: int | None

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        _, matrix = self.preprocessor.transform(frame)
        if self.backend == "constant":
            return np.full(matrix.shape[0], float(self.constant_probability), dtype=float)
        if self.backend != "xgboost" or self.model_path is None:
            raise ValueError(f"Unsupported serialized backend: {self.backend}")
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(self.model_path)
        dmatrix = xgb.DMatrix(matrix)
        if self.best_num_iterations is not None:
            return booster.predict(
                dmatrix, iteration_range=(0, int(self.best_num_iterations))
            )
        return booster.predict(dmatrix)


def serialize_fold_model(
    *,
    fold_dir: Path,
    model: Any | None,
    preprocessor: FoldPreprocessor,
    seed: int,
    fold_metadata: Mapping[str, Any],
    constant_probability: float | None = None,
) -> dict[str, Any]:
    """Persist a native model plus everything needed for OOS replay."""

    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.ubj"
    if model is None:
        if constant_probability is None or not np.isfinite(constant_probability):
            raise ValueError("A constant fallback model requires its probability.")
        backend = "constant"
        model_sha256 = None
        best_num_iterations = None
    else:
        native = getattr(model, "model_", model)
        if not hasattr(native, "save_model"):
            raise TypeError("Fold model does not expose native save_model().")
        native.save_model(model_path)
        backend = "xgboost"
        model_sha256 = _sha256(model_path)
        best_num_iterations = getattr(model, "best_num_iterations_", None)
    manifest = {
        "model_artifact_contract_version": 1,
        "backend": backend,
        "model_file": model_path.name if backend != "constant" else None,
        "model_sha256": model_sha256,
        "features": list(preprocessor.features),
        "global_medians": preprocessor.global_medians,
        "seed": int(seed),
        "best_num_iterations": best_num_iterations,
        "constant_probability": constant_probability if backend == "constant" else None,
        "fold_metadata": dict(fold_metadata),
    }
    (fold_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_serialized_fold_predictor(fold_dir: Path) -> SerializedFoldPredictor:
    manifest_path = fold_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_file = manifest.get("model_file")
    model_path = fold_dir / str(model_file) if model_file else None
    if model_path is not None and _sha256(model_path) != manifest.get("model_sha256"):
        raise RuntimeError("Serialized fold model hash mismatch.")
    features = tuple(str(value) for value in manifest["features"])
    preprocessor = FoldPreprocessor(
        features=features,
        global_medians={
            str(key): float(value)
            for key, value in manifest["global_medians"].items()
        },
    )
    return SerializedFoldPredictor(
        features=features,
        preprocessor=preprocessor,
        backend=str(manifest["backend"]),
        model_path=model_path,
        constant_probability=(
            float(manifest["constant_probability"])
            if manifest.get("constant_probability") is not None
            else None
        ),
        best_num_iterations=(
            int(manifest["best_num_iterations"])
            if manifest.get("best_num_iterations") is not None
            else None
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
