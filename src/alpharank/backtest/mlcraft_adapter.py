from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from alpharank.utils.module_loading import load_module_from_path

LEGACY_PARAM_ALIASES = {
    "n_estimators": ("num_boost_round", "fit"),
    "reg_alpha": ("alpha", "model"),
    "reg_lambda": ("lambda", "model"),
}


def ensure_mlcraft_importable() -> Path | None:
    """Ensure the sibling/local mlcraft checkout can be imported."""

    try:
        importlib.import_module("mlcraft")
        _patch_numpy_compat()
        return None
    except ModuleNotFoundError:
        pass

    project_root = Path(__file__).resolve().parents[3]
    candidates = []
    env_src = os.environ.get("MLCRAFT_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    candidates.extend(
        [
            project_root / "mlcraft" / "src",
            project_root.parent / "mlcraft" / "src",
        ]
    )

    for candidate in candidates:
        package_directory = candidate / "mlcraft"
        package_init = package_directory / "__init__.py"
        if package_init.exists():
            load_module_from_path(
                "mlcraft",
                package_init,
                package_directory=package_directory,
            )
            _patch_numpy_compat()
            return candidate

    raise ModuleNotFoundError(
        "mlcraft is required for the boosting training path. "
        "Install it or set MLCRAFT_SRC to the local mlcraft/src directory."
    )


def _patch_numpy_compat() -> None:
    if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
        np.trapezoid = np.trapz  # type: ignore[attr-defined]


def to_mlcraft_model_and_fit_params(base_params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model_params: Dict[str, Any] = {}
    fit_params: Dict[str, Any] = {}

    ignored = {"objective", "eval_metric", "random_state"}
    for key, value in dict(base_params or {}).items():
        if key in ignored:
            continue
        if key == "n_jobs":
            model_params["nthread"] = value
            continue
        alias = LEGACY_PARAM_ALIASES.get(key)
        if alias is not None:
            target_key, target = alias
            if target == "fit":
                fit_params[target_key] = value
            else:
                model_params[target_key] = value
            continue
        model_params[key] = value

    return model_params, fit_params


def to_mlcraft_search_space(
    search_space: Dict[str, Tuple[str, float, float]],
) -> Dict[str, Dict[str, Any]]:
    converted: Dict[str, Dict[str, Any]] = {}
    for raw_name, (raw_type, low, high) in dict(search_space or {}).items():
        name, target = LEGACY_PARAM_ALIASES.get(raw_name, (raw_name, "model"))
        if raw_type == "int":
            spec: Dict[str, Any] = {"type": "int", "low": int(low), "high": int(high)}
        elif raw_type == "loguniform":
            spec = {"type": "float", "low": float(low), "high": float(high), "log": True}
        else:
            spec = {"type": "float", "low": float(low), "high": float(high)}
        if target == "fit":
            spec["target"] = "fit"
        converted[name] = spec
    return converted
