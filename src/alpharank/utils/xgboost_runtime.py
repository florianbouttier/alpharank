from __future__ import annotations

import importlib
import os
import site
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xgboost as xgb


def _prepend_env_path(name: str, value: Path) -> None:
    value_str = str(value)
    current = os.environ.get(name, "")
    parts = [part for part in current.split(":") if part]
    if value_str in parts:
        return
    os.environ[name] = ":".join([value_str, *parts]) if parts else value_str


def _candidate_libomp_paths() -> list[Path]:
    candidates: list[Path] = []
    for site_dir in site.getsitepackages():
        site_path = Path(site_dir)
        candidates.extend(
            [
                site_path / "cmeel.prefix" / "lib" / "libomp.dylib",
                site_path / "sklearn" / ".dylibs" / "libomp.dylib",
            ]
        )
    return [path for path in candidates if path.exists()]


def _runtime_libomp_path() -> Path:
    return Path(sys.base_prefix) / "lib" / "libomp.dylib"


def prepare_xgboost_runtime() -> None:
    if sys.platform != "darwin":
        return

    runtime_libomp = _runtime_libomp_path()
    if runtime_libomp.exists():
        _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", runtime_libomp.parent)
        _prepend_env_path("DYLD_LIBRARY_PATH", runtime_libomp.parent)
        return

    for candidate in _candidate_libomp_paths():
        runtime_libomp.parent.mkdir(parents=True, exist_ok=True)
        try:
            if runtime_libomp.is_symlink() or runtime_libomp.exists():
                runtime_libomp.unlink()
            runtime_libomp.symlink_to(candidate)
        except OSError:
            continue
        _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", runtime_libomp.parent)
        _prepend_env_path("DYLD_LIBRARY_PATH", runtime_libomp.parent)
        return


def load_xgboost() -> "xgb":
    prepare_xgboost_runtime()
    try:
        return importlib.import_module("xgboost")
    except Exception as exc:
        if sys.platform == "darwin" and "libomp.dylib" in str(exc):
            raise RuntimeError(
                "xgboost could not load libomp.dylib. Install project dependencies in this "
                "environment with `python -m pip install -e .` so the macOS OpenMP runtime "
                "(`cmeel-libomp`) is available."
            ) from exc
        raise
