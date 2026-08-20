"""Explicit loading of local modules without mutating the process import path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_path(
    module_name: str,
    module_path: Path,
    *,
    package_directory: Path | None = None,
) -> ModuleType:
    """Load one named module from a reviewed path without changing ``sys.path``."""

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    search_locations = [str(package_directory)] if package_directory is not None else None
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
