"""Thin compatibility facades for stable script command paths."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import MutableMapping

from alpharank.utils.module_loading import load_module_from_path


def expose_script_implementation(
    namespace: MutableMapping[str, object],
    *,
    target: Path,
    module_name: str,
) -> ModuleType:
    """Load one moved script and expose its non-dunder API through a stable facade."""

    implementation = load_module_from_path(module_name, target)
    for name in dir(implementation):
        if not name.startswith("__"):
            namespace.setdefault(name, getattr(implementation, name))
    facade_name = namespace.get("__name__")
    if (
        isinstance(facade_name, str)
        and facade_name != "__main__"
        and facade_name in sys.modules
    ):
        # Importers must receive the implementation module itself. Besides
        # preserving functions and constants, this keeps monkeypatching and
        # module-level configuration behavior identical after the move.
        sys.modules[facade_name] = implementation
    return implementation
