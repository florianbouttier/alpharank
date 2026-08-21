from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "validate_documentation.py"
SPEC = importlib.util.spec_from_file_location("validate_documentation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_documentation_structure_is_complete() -> None:
    assert MODULE.validate(ROOT) == []
