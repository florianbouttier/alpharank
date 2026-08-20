from __future__ import annotations

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "architecture" / "root_module_ownership_v1.json"


def test_every_root_module_has_an_owner_and_explicit_api() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rows = registry["modules"]
    registered_paths = {row["path"] for row in rows}
    observed_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "alpharank").glob("*.py")
    }

    assert registered_paths == observed_paths
    assert all(row["owner_package"].startswith("alpharank") for row in rows)
    assert all(row["public_api"] for row in rows)


def test_root_facades_reexport_their_owner_objects() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    facades = [row for row in registry["modules"] if row["role"] == "compatibility_facade"]

    assert len(facades) == 6
    for row in facades:
        module_name = row["path"].removeprefix("src/").removesuffix(".py").replace("/", ".")
        facade = importlib.import_module(module_name)
        owner = importlib.import_module(row["owner_module"])
        assert facade.__all__ == row["public_api"]
        assert all(getattr(facade, name) is getattr(owner, name) for name in facade.__all__)
