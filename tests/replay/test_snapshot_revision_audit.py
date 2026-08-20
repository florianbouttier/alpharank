from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/audit_open_source_snapshot_revisions.py"
)
SPEC = importlib.util.spec_from_file_location("snapshot_revision_audit", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_revision_audit_separates_exact_and_material_numeric_drift() -> None:
    previous = pl.DataFrame({"id": [1, 2], "value": [0.1, 2.0]})
    current = pl.DataFrame({"id": [1, 2], "value": [0.1 + 1e-16, 3.0]})

    result = module._compare_frames(
        previous,
        current,
        keys=("id",),
        materiality_tolerance=1e-12,
    )

    assert result["changed_common_rows"] == 2
    assert result["materially_changed_common_rows"] == 1
    assert result["maximum_numeric_absolute_difference"] == 1.0


def test_revision_audit_fails_on_duplicate_natural_keys() -> None:
    duplicate = pl.DataFrame({"id": [1, 1], "value": [1.0, 2.0]})
    current = pl.DataFrame({"id": [1], "value": [1.0]})

    try:
        module._compare_frames(duplicate, current, keys=("id",))
    except ValueError as exc:
        assert "duplicate natural keys" in str(exc)
    else:
        raise AssertionError("Expected duplicate natural keys to fail closed")


def test_revision_audit_reports_additive_schema_without_ignoring_old_values() -> None:
    previous = pl.DataFrame({"id": [1], "value": [1.0]})
    current = pl.DataFrame({"id": [1], "value": [2.0], "cost_detail": [0.1]})

    result = module._compare_frames(
        previous,
        current,
        keys=("id",),
        materiality_tolerance=1e-12,
        allow_additive_schema=True,
    )

    assert result["schema"]["added_columns"] == ["cost_detail"]
    assert result["schema"]["removed_columns"] == []
    assert result["schema"]["compared_columns"] == ["id", "value"]
    assert result["materially_changed_common_rows"] == 1


def test_revision_audit_keeps_schema_strict_by_default() -> None:
    previous = pl.DataFrame({"id": [1], "value": [1.0]})
    current = pl.DataFrame({"id": [1], "value": [1.0], "detail": [0.1]})

    try:
        module._compare_frames(previous, current, keys=("id",))
    except ValueError as exc:
        assert "Schema drift" in str(exc)
    else:
        raise AssertionError("Expected additive columns to fail in strict mode")


def test_revision_audit_refuses_removed_columns_in_additive_mode() -> None:
    previous = pl.DataFrame({"id": [1], "value": [1.0], "detail": [0.1]})
    current = pl.DataFrame({"id": [1], "value": [1.0]})

    try:
        module._compare_frames(
            previous,
            current,
            keys=("id",),
            allow_additive_schema=True,
        )
    except ValueError as exc:
        assert "removed=['detail']" in str(exc)
    else:
        raise AssertionError("Expected removed columns to fail closed")
