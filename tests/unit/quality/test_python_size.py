from __future__ import annotations

from alpharank.quality.python_size import compare_python_size_baselines

POLICY = {
    "complexity_maximum": 10,
    "exception_policy": "owner_approval_required",
    "excluded_directories": ["_archive", "_old"],
    "function_maximum_lines": 80,
    "library_module_maximum_lines": 800,
    "scope_roots": ["src", "scripts", "tests"],
    "script_module_maximum_lines": 250,
}


def _baseline(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline_id": "fixture",
        "policy": POLICY,
        "tool_version_at_baseline": "ruff fixture",
        "summary": {},
        "violations": list(rows),
    }


def _row(*, kind: str, symbol: str, measured: int) -> dict[str, object]:
    return {
        "kind": kind,
        "path": "src/alpharank/example.py",
        "symbol": symbol,
        "line": 10,
        "measured": measured,
        "limit": 80 if kind == "function_lines" else 10,
    }


def test_differential_size_gate_allows_reductions() -> None:
    report = compare_python_size_baselines(
        _baseline(_row(kind="function_lines", symbol="large", measured=100)),
        _baseline(_row(kind="function_lines", symbol="large", measured=90)),
    )

    assert report["passed"] is True
    assert report["regression_count"] == 0


def test_differential_size_gate_rejects_new_and_increased_debt() -> None:
    report = compare_python_size_baselines(
        _baseline(_row(kind="function_lines", symbol="large", measured=100)),
        _baseline(
            _row(kind="function_lines", symbol="large", measured=101),
            _row(kind="complexity", symbol="branchy", measured=11),
        ),
    )

    assert report["passed"] is False
    assert report["regression_count"] == 2
    assert {row["reason"] for row in report["regressions"]} == {
        "measurement_increase",
        "new_violation",
    }
