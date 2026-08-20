from __future__ import annotations

from alpharank.quality.ruff_baseline import compare_ruff_baseline


def _baseline(*, first_count: int, include_second: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = [
        {
            "fingerprint": "first",
            "count": first_count,
            "path": "src/one.py",
            "code": "F401",
            "message": "unused import",
            "source": "import unused",
        }
    ]
    if include_second:
        rows.append(
            {
                "fingerprint": "second",
                "count": 1,
                "path": "src/two.py",
                "code": "F821",
                "message": "undefined name",
                "source": "missing_name",
            }
        )
    return {
        "schema_version": 1,
        "tool": "ruff",
        "tool_version_at_baseline": "ruff fixture",
        "scope": ["src", "scripts", "tests"],
        "total_diagnostics": first_count + int(include_second),
        "diagnostics_by_code": {"F401": first_count},
        "diagnostics_by_path": {"src/one.py": first_count},
        "fingerprints": rows,
    }


def test_differential_baseline_allows_resolved_historical_diagnostics() -> None:
    report = compare_ruff_baseline(
        _baseline(first_count=2),
        _baseline(first_count=1),
    )

    assert report["passed"] is True
    assert report["new_diagnostic_count"] == 0
    assert report["resolved_diagnostic_count"] == 1


def test_differential_baseline_rejects_new_or_duplicated_diagnostics() -> None:
    report = compare_ruff_baseline(
        _baseline(first_count=1),
        _baseline(first_count=2, include_second=True),
    )

    assert report["passed"] is False
    assert report["new_diagnostic_count"] == 2
    assert {row["fingerprint"] for row in report["regressions"]} == {"first", "second"}
