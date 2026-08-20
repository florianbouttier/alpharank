"""Deterministic Ruff debt baseline and differential regression gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

RUFF_BASELINE_SCHEMA_VERSION = 1
DEFAULT_SCOPE = ("src", "scripts", "tests")


def run_ruff(
    root: Path,
    *,
    ruff_executable: str = "ruff",
    scope: Sequence[str] = DEFAULT_SCOPE,
) -> tuple[list[dict[str, object]], str]:
    """Run Ruff without failing on existing diagnostics and parse its JSON output."""

    resolved_root = root.resolve()
    result = subprocess.run(
        [
            ruff_executable,
            "check",
            *scope,
            "--output-format=json",
            "--exit-zero",
        ],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(result.stdout)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("Ruff JSON output must be a list of diagnostic objects")
    version = subprocess.run(
        [ruff_executable, "--version"],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return raw, version


def build_ruff_baseline(
    root: Path,
    diagnostics: Sequence[dict[str, object]],
    *,
    ruff_version: str,
    scope: Sequence[str] = DEFAULT_SCOPE,
) -> dict[str, object]:
    """Build a stable, timestamp-free multiset of Ruff diagnostic fingerprints."""

    normalized = [_normalize_diagnostic(root, row) for row in diagnostics]
    counts = Counter(row["fingerprint"] for row in normalized)
    examples = {str(row["fingerprint"]): row for row in normalized}
    fingerprints = []
    for fingerprint, count in sorted(counts.items()):
        example = examples[fingerprint]
        fingerprints.append(
            {
                "fingerprint": fingerprint,
                "count": count,
                "path": example["path"],
                "code": example["code"],
                "message": example["message"],
                "source": example["source"],
            }
        )
    return {
        "schema_version": RUFF_BASELINE_SCHEMA_VERSION,
        "tool": "ruff",
        "tool_version_at_baseline": ruff_version,
        "scope": list(scope),
        "total_diagnostics": len(normalized),
        "diagnostics_by_code": _counter_dict(row["code"] for row in normalized),
        "diagnostics_by_path": _counter_dict(row["path"] for row in normalized),
        "fingerprints": fingerprints,
    }


def compare_ruff_baseline(
    baseline: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    """Compare two baselines and identify only diagnostic-count regressions."""

    _validate_baseline(baseline)
    _validate_baseline(current)
    expected = _fingerprint_counts(baseline)
    observed = _fingerprint_counts(current)
    current_examples = _fingerprint_examples(current)
    baseline_examples = _fingerprint_examples(baseline)
    regressions = _deltas(observed, expected, current_examples)
    resolved = _deltas(expected, observed, baseline_examples)
    return {
        "schema_version": RUFF_BASELINE_SCHEMA_VERSION,
        "passed": not regressions,
        "baseline_total": _require_int(baseline["total_diagnostics"], "baseline total"),
        "current_total": _require_int(current["total_diagnostics"], "current total"),
        "new_diagnostic_count": sum(
            _require_int(row["count_delta"], "regression count") for row in regressions
        ),
        "resolved_diagnostic_count": sum(
            _require_int(row["count_delta"], "resolved count") for row in resolved
        ),
        "regressions": regressions,
        "resolved": resolved,
        "current_by_code": current["diagnostics_by_code"],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a deterministic JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_baseline(path: Path) -> dict[str, object]:
    """Load and validate a baseline file."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Ruff baseline must be a JSON object: {path}")
    _validate_baseline(raw)
    return raw


def _normalize_diagnostic(root: Path, row: dict[str, object]) -> dict[str, str]:
    filename = row.get("filename")
    code = row.get("code")
    message = row.get("message")
    location = row.get("location")
    if not isinstance(filename, str) or not isinstance(code, str) or not isinstance(message, str):
        raise ValueError("Ruff diagnostic is missing filename, code or message")
    if not isinstance(location, dict) or not isinstance(location.get("row"), int):
        raise ValueError("Ruff diagnostic is missing a numeric location row")
    path = Path(filename).resolve()
    try:
        relative_path = path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Ruff diagnostic is outside repository root: {path}") from error
    source_lines = path.read_text(encoding="utf-8").splitlines()
    row_index = int(location["row"]) - 1
    source = source_lines[row_index].strip() if 0 <= row_index < len(source_lines) else ""
    identity = {
        "path": relative_path,
        "code": code,
        "message": message,
        "source": source,
    }
    serialized = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {**identity, "fingerprint": hashlib.sha256(serialized).hexdigest()}


def _counter_dict(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Ruff baseline {label} must be an integer")
    return value


def _validate_baseline(payload: dict[str, object]) -> None:
    if payload.get("schema_version") != RUFF_BASELINE_SCHEMA_VERSION:
        raise ValueError("Unsupported Ruff baseline schema version")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, list):
        raise ValueError("Ruff baseline fingerprints must be a list")
    required = {"fingerprint", "count", "path", "code", "message", "source"}
    for row in fingerprints:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Ruff baseline contains an invalid fingerprint row")
        if not isinstance(row["count"], int) or row["count"] <= 0:
            raise ValueError("Ruff baseline fingerprint counts must be positive integers")


def _fingerprint_counts(payload: dict[str, object]) -> Counter[str]:
    fingerprints = payload["fingerprints"]
    if not isinstance(fingerprints, list):
        raise ValueError("Ruff baseline fingerprints must be a list")
    return Counter(
        {
            str(row["fingerprint"]): int(row["count"])
            for row in fingerprints
            if isinstance(row, dict)
        }
    )


def _fingerprint_examples(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    fingerprints = payload["fingerprints"]
    if not isinstance(fingerprints, list):
        raise ValueError("Ruff baseline fingerprints must be a list")
    return {str(row["fingerprint"]): row for row in fingerprints if isinstance(row, dict)}


def _deltas(
    left: Counter[str],
    right: Counter[str],
    examples: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for fingerprint in sorted(left):
        delta = left[fingerprint] - right[fingerprint]
        if delta <= 0:
            continue
        example = examples[fingerprint]
        rows.append(
            {
                "fingerprint": fingerprint,
                "count_delta": delta,
                "path": example["path"],
                "code": example["code"],
                "message": example["message"],
                "source": example["source"],
            }
        )
    return rows
