"""Data-lineage checks required before comparing portfolio strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def input_hashes_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Extract normalized input hashes from Legacy or boosting manifests."""

    run_hashes = manifest.get("run_config", {}).get("source_input_sha256")
    if isinstance(run_hashes, Mapping):
        return {str(key): str(value) for key, value in run_hashes.items()}

    input_paths = manifest.get("input_paths")
    if isinstance(input_paths, Mapping):
        hashes: dict[str, str] = {}
        for key, value in input_paths.items():
            if isinstance(value, Mapping) and value.get("sha256"):
                hashes[str(key)] = str(value["sha256"])
        if hashes:
            return hashes

    raise ValueError("Manifest does not expose input data hashes.")


def ticker_exclusions_from_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract full-trajectory data-quality exclusions from a run manifest."""

    for section_name in ("run_config", "config"):
        section = manifest.get(section_name)
        if not isinstance(section, Mapping):
            continue
        raw = section.get("excluded_tickers")
        if isinstance(raw, (list, tuple)):
            return tuple(
                sorted(
                    {
                        str(value).strip().upper()
                        for value in raw
                        if str(value).strip()
                    }
                )
            )
    raise ValueError("Manifest does not expose excluded_tickers.")


def compare_input_hashes(
    left: Mapping[str, str],
    right: Mapping[str, str],
    *,
    required_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Compare the data files used by both strategies, not their output hashes."""

    keys = sorted(required_keys if required_keys is not None else set(left) | set(right))
    missing_left = [key for key in keys if key not in left]
    missing_right = [key for key in keys if key not in right]
    differing = [
        key
        for key in keys
        if key in left and key in right and left[key] != right[key]
    ]
    return {
        "required_keys": keys,
        "missing_left": missing_left,
        "missing_right": missing_right,
        "differing_keys": differing,
        "matching_keys": [
            key
            for key in keys
            if key in left and key in right and left[key] == right[key]
        ],
        "passed": not missing_left and not missing_right and not differing,
    }


def compare_ticker_exclusions(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> dict[str, Any]:
    """Compare data-quality universe removals independently of model filters."""

    left_set = set(left)
    right_set = set(right)
    return {
        "left_excluded_tickers": sorted(left_set),
        "right_excluded_tickers": sorted(right_set),
        "missing_left": sorted(right_set - left_set),
        "missing_right": sorted(left_set - right_set),
        "passed": left_set == right_set,
    }


def require_matching_data_contexts(
    left_manifest: Path,
    right_manifest: Path,
    *,
    required_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Fail closed when two strategy artifacts do not share input data."""

    left = input_hashes_from_manifest(load_manifest(left_manifest))
    right = input_hashes_from_manifest(load_manifest(right_manifest))
    report = compare_input_hashes(left, right, required_keys=required_keys)
    report.update(
        {
            "left_manifest": str(left_manifest.resolve()),
            "right_manifest": str(right_manifest.resolve()),
        }
    )
    if not report["passed"]:
        raise ValueError(
            "Portfolio comparison data contexts differ: "
            f"missing_left={report['missing_left']}, "
            f"missing_right={report['missing_right']}, "
            f"differing_keys={report['differing_keys']}"
        )
    return report


def require_matching_ticker_exclusions(
    left_manifest: Path,
    right_manifest: Path,
) -> dict[str, Any]:
    """Fail closed when preprocessing quarantines differ across strategies."""

    left = ticker_exclusions_from_manifest(load_manifest(left_manifest))
    right = ticker_exclusions_from_manifest(load_manifest(right_manifest))
    report = compare_ticker_exclusions(left, right)
    report.update(
        {
            "left_manifest": str(left_manifest.resolve()),
            "right_manifest": str(right_manifest.resolve()),
        }
    )
    if not report["passed"]:
        raise ValueError(
            "Portfolio comparison ticker exclusions differ: "
            f"missing_left={report['missing_left']}, "
            f"missing_right={report['missing_right']}"
        )
    return report
