#!/usr/bin/env python3
"""Build the offline human report for a four-scenario refresh replay audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.replay.refresh_attribution import (
    RefreshAttributionInputs,
    ScenarioArtifacts,
    build_refresh_attribution,
)
from alpharank.reporting.refresh_replay_html import write_refresh_replay_html


def parse_args() -> argparse.Namespace:
    """Parse the canonical audit plus price-only and SEC-only ablations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--price-only-legacy", type=Path, required=True)
    parser.add_argument("--price-only-boosting", type=Path, required=True)
    parser.add_argument("--price-only-common", type=Path, required=True)
    parser.add_argument("--sec-only-legacy", type=Path, required=True)
    parser.add_argument("--sec-only-boosting", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    """Write machine-readable attribution and its self-contained HTML view."""

    args = parse_args()
    audit = _read_audit(args.audit_report)
    paths = _audit_paths(audit)
    scenarios = (
        ScenarioArtifacts(
            "baseline",
            "Prix baseline + SEC baseline",
            paths["baseline_legacy"],
            paths["baseline_boosting"],
            "passe",
            paths["baseline_common"],
        ),
        ScenarioArtifacts(
            "price_only",
            "Prix candidats + SEC baseline",
            args.price_only_legacy,
            args.price_only_boosting,
            "passe",
            args.price_only_common,
        ),
        ScenarioArtifacts(
            "sec_only",
            "Prix baseline + SEC candidat",
            args.sec_only_legacy,
            args.sec_only_boosting,
            "bloqué sur CVC.US",
        ),
        ScenarioArtifacts(
            "full",
            "Prix candidats + SEC candidat",
            paths["candidate_legacy"],
            paths["candidate_boosting"],
            "bloqué sur CVC.US",
        ),
    )
    report = build_refresh_attribution(
        RefreshAttributionInputs(audit_report=args.audit_report, scenarios=scenarios)
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    normalized = json.loads(serialized)
    output_json = args.output_json or args.output_html.with_name(
        f"{args.output_html.stem}_attribution.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(serialized, encoding="utf-8")
    write_refresh_replay_html(normalized, args.output_html)
    print(f"Attribution JSON: {output_json.resolve()}")
    print(f"Human report: {args.output_html.resolve()}")
    return 0


def _read_audit(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected audit JSON object: {path}")
    return value


def _audit_paths(audit: dict[str, object]) -> dict[str, Path]:
    raw = audit.get("inputs")
    if not isinstance(raw, dict):
        raise ValueError("Audit report lacks inputs")
    required = (
        "baseline_legacy",
        "baseline_boosting",
        "baseline_common",
        "candidate_legacy",
        "candidate_boosting",
    )
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise ValueError(f"Audit report lacks replay paths: {missing}")
    return {name: Path(str(raw[name])) for name in required}


if __name__ == "__main__":
    raise SystemExit(main())
