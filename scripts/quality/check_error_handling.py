#!/usr/bin/env python3
"""Validate AlphaRank's explicit error-handling policy."""

from __future__ import annotations

import json
from pathlib import Path

from alpharank.quality.error_handling import audit_error_handling

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    report = audit_error_handling(PROJECT_ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
