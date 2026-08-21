#!/usr/bin/env python3
"""Compatibility facade; implementation lives under a named responsibility."""

from pathlib import Path

from alpharank.utils.script_compat import expose_script_implementation

_IMPLEMENTATION = expose_script_implementation(
    globals(),
    target=Path(__file__).resolve().parent / "legacy/analyze_legacy_return_attribution.py",
    module_name="alpharank_script_compat_analyze_legacy_return_attribution",
)

if __name__ == "__main__":
    raise SystemExit(_IMPLEMENTATION.main())
