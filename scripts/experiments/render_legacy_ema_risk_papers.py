#!/usr/bin/env python3
"""Compatibility facade; implementation lives under a named responsibility."""

from pathlib import Path

from alpharank.utils.script_compat import expose_script_implementation

_IMPLEMENTATION = expose_script_implementation(
    globals(),
    target=Path(__file__).resolve().parent / "legacy/render_legacy_ema_risk_papers.py",
    module_name="alpharank_script_compat_render_legacy_ema_risk_papers",
)

if __name__ == "__main__":
    raise SystemExit(_IMPLEMENTATION.main())
