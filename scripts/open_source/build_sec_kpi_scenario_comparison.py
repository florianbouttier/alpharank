#!/usr/bin/env python3
"""Compatibility facade for the documented SEC scenario comparison command."""

from pathlib import Path

from alpharank.utils.script_compat import expose_script_implementation

_IMPLEMENTATION = expose_script_implementation(
    globals(),
    target=(
        Path(__file__).resolve().parent
        / "reporting/build_sec_kpi_scenario_comparison.py"
    ),
    module_name="alpharank_script_compat_build_sec_kpi_scenario_comparison",
)


if __name__ == "__main__":
    raise SystemExit(_IMPLEMENTATION.main())
