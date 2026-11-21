"""Reusable offline frontend for strategy analysis, without financial calculations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

# Plotly has no typing marker; keep the exception at this presentation-only boundary.
from plotly.offline import get_plotlyjs, get_plotlyjs_version  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class AnalysisUiAssets:
    """Installed chart runtime and the maintained presentation-only adapter."""

    runtime: str
    runtime_version: str
    script: str
    styles: str


def load_analysis_ui_assets() -> AnalysisUiAssets:
    """Read bundled UI assets at render time; no network or import-time IO."""
    root = files(__package__)
    notice = root.joinpath("PLOTLY_LICENSE.txt").read_text(encoding="utf-8")
    return AnalysisUiAssets(
        runtime="/*\n" + notice + "\n*/\n" + get_plotlyjs(),
        runtime_version=get_plotlyjs_version(),
        script=root.joinpath("charts.js").read_text(encoding="utf-8"),
        styles=root.joinpath("layout.css").read_text(encoding="utf-8"),
    )


__all__ = ["AnalysisUiAssets", "load_analysis_ui_assets"]
