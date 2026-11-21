"""Reusable offline React frontend for source-backed strategy analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class AnalysisUiAssets:
    """Versioned frontend bundle and styles; no acquisition or financial logic."""

    bundle: str
    styles: str
    licenses: str
    chart_version: str
    react_version: str


def load_analysis_ui_assets() -> AnalysisUiAssets:
    """Read the reproducibly built local assets; never download code at render time."""
    root = files(__package__)
    package = json.loads(root.joinpath("package.json").read_text(encoding="utf-8"))
    return AnalysisUiAssets(
        bundle=root.joinpath("client.bundle.js").read_text(encoding="utf-8"),
        styles=root.joinpath("layout.css").read_text(encoding="utf-8"),
        licenses=root.joinpath("THIRD_PARTY_LICENSES.txt").read_text(encoding="utf-8"),
        chart_version=str(package["dependencies"]["lightweight-charts"]),
        react_version=str(package["dependencies"]["react"]),
    )


__all__ = ["AnalysisUiAssets", "load_analysis_ui_assets"]
