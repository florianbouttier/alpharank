from __future__ import annotations

import base64
import gzip
import json
import re
import shutil
import subprocess

import pytest

from alpharank.reporting._performance_report_html import render_performance_report_html
from alpharank.reporting.analysis_ui import load_analysis_ui_assets


def test_shared_frontend_uses_react_financial_charts_and_complete_local_licenses() -> None:
    assets = load_analysis_ui_assets()
    assert "AlphaRankAnalysis" in assets.bundle
    assert "Plotly.react" not in assets.bundle
    assert assets.chart_version == "5.3.0"
    assert assets.react_version == "19.3.1"
    assert "TradingView" in assets.licenses
    assert "Apache License" in assets.licenses
    assert "Permission is hereby granted" in assets.licenses


def test_renderer_preserves_source_payload_and_escapes_untrusted_titles() -> None:
    payload = {
        "title": "<script>injected()</script>",
        "calendar": {},
        "strategy_order": ["Alpha"],
        "methodologies": [{"summary": "</script>source data"}],
    }
    html = render_performance_report_html(payload)
    match = re.search(r'const PAYLOAD_GZIP_BASE64="([^"]+)"', html)
    assert match is not None
    assert json.loads(gzip.decompress(base64.b64decode(match.group(1)))) == payload
    assert "<script>injected()</script>" not in html
    assert re.search(r"<script[^>]+src=", html) is None
    assert "<noscript>" in html


def test_bundled_runtime_and_bootstrap_are_valid_javascript_without_node_at_render_time() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required only for the JavaScript syntax gate")
    html = render_performance_report_html({"calendar": {}})
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert len(scripts) == 2
    for script in scripts:
        result = subprocess.run(
            [node, "--check"], input=script, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
