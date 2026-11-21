from __future__ import annotations

import base64
import gzip
import json
from html import escape
from typing import Any

from alpharank.reporting.analysis_ui import load_analysis_ui_assets


def render_performance_report_html(payload: dict[str, Any]) -> str:
    """Render the shared React analysis and its complete frozen source payload."""
    encoded = base64.b64encode(
        gzip.compress(
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode(),
            compresslevel=9,
            mtime=0,
        )
    ).decode("ascii")
    assets = load_analysis_ui_assets()
    boot = r"""
const PAYLOAD_GZIP_BASE64=__PACKED__;
async function openReport() {
  const bytes = Uint8Array.from(atob(PAYLOAD_GZIP_BASE64), value => value.charCodeAt(0));
  if (!("DecompressionStream" in window)) throw new Error("Ce navigateur ne prend pas en charge la décompression du rapport.");
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const payload = JSON.parse(await new Response(stream).text());
  AlphaRankAnalysis.mount(document.getElementById("app"), payload);
}
openReport().catch(error => {
  document.getElementById("app").textContent = "Chargement impossible : " + error.message;
});
""".replace("__PACKED__", json.dumps(encoded))
    return (
        '<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(str(payload.get('title', 'AlphaRank · Analyse des performances')))}</title>"
        f'<meta name="alpharank-chart-engine" content="Lightweight Charts {assets.chart_version}">'
        f'<meta name="alpharank-ui-framework" content="React {assets.react_version}">'
        f"<style>{assets.styles}</style></head><body>"
        '<div id="app"><p class="initial-loading" role="status">Chargement de l’analyse…</p></div>'
        f"<noscript>{_static_summary(payload)}</noscript>"
        f"<script>/*\n{assets.licenses}\n*/\n{assets.bundle}</script>"
        f"<script>{boot}</script></body></html>\n"
    )


def _static_summary(payload: dict[str, Any]) -> str:
    """Provide readable source context without inventing a second KPI calculation."""
    labels = ", ".join(str(label) for label in payload.get("strategy_order", []))
    calendar = payload.get("calendar", {})
    return (
        "<h1>AlphaRank · Analyse des performances</h1>"
        "<p>Activez JavaScript pour les graphiques et les filtres interactifs.</p>"
        f"<p>Stratégies conservées : {escape(labels)}.</p>"
        f"<p>Historique réalisé : {escape(str(calendar.get('start', '')))}"
        f" au {escape(str(calendar.get('end', '')))}.</p>"
        "<p>Les fenêtres quotidiennes personnalisées utilisent le service local Python "
        "en lecture seule ; aucun calcul de KPI n’est dupliqué dans le navigateur.</p>"
    )
