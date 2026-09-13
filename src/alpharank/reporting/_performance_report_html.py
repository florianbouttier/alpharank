from __future__ import annotations

import base64
import gzip
import json
from typing import Any

from alpharank.reporting._performance_report_chart_script import (
    PERFORMANCE_REPORT_CHART_SCRIPT,
)
from alpharank.reporting._performance_report_composer_script import (
    PERFORMANCE_REPORT_COMPOSER_SCRIPT,
)
from alpharank.reporting._performance_report_matrix_script import (
    PERFORMANCE_REPORT_MATRIX_SCRIPT,
)
from alpharank.reporting._performance_report_script import PERFORMANCE_REPORT_SCRIPT
from alpharank.reporting._performance_report_studio_script import PERFORMANCE_REPORT_STUDIO_SCRIPT
from alpharank.reporting._performance_report_styles import PERFORMANCE_REPORT_STYLES


def render_performance_report_html(payload: dict[str, Any]) -> str:
    """Render one autonomous browser report with an inline compressed payload."""

    packed = gzip.compress(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )
    encoded = base64.b64encode(packed).decode("ascii")
    return (
        '<!doctype html>\n<html lang="fr">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>AlphaRank · Rapport de backtest complet</title>\n"
        f"<style>{PERFORMANCE_REPORT_STYLES}</style>\n"
        "</head>\n<body>\n"
        '<div id="loading" class="loading">Chargement du rapport canonique…</div>\n'
        '<div id="app" class="shell" hidden>\n' + _sidebar() + _main() + "</div>\n"
        f"<script>const PAYLOAD_GZIP_BASE64={json.dumps(encoded)};\n"
        f"{PERFORMANCE_REPORT_COMPOSER_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_STUDIO_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_CHART_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_MATRIX_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )


def _sidebar() -> str:
    return """
<aside class="sidebar">
  <div class="brand">
    <div class="brand-mark">AR</div>
    <div><strong>AlphaRank</strong><small>Backtest reporting</small></div>
  </div>
  <nav>
    <div class="nav-label">Performance</div>
    <a class="nav-link is-active" href="#overview">Studio de comparaison</a>
    <div class="nav-label">Audit</div>
    <a class="nav-link" href="#current-portfolio">Portefeuille en vigueur</a>
    <a class="nav-link" href="#portfolios">Portefeuilles historiques</a>
    <a class="nav-link" href="#methodologies">Méthodologies</a>
    <a class="nav-link" href="#lineage">Lignée et contrats</a>
  </nav>
  <div class="sidebar-meta">
    <div id="report-calendar">—</div>
    <div id="report-generated">—</div>
  </div>
</aside>
"""


def _main() -> str:
    return (
        """
<main><div class="content">
  <header class="hero">
    <div>
      <span class="eyebrow">Standard de performance · REPORT-012</span>
      <h1>Rapport de backtest complet</h1>
      <p>Legacy, Boosting natif, variantes filtrées par tendance et SPY sur un même
      calendrier. Les KPI de chaque fenêtre sont pré-calculés par le moteur commun ;
      cette page ne possède aucune formule financière parallèle.</p>
    </div>
    <div class="status-badge"><strong>Statut de preuve</strong><br><span id="status-message">—</span></div>
  </header>
"""
        + _performance_sections()
        + _audit_sections()
        + """
</div></main>
"""
    )


def _performance_sections() -> str:
    return (
        """
  <section class="section studio-section" id="overview">
    <article class="analysis-studio" id="analysis-studio">
      <div class="studio-head">
        <div><span class="section-kicker">01 · Studio de comparaison</span><h2>Stratégies et portefeuilles, au même endroit</h2></div>
        <div class="mode-switch" role="tablist" aria-label="Type de comparaison">
          <button class="is-active" type="button" role="tab" aria-selected="true" data-analysis-mode="strategies">Stratégies</button>
          <button type="button" role="tab" aria-selected="false" data-analysis-mode="composer">Portefeuille composé</button>
        </div>
      </div>
"""
        + _studio_toolbar()
        + """
      <div class="studio-context">
        <strong id="window-label">—</strong>
        <span>SPY total return reste toujours la référence.</span>
      </div>
      <div class="studio-outcome" id="studio-outcome" aria-live="polite"></div>
      <div class="kpi-grid" id="studio-kpis"></div>
"""
        + _studio_chart()
        + _studio_drawers()
        + """
      <aside class="studio-contract composer-only" id="composer-contract" hidden>
        <strong>Laboratoire post-hoc, non promu.</strong>
        Équipondération mensuelle des poches ; rendements déjà nets de leurs frais propres et aucun coût supplémentaire entre poches. Un titre détenu par plusieurs stratégies reste exposé dans chacune.
      </aside>
    </article>
  </section>
"""
    )


def _studio_toolbar() -> str:
    return """
      <div class="toolbar" aria-label="Filtres de performance">
        <label>Début<select id="start-month"></select></label>
        <label>Fin<select id="end-month"></select></label>
        <div class="curve-control" id="strategy-mode-controls">
          <span class="field-label">Stratégies comparées à SPY</span>
          <details class="multi-select" id="curve-multiselect">
            <summary id="curve-select-label">Choisir les stratégies</summary>
            <div class="multi-select-menu">
              <div class="multi-select-actions">
                <button id="select-all-curves" type="button">Toutes</button>
                <button id="select-reference-curves" type="button">Legacy</button>
              </div>
              <div class="curve-options" id="curve-options"></div>
            </div>
          </details>
        </div>
        <div class="curve-control" id="composer-mode-controls" hidden>
          <span class="field-label">Poches du portefeuille</span>
          <details class="multi-select" id="composer-multiselect">
            <summary id="composer-select-label">Choisir les poches</summary>
            <div class="multi-select-menu composer-menu">
              <p class="composer-summary" id="composer-summary">—</p>
              <div class="multi-select-actions composer-actions">
                <button id="composer-reference" type="button">Legacy + tendance</button>
                <button id="composer-boosting-pair" type="button">Deux Boosting</button>
                <button id="composer-all" type="button">Toutes</button>
              </div>
              <div class="composer-options" id="composer-options"></div>
            </div>
          </details>
        </div>
        <button class="button secondary" id="reset-window" type="button">Toute la période</button>
      </div>
"""


def _studio_chart() -> str:
    return """
      <div class="studio-chart-panel">
        <div class="chart-head">
          <div><h3 id="studio-chart-title">Croissance composée</h3><p class="panel-subtitle" id="studio-chart-subtitle">—</p></div>
          <div class="chart-switch" role="tablist" aria-label="Vue graphique">
            <button class="is-active" type="button" role="tab" aria-selected="true" data-chart-view="wealth">Performance</button>
            <button type="button" role="tab" aria-selected="false" data-chart-view="drawdown">Drawdown</button>
            <button type="button" role="tab" aria-selected="false" data-chart-view="relative">Vs SPY</button>
          </div>
        </div>
        <div class="chart-stage">
          <canvas id="studio-chart" aria-label="Graphique de comparaison"></canvas>
          <div class="chart-tooltip" id="studio-chart-tooltip" hidden></div>
        </div>
        <div class="legend" id="studio-legend"></div>
      </div>
"""


def _studio_drawers() -> str:
    return """
      <div class="studio-drawers">
        <details class="studio-drawer" id="full-kpis-drawer">
          <summary><span>Tous les KPI de la sélection</span><small>33 mesures pour les stratégies · 7 pour un portefeuille composé</small></summary>
          <div class="drawer-content"><div class="table-wrap"><table class="metric-table"><thead><tr id="metric-head"></tr></thead><tbody id="metric-body"></tbody></table></div></div>
        </details>
        <details class="studio-drawer" id="model-cards-drawer">
          <summary><span>Model cards par année</span><small>Cumul depuis chaque année et années isolées</small></summary>
          <div class="drawer-content">
            <div class="matrix-controls">
              <button class="is-active" type="button" data-matrix-metric="cagr">CAGR</button>
              <button type="button" data-matrix-metric="annualized_volatility">Volatilité</button>
              <button type="button" data-matrix-metric="max_drawdown">Max drawdown</button>
            </div>
            <div class="matrix-block">
              <h3>Depuis chaque année jusqu'à la fin sélectionnée</h3>
              <p class="panel-subtitle" id="cumulative-matrix-window">—</p>
              <div class="heatmap-wrap"><div class="heatmap" id="cumulative-heatmap"></div></div>
              <div class="viridis-legend"><span>Faible</span><i class="viridis-bar"></i><span>Élevé</span><strong id="cumulative-matrix-caption"></strong></div>
            </div>
            <div class="matrix-block incremental-block">
              <h3>Chaque année isolée · incrémental</h3>
              <p class="panel-subtitle">Chaque cellule utilise seulement les mois de l'année indiquée, sans capital antérieur.</p>
              <div class="heatmap-wrap"><div class="heatmap" id="incremental-heatmap"></div></div>
              <div class="viridis-legend"><span>Faible</span><i class="viridis-bar"></i><span>Élevé</span><strong id="incremental-matrix-caption"></strong></div>
            </div>
          </div>
        </details>
        <details class="studio-drawer composer-only" id="composer-correlation" hidden>
          <summary><span>Corrélations entre les poches</span><small>Pearson sur les rendements mensuels sélectionnés</small></summary>
          <div class="drawer-content"><div class="table-wrap"><table id="composer-correlation-matrix"></table></div></div>
        </details>
      </div>
"""


def _audit_sections() -> str:
    return """
  <section class="section" id="current-portfolio">
    <div class="section-head">
      <div><span class="section-kicker">02 · Portefeuille en vigueur</span><h2 id="current-portfolio-title">Portefeuille en vigueur</h2></div>
      <p>Le mois courant reste séparé des KPI tant que son rendement complet n'est pas réalisé.</p>
    </div>
    <div class="current-portfolio-meta">
      <article><span>Dernière séance observée</span><strong id="current-as-of-date">—</strong></article>
      <article><span>Mois de décision</span><strong id="current-decision-month">—</strong></article>
      <article><span>Mois de détention</span><strong id="current-holding-month">—</strong></article>
      <article class="is-pending"><span>Statut</span><strong>Rendement mensuel non réalisé</strong></article>
    </div>
    <div class="current-portfolio-controls">
      <label>Stratégie<select id="current-portfolio-strategy"></select></label>
      <button class="button secondary" id="export-current-holdings" type="button">Exporter le panier courant</button>
    </div>
    <p class="portfolio-summary" id="current-portfolio-summary"></p>
    <div class="table-wrap"><table><thead><tr><th>Ticker</th><th>Rang</th><th>Poids cible</th><th>Score</th><th>Secteur</th><th>Votes</th></tr></thead><tbody id="current-holdings-body"></tbody></table></div>
  </section>
  <section class="section" id="portfolios">
    <div class="section-head"><div><span class="section-kicker">03 · Historique réalisé</span><h2>Tous les portefeuilles historiques</h2></div><p>Poids décidés à t, rendement réalisé pendant t+1, score OOS lorsqu'il existe.</p></div>
    <div class="portfolio-controls">
      <label>Stratégie<select id="portfolio-strategy"></select></label>
      <label>Mois de détention<select id="portfolio-month"></select></label>
      <label>Filtrer un ticker<input id="ticker-search" type="search" placeholder="SATS, DELL…"></label>
      <button class="button secondary" id="export-holdings" type="button">Exporter CSV</button>
    </div>
    <p class="portfolio-summary" id="portfolio-summary"></p>
    <div class="table-wrap"><table><thead><tr><th>Ticker</th><th>Rang</th><th>Poids</th><th>Score</th><th>Secteur</th><th>Rendement réalisé</th><th>Votes</th></tr></thead><tbody id="holdings-body"></tbody></table></div>
    <div class="pager"><button id="page-prev" type="button">Précédent</button><span id="page-label">—</span><button id="page-next" type="button">Suivant</button></div>
  </section>
  <section class="section" id="methodologies">
    <div class="section-head"><div><span class="section-kicker">04 · Méthodes</span><h2>Règles et pseudo-codes</h2></div><p>Projection lisible des contrats canoniques ; aucun statut R&D n'est présenté comme une recommandation.</p></div>
    <div class="method-grid" id="method-grid"></div>
  </section>
  <section class="section" id="lineage">
    <div class="section-head"><div><span class="section-kicker">05 · Audit</span><h2>Lignée, hashes et conventions</h2></div><p>Le rapport cite ses entrées ; il ne résout jamais un artefact au nom « latest ».</p></div>
    <div class="lineage-grid">
      <article class="lineage-card"><h3>Contrats économiques</h3><dl class="definition" id="lineage-contracts"></dl></article>
      <article class="lineage-card"><h3>Snapshot et sources</h3><dl class="definition" id="lineage-data"></dl></article>
    </div>
  </section>
"""
