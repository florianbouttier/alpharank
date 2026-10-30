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
from alpharank.reporting._performance_report_frontier_script import (
    PERFORMANCE_REPORT_FRONTIER_SCRIPT,
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
        '<a class="skip-link" href="#overview">Aller au studio de comparaison</a>\n'
        '<div id="loading" class="loading">Chargement du rapport canonique…</div>\n'
        '<div id="app" class="shell" hidden>\n' + _sidebar() + _main() + "</div>\n"
        f"<script>const PAYLOAD_GZIP_BASE64={json.dumps(encoded)};\n"
        f"{PERFORMANCE_REPORT_COMPOSER_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_FRONTIER_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_STUDIO_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_CHART_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_MATRIX_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )


def _sidebar() -> str:
    return """
<header class="topbar" aria-label="Navigation du rapport">
  <a class="brand" href="#top" aria-label="AlphaRank · haut du rapport">
    <div class="brand-mark">AR</div>
    <div><strong>AlphaRank</strong><small>Performance desk</small></div>
  </a>
  <nav aria-label="Sections du rapport">
    <a class="nav-link is-active" href="#overview">Analyse</a>
    <a class="nav-link" href="#current-portfolio">Portefeuille</a>
    <a class="nav-link" href="#portfolios">Historique</a>
    <a class="nav-link" href="#methodologies">Méthodes</a>
    <a class="nav-link" href="#lineage">Sources</a>
  </nav>
  <div class="topbar-actions">
    <div class="topbar-meta" aria-label="Fraîcheur du rapport">
      <strong id="report-calendar">—</strong><span id="report-generated">—</span>
    </div>
    <button class="theme-toggle" id="theme-toggle" type="button" aria-pressed="false" aria-label="Activer le thème sombre">
      <span aria-hidden="true">◐</span><strong>Thème</strong>
    </button>
  </div>
</header>
"""


def _main() -> str:
    return (
        """
<main><div class="content">
  <header class="hero" id="top">
    <div class="hero-copy">
      <span class="eyebrow">Backtests comparables · REPORT-016</span>
      <h1>Performance desk</h1>
      <p>Une fenêtre, un benchmark, toutes les stratégies et leurs portefeuilles.</p>
    </div>
    <div class="hero-actions">
      <div class="status-badge"><span>Statut de preuve</span><strong id="status-message">—</strong></div>
    </div>
  </header>
  <section class="coverage-strip" aria-label="Couverture du rapport">
    <article><span>Stratégies</span><strong id="coverage-series">—</strong><small>SPY inclus</small></article>
    <article><span>Mesures</span><strong id="coverage-kpis">—</strong><small>moteur commun</small></article>
    <article><span>Historique</span><strong id="coverage-months">—</strong><small id="coverage-window">—</small></article>
    <article><span>Positions</span><strong id="coverage-holdings">—</strong><small>lignes historiques</small></article>
    <article class="coverage-source"><span>Source</span><strong>Payload embarqué</strong><small>autonome · sans réseau</small></article>
  </section>
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
        <div><span class="section-kicker">Comparateur</span><h2>Lire la performance, puis comprendre le risque</h2></div>
        <div class="mode-switch" role="tablist" aria-label="Type de comparaison">
          <button class="is-active" type="button" role="tab" aria-selected="true" data-analysis-mode="strategies">Stratégies</button>
          <button type="button" role="tab" aria-selected="false" data-analysis-mode="composer">Portefeuille composé</button>
          <button type="button" role="tab" aria-selected="false" data-analysis-mode="frontier">Frontières efficientes</button>
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
      <div class="studio-workspace">
"""
        + _studio_chart()
        + """
        <aside class="kpi-grid" id="studio-kpis" aria-label="KPI de la sélection"></aside>
      </div>
"""
        + _studio_drawers()
        + """
      <aside class="studio-contract composer-only" id="composer-contract" hidden>
        <strong>Laboratoire post-hoc, non promu.</strong>
        Équipondération mensuelle des poches ; rendements déjà nets de leurs frais propres et aucun coût supplémentaire entre poches. Un titre détenu par plusieurs stratégies reste exposé dans chacune.
      </aside>
      <section class="frontier-only frontier-studio" id="frontier-studio" hidden>
        <aside class="studio-contract">
          <strong>Diagnostic post-hoc, non promu.</strong>
          Trois poches distinctes, longues sans levier, rééquilibrées mensuellement par pas de 5 % et plafonnées à 70 %. La fenêtre « 2010 » commence en août 2011, première observation commune.
        </aside>
        <div class="frontier-recommendations" id="frontier-recommendations"></div>
        <div class="frontier-chart-grid">
          <article class="frontier-panel">
            <header><h3>Rendement / volatilité</h3><p>À gauche : moins de volatilité. En haut : plus de CAGR.</p></header>
            <div class="chart-stage"><canvas id="frontier-volatility-chart"></canvas><div class="chart-tooltip frontier-tooltip" hidden></div></div>
          </article>
          <article class="frontier-panel">
            <header><h3>Rendement / drawdown</h3><p>À gauche : drawdown absolu plus faible. En haut : plus de CAGR.</p></header>
            <div class="chart-stage"><canvas id="frontier-drawdown-chart"></canvas><div class="chart-tooltip frontier-tooltip" hidden></div></div>
          </article>
        </div>
        <details class="studio-drawer frontier-anchor-table" open>
          <summary><span>Recommandations sur plusieurs horizons</span><small>2010 demandé, puis 2015, 2020, 2023 et 2025 jusqu'à la dernière date commune</small></summary>
          <div class="drawer-content"><div class="table-wrap"><table><thead><tr><th>Fenêtre</th><th>Profil</th><th>Composition</th><th>CAGR</th><th>Volatilité</th><th>Max drawdown</th></tr></thead><tbody id="frontier-anchor-body"></tbody></table></div></div>
        </details>
      </section>
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
