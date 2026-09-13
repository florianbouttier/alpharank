from __future__ import annotations

PERFORMANCE_REPORT_STUDIO_SCRIPT = r"""
const STUDIO_DISPLAY_METRICS = [
  "cagr", "total_return", "annualized_volatility", "max_drawdown", "sharpe", "sortino",
];
const STUDIO_CHART_VIEWS = new Set(["wealth", "drawdown", "relative"]);

function initializeStudioControls() {
  document.querySelectorAll("[data-analysis-mode]").forEach(button => {
    button.addEventListener("click", () => setAnalysisMode(button.dataset.analysisMode));
  });
  document.querySelectorAll("[data-chart-view]").forEach(button => {
    button.addEventListener("click", () => setChartView(button.dataset.chartView));
  });
}

function setAnalysisMode(mode) {
  if (mode !== "strategies" && mode !== "composer") return;
  state.analysisMode = mode;
  const isComposer = mode === "composer";
  document.querySelectorAll("[data-analysis-mode]").forEach(button => {
    const active = button.dataset.analysisMode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.getElementById("strategy-mode-controls").hidden = isComposer;
  document.getElementById("composer-mode-controls").hidden = !isComposer;
  document.querySelectorAll(".composer-only").forEach(element => {
    element.hidden = !isComposer;
  });
  document.getElementById("curve-multiselect").open = false;
  document.getElementById("composer-multiselect").open = false;
  renderStudio();
}

function setChartView(view) {
  if (!STUDIO_CHART_VIEWS.has(view)) return;
  state.chartView = view;
  document.querySelectorAll("[data-chart-view]").forEach(button => {
    const active = button.dataset.chartView === view;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  drawStudioChart();
}

function studioDisplaySeries() {
  return state.analysisMode === "composer"
    ? [COMPOSER_NAME, BENCHMARK_STRATEGY]
    : state.curves;
}

function studioMetricValueAtWindow(series, field, key=windowKey()) {
  if (series === COMPOSER_NAME) {
    const composer = state.data.portfolio_composer;
    const rows = composer.metric_windows[key] || [];
    const metricIndex = composer.metric_fields.indexOf(field);
    return metricIndex < 0 ? null : rows[composerCombinationIndex()]?.[metricIndex];
  }
  const rows = state.data.metric_windows[key] || [];
  const strategyIndex = state.data.strategy_order.indexOf(series);
  const metricIndex = state.data.metric_fields.indexOf(field);
  return metricIndex < 0 ? null : rows[strategyIndex]?.[metricIndex];
}

function studioComparisonState(field, series, value) {
  if (series === BENCHMARK_STRATEGY) return "benchmark";
  if (series === COMPOSER_NAME) return composerComparisonState(field, value);
  return comparisonState(field, series, value);
}

function comparisonDeltaText(field, value, benchmark) {
  if (!Number.isFinite(value) || !Number.isFinite(benchmark)) return "Écart indisponible";
  const difference = value - benchmark;
  const sign = difference > 0 ? "+" : difference < 0 ? "−" : "";
  const magnitude = Math.abs(difference);
  const type = METRICS[field]?.[1] || "num";
  if (type === "pct") {
    const points = (100 * magnitude).toLocaleString("fr-FR", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    return `${sign}${points} pts vs SPY`;
  }
  return `${sign}${magnitude.toLocaleString("fr-FR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 3,
  })} vs SPY`;
}

function renderStudioKpis() {
  const series = studioDisplaySeries();
  document.getElementById("studio-kpis").innerHTML = STUDIO_DISPLAY_METRICS.map(field => {
    const [label, type] = METRICS[field];
    const benchmark = studioMetricValueAtWindow(BENCHMARK_STRATEGY, field);
    const rows = series.map(name => {
      const value = studioMetricValueAtWindow(name, field);
      const status = studioComparisonState(field, name, value);
      const comparison = name === BENCHMARK_STRATEGY
        ? "Référence"
        : comparisonDeltaText(field, value, benchmark);
      const color = name === COMPOSER_NAME ? COMPOSER_COLOR : strategyMeta(name).color;
      return `<div class="kpi-strategy-row comparison-${status}">
        <span class="strategy-name"><i style="background:${color}"></i>${escapeHtml(name)}</span>
        <strong>${format(value, type)}</strong>
        <small class="delta-chip">${escapeHtml(comparison)}</small>
      </div>`;
    }).join("");
    return `<article class="kpi-card"><header><span>${label}</span><small>même fenêtre</small></header><div class="kpi-strategy-list">${rows}</div></article>`;
  }).join("");
}

function renderStudioOutcome() {
  const candidates = studioDisplaySeries().filter(name => name !== BENCHMARK_STRATEGY);
  const leader = candidates.reduce((best, name) => {
    const value = studioMetricValueAtWindow(name, "cagr");
    return !best || value > best.value ? {name, value} : best;
  }, null);
  const benchmark = studioMetricValueAtWindow(BENCHMARK_STRATEGY, "cagr");
  const status = leader ? studioComparisonState("cagr", leader.name, leader.value) : "neutral";
  const wins = leader ? STUDIO_DISPLAY_METRICS.filter(field => {
    const value = studioMetricValueAtWindow(leader.name, field);
    return studioComparisonState(field, leader.name, value) === "beats";
  }).length : 0;
  const element = document.getElementById("studio-outcome");
  element.className = `studio-outcome comparison-${status}`;
  element.innerHTML = leader ? `
    <span>${state.analysisMode === "composer" ? "Portefeuille composé" : "Leader CAGR de la sélection"}</span>
    <strong>${escapeHtml(leader.name)} · ${comparisonDeltaText("cagr", leader.value, benchmark)}</strong>
    <small>${wins} / ${STUDIO_DISPLAY_METRICS.length} KPI synthétiques surpassent SPY selon leur sens économique.</small>
  ` : "";
}

function renderStudio() {
  if (!state.data) return;
  renderStudioKpis();
  renderStudioOutcome();
  drawStudioChart();
  renderMetricTable();
  renderMatrices();
  if (state.analysisMode === "composer") renderComposerCorrelation();
}
"""
