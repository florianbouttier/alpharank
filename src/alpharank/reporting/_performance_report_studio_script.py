from __future__ import annotations

PERFORMANCE_REPORT_STUDIO_SCRIPT = r"""
const STUDIO_DISPLAY_METRICS = [
  "cagr", "total_return", "annualized_volatility", "max_drawdown", "sharpe", "sortino",
];
const STUDIO_CHART_VIEWS = new Set(["wealth", "drawdown", "relative"]);

function initializeStudioControls() {
  document.getElementById("reset-chart-zoom").addEventListener("click", () => AlphaRankStrategyCharts.reset(document.getElementById("studio-chart")));
  const recent = state.data.recent_valuation;
  document.querySelector('[data-analysis-mode="recent"]').hidden = !recent;
  if (recent) {
    state.recentWindow = recent.default_window;
    document.querySelector('[data-analysis-mode="strategies"]').textContent = "Backtest complet";
    const shortDate = value => new Date(`${value}T12:00:00Z`).toLocaleDateString("fr-FR", {day: "2-digit", month: "2-digit"});
    document.getElementById("recent-window").innerHTML = Object.values(recent.windows)
      .map(window => option(window.name, `${window.name.replaceAll("_", " ")} · ${shortDate(window.start_date)} → ${shortDate(window.end_date)}`)).join("");
    document.getElementById("recent-window").value = state.recentWindow;
    document.getElementById("recent-window").addEventListener("change", event => {
      state.recentWindow = event.target.value;
      renderStudio();
    });
  }
  document.querySelectorAll("[data-analysis-mode]").forEach(button => {
    button.addEventListener("click", () => setAnalysisMode(button.dataset.analysisMode));
  });
  document.querySelectorAll("[data-chart-view]").forEach(button => {
    button.addEventListener("click", () => setChartView(button.dataset.chartView));
  });
}

function setAnalysisMode(mode) {
  if (!new Set(["recent", "strategies", "composer", "frontier"]).has(mode)) return;
  if (mode === "recent" && !state.data.recent_valuation) return;
  state.analysisMode = mode;
  const isComposer = mode === "composer";
  const isFrontier = mode === "frontier";
  const isRecent = mode === "recent";
  document.getElementById("analysis-studio").dataset.mode = mode;
  document.getElementById("series-selector").hidden = isFrontier;
  document.querySelector(".period-actions").hidden = isRecent;
  document.getElementById("recent-period-control").hidden = !isRecent;
  document.getElementById("recent-contract").hidden = !isRecent;
  document.getElementById("model-cards-drawer").hidden = isRecent;
  document.getElementById("full-kpis-caption").textContent = isRecent
    ? "Suivi quotidien brut · aucune annualisation d'une fenêtre courte"
    : "33 mesures pour les stratégies · 7 pour un portefeuille composé";
  document.querySelectorAll("[data-analysis-mode]").forEach(button => {
    const active = button.dataset.analysisMode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.getElementById("strategy-mode-controls").hidden = isComposer || isFrontier;
  document.getElementById("composer-mode-controls").hidden = !isComposer;
  document.querySelectorAll(".composer-only").forEach(element => {
    element.hidden = !isComposer;
  });
  document.querySelectorAll(".frontier-only").forEach(element => {
    element.hidden = !isFrontier;
  });
  document.getElementById("studio-outcome").hidden = isFrontier;
  document.getElementById("studio-kpis").hidden = isFrontier;
  document.querySelector(".studio-chart-panel").hidden = isFrontier;
  document.querySelector(".studio-drawers").hidden = isFrontier;
  document.getElementById("composer-multiselect").open = isComposer;
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
  if (state.analysisMode === "recent") return recentWindow().metrics[series]?.[field];
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

function compactStrategyName(name) {
  if (name === BENCHMARK_STRATEGY) return "SPY";
  if (name === COMPOSER_NAME) return "Composé";
  return name
    .replace("Legacy · Frequency", "Legacy")
    .replace("Boosting tendance", "Tendance")
    .replace("Boosting", "Boost")
    .replace(" · ", " ");
}

function recentWindow() {
  return state.data.recent_valuation.windows[state.recentWindow];
}

function studioDisplayMetrics() {
  return state.analysisMode === "recent"
    ? state.data.recent_valuation.metric_fields
    : STUDIO_DISPLAY_METRICS;
}

function renderStudioContext() {
  if (state.analysisMode !== "recent") {
    document.getElementById("window-label").textContent = `${monthLabel(state.start)} → ${monthLabel(state.end)}`;
    return;
  }
  const window = recentWindow();
  const recent = state.data.recent_valuation;
  document.getElementById("window-label").textContent = `${dateLabel(window.start_date)} → ${dateLabel(window.end_date)}`;
  document.getElementById("recent-contract").innerHTML = `
    <strong>Panier figé de ${escapeHtml(monthLabel(recent.holding_month))}, pas un nouveau signal.</strong>
    Maintien hypothétique, sans rééquilibrage. Suivi brut en USD, distinct du backtest net de frais.
    <a href="#methodologies">Méthode et limites</a>
  `;
}

function renderStudioKpis() {
  const series = studioDisplaySeries();
  const fields = studioDisplayMetrics();
  const header = fields.map(field => `<span>${escapeHtml(METRICS[field][0])}</span>`).join("");
  const rows = series.map(name => {
    const color = name === COMPOSER_NAME ? COMPOSER_COLOR : strategyMeta(name).color;
    const cells = fields.map(field => {
      const [, type] = METRICS[field];
      const value = studioMetricValueAtWindow(name, field);
      const status = studioComparisonState(field, name, value);
      const benchmark = studioMetricValueAtWindow(BENCHMARK_STRATEGY, field);
      const comparison = name === BENCHMARK_STRATEGY
        ? "Référence"
        : comparisonDeltaText(field, value, benchmark);
      return `<span class="scoreboard-value comparison-${status}" title="${escapeHtml(comparison)}"><strong>${format(value, type)}</strong><small>${escapeHtml(comparison)}</small></span>`;
    }).join("");
    return `<div class="scoreboard-row"><span class="scoreboard-strategy" title="${escapeHtml(name)}"><i style="background:${color}"></i><strong>${escapeHtml(compactStrategyName(name))}</strong></span>${cells}</div>`;
  }).join("");
  document.getElementById("studio-kpis").innerHTML = `<div class="scoreboard"><div class="scoreboard-row scoreboard-head"><span>Stratégie</span>${header}</div>${rows}</div>`;
}

function renderStudioOutcome() {
  const field = state.analysisMode === "recent" ? "total_return" : "cagr";
  const fields = studioDisplayMetrics();
  const candidates = studioDisplaySeries().filter(name => name !== BENCHMARK_STRATEGY);
  const leader = candidates.reduce((best, name) => {
    const value = studioMetricValueAtWindow(name, field);
    return !best || value > best.value ? {name, value} : best;
  }, null);
  const benchmark = studioMetricValueAtWindow(BENCHMARK_STRATEGY, field);
  const status = leader ? studioComparisonState(field, leader.name, leader.value) : "neutral";
  const wins = leader ? fields.filter(field => {
    const value = studioMetricValueAtWindow(leader.name, field);
    return studioComparisonState(field, leader.name, value) === "beats";
  }).length : 0;
  const element = document.getElementById("studio-outcome");
  element.className = `studio-outcome comparison-${status}`;
  element.innerHTML = leader ? `
    <span>${state.analysisMode === "composer" ? "Votre composition" : state.analysisMode === "recent" ? "Performance de la fenêtre" : "Meilleur CAGR"}</span>
    <strong>${escapeHtml(compactStrategyName(leader.name))} : ${comparisonDeltaText(field, leader.value, benchmark)}</strong>
    <small>Surpasse SPY sur ${wins} des ${fields.length} indicateurs clés.</small>
  ` : "";
}

function renderStudio() {
  if (!state.data) return;
  updateCurveSummary();
  renderStudioContext();
  if (state.analysisMode === "frontier") {
    renderFrontierStudio();
    return;
  }
  renderStudioKpis();
  renderStudioOutcome();
  drawStudioChart();
  renderMetricTable();
  if (state.analysisMode !== "recent") renderMatrices();
  if (state.analysisMode === "composer") renderComposerCorrelation();
}

function resizeStudioCharts() {
  ["studio-chart", "frontier-volatility-chart", "frontier-drawdown-chart"].forEach(id => AlphaRankStrategyCharts.resize(document.getElementById(id)));
}
"""
