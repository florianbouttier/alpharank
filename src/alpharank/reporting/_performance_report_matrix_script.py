from __future__ import annotations

PERFORMANCE_REPORT_MATRIX_SCRIPT = r"""
function renderMetricTable() {
  const fields = state.analysisMode === "composer"
    ? state.data.portfolio_composer.metric_fields
    : state.data.metric_fields;
  const series = studioDisplaySeries();
  const table = document.querySelector(".metric-table");
  table.style.minWidth = `${Math.max(760, 460 + series.length * 148)}px`;
  document.getElementById("metric-head").innerHTML = `<th>KPI</th>${series.map(name => {
    const color = name === COMPOSER_NAME ? COMPOSER_COLOR : strategyMeta(name).color;
    return `<th class="${name === BENCHMARK_STRATEGY ? "benchmark-head" : ""}"><i style="background:${color}"></i>${escapeHtml(name)}</th>`;
  }).join("")}<th>Définition</th>`;
  document.getElementById("metric-body").innerHTML = fields.map(field => {
    const [label, type, definition] = METRICS[field] || [field, "num", ""];
    const benchmark = studioMetricValueAtWindow(BENCHMARK_STRATEGY, field);
    const values = series.map(name => {
      const value = studioMetricValueAtWindow(name, field);
      const status = studioComparisonState(field, name, value);
      const delta = name === BENCHMARK_STRATEGY
        ? "Référence"
        : comparisonDeltaText(field, value, benchmark);
      return `<td class="metric-value comparison-${status}"><strong>${format(value, type)}</strong><small>${escapeHtml(delta)}</small></td>`;
    }).join("");
    return `<tr><td>${escapeHtml(label)}</td>${values}<td class="metric-definition">${escapeHtml(definition)}</td></tr>`;
  }).join("");
}

function viridis(value) {
  const x = Math.max(0, Math.min(1, value));
  const scaled = x * (VIRIDIS.length - 1);
  const index = Math.min(VIRIDIS.length - 2, Math.floor(scaled));
  const interpolation = scaled - index;
  const rgb = VIRIDIS[index].map((valueAtIndex, channel) => Math.round(
    valueAtIndex + (VIRIDIS[index + 1][channel] - valueAtIndex) * interpolation,
  ));
  return `rgb(${rgb.join(",")})`;
}

function matrixYears() {
  const first = Number(state.start.slice(0, 4));
  const last = Number(state.end.slice(0, 4));
  return Array.from({length: last - first + 1}, (_, index) => first + index);
}

function yearBoundary(year, side) {
  const values = side === "start"
    ? state.data.calendar.available_start_months
    : state.data.calendar.available_end_months;
  return values.find(value => Number(value.slice(0, 4)) === year);
}

function matrixWindows(mode) {
  const endYear = Number(state.end.slice(0, 4));
  return matrixYears().map(year => {
    const start = yearBoundary(year, "start");
    const end = mode === "cumulative" || year === endYear
      ? state.end
      : yearBoundary(year, "end");
    if (!start || !end || start > end) return null;
    const key = `${start}|${end}`;
    return state.data.metric_windows[key] ? {year, start, end, key} : null;
  }).filter(Boolean);
}

function renderHeatmap(id, windows, field) {
  const series = studioDisplaySeries();
  const values = windows.flatMap(window => series.map(name => (
    studioMetricValueAtWindow(name, field, window.key)
  )));
  const colorValues = values.map(value => field === "max_drawdown" ? Math.abs(value) : value)
    .filter(Number.isFinite);
  const min = Math.min(...colorValues);
  const max = Math.max(...colorValues);
  let html = `<div class="heatmap-head"></div>${windows.map(window => `<div class="heatmap-head">${window.year}</div>`).join("")}`;
  series.forEach(name => {
    html += `<div class="heatmap-label">${escapeHtml(name)}</div>`;
    windows.forEach(window => {
      const shown = studioMetricValueAtWindow(name, field, window.key);
      const raw = field === "max_drawdown" ? Math.abs(shown) : shown;
      const level = Number.isFinite(raw) && max > min ? (raw - min) / (max - min) : 0.5;
      const text = level > 0.62 ? "#172033" : "#fff";
      const title = `${name} · ${monthLabel(window.start)} → ${monthLabel(window.end)}`;
      html += `<div class="heatmap-cell" style="background:${viridis(level)};color:${text}" title="${escapeHtml(title)}">${format(shown, "pct")}</div>`;
    });
  });
  const matrix = document.getElementById(id);
  matrix.style.gridTemplateColumns = `220px repeat(${windows.length}, minmax(72px, 1fr))`;
  matrix.innerHTML = html;
}

function renderMatrices() {
  const cumulative = matrixWindows("cumulative");
  const incremental = matrixWindows("incremental");
  const annualField = state.matrixMetric === "cagr" ? "total_return" : state.matrixMetric;
  renderHeatmap("cumulative-heatmap", cumulative, state.matrixMetric);
  renderHeatmap("incremental-heatmap", incremental, annualField);
  document.getElementById("cumulative-matrix-window").textContent = `${monthLabel(state.start)} → ${monthLabel(state.end)} · chaque colonne repart du début de son année.`;
  document.getElementById("cumulative-matrix-caption").textContent = state.matrixMetric === "cagr" ? "CAGR calculé de chaque année de départ jusqu'à la fin sélectionnée." : state.matrixMetric === "annualized_volatility" ? "Volatilité annualisée de chaque départ jusqu'à la fin sélectionnée." : "Profondeur du drawdown de chaque départ jusqu'à la fin sélectionnée.";
  document.getElementById("incremental-matrix-caption").textContent = state.matrixMetric === "cagr" ? "Rendement composé de l'année isolée ; les années de bord peuvent être partielles." : state.matrixMetric === "annualized_volatility" ? "Volatilité annualisée calculée uniquement avec les mois de l'année." : "Drawdown calculé uniquement à l'intérieur de chaque année.";
}
"""
