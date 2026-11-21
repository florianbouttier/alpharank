from __future__ import annotations

PERFORMANCE_REPORT_CHART_SCRIPT = r"""
function chartMonthEnd(month) {
  const year = Number(month.slice(0, 4)), number = Number(month.slice(5, 7));
  return new Date(Date.UTC(year, number, 0)).toISOString().slice(0, 10);
}

function chartInitialPoint(value) {
  const year = Number(state.start.slice(0, 4)), number = Number(state.start.slice(5, 7));
  return {
    date: new Date(Date.UTC(year, number - 1, 0)).toISOString().slice(0, 10),
    value,
    label: "Point initial · avant le premier rendement",
  };
}

function monthlyChartProjection(rows, view) {
  const benchmark = new Map(periodMonthly(BENCHMARK_STRATEGY).map(row => [row.holding_month, row.net_return]));
  const points = [chartInitialPoint(view === "drawdown" ? 0 : 1)];
  let wealth = 1, reference = 1, peak = 1;
  rows.forEach(row => {
    wealth *= 1 + row.value;
    if (view === "relative") {
      if (!benchmark.has(row.date)) throw new Error("Mois SPY absent de la comparaison : " + row.date);
      reference *= 1 + benchmark.get(row.date);
    }
    peak = Math.max(peak, wealth);
    const value = view === "relative" ? wealth / reference : view === "drawdown" ? wealth / peak - 1 : wealth;
    points.push({date: chartMonthEnd(row.date), value, label: monthLabel(row.date)});
  });
  return points;
}

function relativeWealthSeries(strategy) {
  return monthlyChartProjection(periodMonthly(strategy).map(row => ({date: row.holding_month, value: row.net_return})), "relative");
}

function studioChartSeries() {
  if (state.analysisMode === "recent") {
    const field = state.chartView === "drawdown" ? "drawdown" : state.chartView === "relative" ? "relative_wealth" : "wealth";
    const series = state.curves.filter(name => state.chartView !== "relative" || name !== BENCHMARK_STRATEGY)
      .map(name => ({
        name: state.chartView === "relative" ? name + " ÷ SPY" : name,
        color: strategyMeta(name).color,
        reference: name === BENCHMARK_STRATEGY,
        values: recentWindow().series[name].map(point => ({date: point.date, value: point[field]})),
      }));
    return state.chartView === "relative" ? [...series, paritySeries()] : series;
  }
  if (state.analysisMode === "composer") {
    const values = state.chartView === "drawdown" ? composerDrawdownSeries() : state.chartView === "relative" ? composerRelativeWealthSeries() : composerWealthSeries();
    return [
      {name: COMPOSER_NAME, color: COMPOSER_COLOR, values},
      state.chartView === "relative" ? paritySeries() : {
        name: BENCHMARK_STRATEGY, color: strategyMeta(BENCHMARK_STRATEGY).color, reference: true,
        values: state.chartView === "drawdown" ? drawdownSeries(BENCHMARK_STRATEGY) : wealthSeries(BENCHMARK_STRATEGY),
      },
    ];
  }
  if (state.chartView === "relative") {
    return [
      ...state.curves.filter(name => name !== BENCHMARK_STRATEGY).map(name => ({
        name: name + " ÷ SPY", color: strategyMeta(name).color, values: relativeWealthSeries(name),
      })),
      paritySeries(),
    ];
  }
  const valuesFor = state.chartView === "drawdown" ? drawdownSeries : wealthSeries;
  return state.curves.map(name => ({
    name, color: strategyMeta(name).color, reference: name === BENCHMARK_STRATEGY, values: valuesFor(name),
  }));
}

function paritySeries() {
  const dates = state.analysisMode === "recent"
    ? recentWindow().series[BENCHMARK_STRATEGY].map(point => point.date)
    : [chartInitialPoint(1).date, ...periodMonthly(BENCHMARK_STRATEGY).map(row => chartMonthEnd(row.holding_month))];
  return {
    name: "SPY · parité 1", color: strategyMeta(BENCHMARK_STRATEGY).color, reference: true,
    values: dates.map(date => ({date, value: 1})),
  };
}

function drawStudioChart() {
  if (!state.data) return;
  const drawdown = state.chartView === "drawdown", relative = state.chartView === "relative";
  document.getElementById("studio-chart-title").textContent = drawdown ? "Drawdown" : relative ? "Richesse relative au SPY" : "Capital · base 1";
  document.getElementById("studio-chart-subtitle").textContent = drawdown
    ? "Départ à 0 ; écart à chaque plus-haut de la fenêtre."
    : relative ? "Départ à 1. Au-dessus : la sélection surpasse SPY ; en dessous : elle est en retard."
    : "1 au départ, avant le premier rendement. Aucun mois n'est retiré du calcul.";
  const series = studioChartSeries();
  drawLineChart(document.getElementById("studio-chart"), series);
  renderLegend("studio-legend", series);
}

function renderLegend(id, series) {
  document.getElementById(id).innerHTML = series.map(item =>
    '<span title="' + escapeHtml(item.name) + '"><i style="background:' + item.color + '"></i><strong>' +
    escapeHtml(compactStrategyName(item.name)) + '</strong></span>'
  ).join("");
}

function drawLineChart(element, series) {
  element._chartModel = {series, length: Math.max(0, ...series.map(item => item.values.length))};
  const key = state.analysisMode === "recent" ? state.recentWindow : windowKey();
  document.getElementById("chart-hint-caption").textContent = "Glisser pour zoomer · double-clic pour réinitialiser";
  return AlphaRankStrategyCharts.renderTimeSeries(element, {
    series, view: state.chartView,
    periodKey: state.analysisMode + "|" + key + "|" + state.chartView,
    height: window.innerWidth < 760 ? 320 : 440,
    onViewportChange: zoomed => {
      document.getElementById("chart-hint-caption").textContent = zoomed
        ? "Zoom d'affichage · les KPI restent sur la période sélectionnée"
        : "Glisser pour zoomer · double-clic pour réinitialiser";
    },
  });
}
"""
