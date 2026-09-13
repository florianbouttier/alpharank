from __future__ import annotations

PERFORMANCE_REPORT_CHART_SCRIPT = r"""
function relativeWealthSeries(strategy) {
  const spyByMonth = new Map(
    periodMonthly(BENCHMARK_STRATEGY).map(row => [row.holding_month, row.net_return]),
  );
  let strategyWealth = 1;
  let spyWealth = 1;
  return periodMonthly(strategy).map(row => {
    strategyWealth *= 1 + row.net_return;
    spyWealth *= 1 + spyByMonth.get(row.holding_month);
    return {date: row.holding_month, value: strategyWealth / spyWealth};
  });
}

function studioChartSeries() {
  if (state.analysisMode === "composer") {
    if (state.chartView === "wealth") return [
      {name: COMPOSER_NAME, color: COMPOSER_COLOR, values: composerWealthSeries()},
      {name: BENCHMARK_STRATEGY, color: strategyMeta(BENCHMARK_STRATEGY).color, values: wealthSeries(BENCHMARK_STRATEGY)},
    ];
    if (state.chartView === "drawdown") return [
      {name: COMPOSER_NAME, color: COMPOSER_COLOR, values: composerDrawdownSeries()},
      {name: BENCHMARK_STRATEGY, color: strategyMeta(BENCHMARK_STRATEGY).color, values: drawdownSeries(BENCHMARK_STRATEGY)},
    ];
    return [
      {name: `${COMPOSER_NAME} ÷ SPY`, color: COMPOSER_COLOR, values: composerRelativeWealthSeries()},
      paritySeries(),
    ];
  }
  if (state.chartView === "relative") {
    return [
      ...state.curves.filter(name => name !== BENCHMARK_STRATEGY).map(name => ({
        name: `${name} ÷ SPY`,
        color: strategyMeta(name).color,
        values: relativeWealthSeries(name),
      })),
      paritySeries(),
    ];
  }
  const valuesFor = state.chartView === "drawdown" ? drawdownSeries : wealthSeries;
  return state.curves.map(name => ({
    name,
    color: strategyMeta(name).color,
    values: valuesFor(name),
  }));
}

function paritySeries() {
  return {
    name: "Parité SPY = 1",
    color: strategyMeta(BENCHMARK_STRATEGY).color,
    values: periodMonthly(BENCHMARK_STRATEGY).map(row => ({date: row.holding_month, value: 1})),
  };
}

function drawStudioChart() {
  if (!state.data) return;
  const isDrawdown = state.chartView === "drawdown";
  const isRelative = state.chartView === "relative";
  const title = isDrawdown ? "Drawdown" : isRelative ? "Richesse relative au SPY" : "Croissance composée";
  const subtitle = isDrawdown
    ? "Écart à chaque plus-haut de richesse."
    : isRelative
      ? "Au-dessus de 1 : la sélection a davantage composé que SPY ; en dessous, elle est en retard."
      : "Courbes rebasées à 1 au début de la fenêtre.";
  document.getElementById("studio-chart-title").textContent = title;
  document.getElementById("studio-chart-subtitle").textContent = subtitle;
  const formatter = isDrawdown
    ? value => `${(100 * value).toFixed(0)}%`
    : value => `${value.toFixed(2)}×`;
  const series = studioChartSeries();
  drawLineChart(
    document.getElementById("studio-chart"),
    series,
    formatter,
    {referenceValue: isDrawdown ? 0 : isRelative ? 1 : null, splitTone: isRelative},
  );
  renderLegend("studio-legend", series);
}

function renderLegend(id, series) {
  document.getElementById(id).innerHTML = series.map(item => `
    <span><i style="background:${item.color}"></i>${escapeHtml(item.name)}</span>
  `).join("");
}

function drawLineChart(canvas, series, tickFormat, options={}) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const height = 360;
  canvas.width = Math.max(600, rect.width * ratio);
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const pad = {l: 58, r: 18, t: 18, b: 32};
  const values = series.flatMap(item => item.values.map(point => point.value));
  if (Number.isFinite(options.referenceValue)) values.push(options.referenceValue);
  if (!values.length) return;
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 0.1;
    max += 0.1;
  }
  const chartHeight = height - pad.t - pad.b;
  const referenceY = Number.isFinite(options.referenceValue)
    ? pad.t + chartHeight * (max - options.referenceValue) / (max - min)
    : null;
  ctx.clearRect(0, 0, width, height);
  if (options.splitTone && referenceY !== null) {
    ctx.fillStyle = "rgba(38,85,17,.055)";
    ctx.fillRect(pad.l, pad.t, width - pad.l - pad.r, Math.max(0, referenceY - pad.t));
    ctx.fillStyle = "rgba(128,35,49,.045)";
    ctx.fillRect(pad.l, referenceY, width - pad.l - pad.r, Math.max(0, height - pad.b - referenceY));
  }
  ctx.font = "11px IBM Plex Mono, monospace";
  ctx.fillStyle = "#617087";
  ctx.strokeStyle = "#e1e7ee";
  ctx.lineWidth = 1;
  for (let index = 0; index < 5; index += 1) {
    const y = pad.t + chartHeight * index / 4;
    const value = max - (max - min) * index / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(width - pad.r, y);
    ctx.stroke();
    ctx.fillText(tickFormat(value), 4, y + 4);
  }
  if (referenceY !== null) {
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = strategyMeta(BENCHMARK_STRATEGY).color;
    ctx.beginPath();
    ctx.moveTo(pad.l, referenceY);
    ctx.lineTo(width - pad.r, referenceY);
    ctx.stroke();
    ctx.restore();
  }
  const length = Math.max(...series.map(item => item.values.length));
  series.forEach(item => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.name.includes("SPY") || item.name.includes("Parité") ? 2.4 : 2;
    ctx.beginPath();
    item.values.forEach((point, index) => {
      const x = pad.l + (width - pad.l - pad.r) * (length === 1 ? 0 : index / (length - 1));
      const y = pad.t + chartHeight * (max - point.value) / (max - min);
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#617087";
  ctx.textAlign = "left";
  ctx.fillText(monthLabel(state.start), pad.l, height - 8);
  ctx.textAlign = "right";
  ctx.fillText(monthLabel(state.end), width - pad.r, height - 8);
  ctx.textAlign = "left";
  canvas._chartModel = {series, tickFormat, pad, width, height, length};
  canvas.onmousemove = showStudioChartTooltip;
  canvas.onmouseleave = hideStudioChartTooltip;
  hideStudioChartTooltip();
}

function showStudioChartTooltip(event) {
  const canvas = event.currentTarget;
  const model = canvas._chartModel;
  if (!model?.length) return;
  const chartWidth = model.width - model.pad.l - model.pad.r;
  const rawIndex = (event.offsetX - model.pad.l) / chartWidth * (model.length - 1);
  const index = Math.max(0, Math.min(model.length - 1, Math.round(rawIndex)));
  const date = model.series.find(item => item.values[index])?.values[index]?.date;
  if (!date) return;
  const tooltip = document.getElementById("studio-chart-tooltip");
  const rows = model.series.map(item => {
    const point = item.values[index];
    return point ? `<div><span><i style="background:${item.color}"></i>${escapeHtml(item.name)}</span><strong>${model.tickFormat(point.value)}</strong></div>` : "";
  }).join("");
  tooltip.innerHTML = `<header>${monthLabel(date)}</header>${rows}`;
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(8, Math.min(canvas.clientWidth - 250, event.offsetX + 14))}px`;
  tooltip.style.top = `${Math.max(8, event.offsetY - 36)}px`;
}

function hideStudioChartTooltip() {
  document.getElementById("studio-chart-tooltip").hidden = true;
}
"""
