from __future__ import annotations

PERFORMANCE_REPORT_FRONTIER_SCRIPT = r"""
const FRONTIER_PROFILE_LABELS = {
  minimum_volatility: "Volatilité minimale",
  maximum_sharpe: "Meilleur Sharpe",
  maximum_calmar: "Meilleur Calmar",
};
const FRONTIER_PROFILE_COLORS = {
  minimum_volatility: "#0369a1",
  maximum_sharpe: "#265511",
  maximum_calmar: "#9b8816",
};

function frontierWindow(key=windowKey()) {
  return state.data.strategy_frontiers.windows[key];
}

function frontierMetricIndex(field) {
  return state.data.strategy_frontiers.metric_fields.indexOf(field);
}

function frontierCandidate(index, window=frontierWindow()) {
  if (!window || !Number.isInteger(index)) return null;
  return {
    index,
    weights: state.data.strategy_frontiers.weights[index],
    metrics: window.candidate_metrics[index],
  };
}

function frontierWeightText(weights) {
  return state.data.strategy_frontiers.strategy_order.map((strategy, index) =>
    `${strategy.replace("Boosting tendance", "Tendance").replace(" · Frequency", "").replace(" · Top 5", "")}: ${format(weights[index], "pct")}`
  ).join(" · ");
}

function renderFrontierRecommendations() {
  const window = frontierWindow();
  const container = document.getElementById("frontier-recommendations");
  if (!window) {
    container.innerHTML = '<p class="frontier-empty">Aucune frontière pré-calculée pour cette fenêtre.</p>';
    return;
  }
  const cagr = frontierMetricIndex("cagr");
  const volatility = frontierMetricIndex("annualized_volatility");
  const drawdown = frontierMetricIndex("max_drawdown");
  const sharpe = frontierMetricIndex("sharpe");
  const calmar = frontierMetricIndex("calmar");
  container.innerHTML = Object.entries(FRONTIER_PROFILE_LABELS).map(([profile, label]) => {
    const candidate = frontierCandidate(window.recommendations[profile], window);
    return `<article class="frontier-recommendation" style="--profile-color:${FRONTIER_PROFILE_COLORS[profile]}">
      <header><span>${escapeHtml(label)}</span><small>Post-hoc · ${window.months} mois</small></header>
      <strong>${escapeHtml(frontierWeightText(candidate.weights))}</strong>
      <dl>
        <div><dt>CAGR</dt><dd>${format(candidate.metrics[cagr], "pct")}</dd></div>
        <div><dt>Volatilité</dt><dd>${format(candidate.metrics[volatility], "pct")}</dd></div>
        <div><dt>Max drawdown</dt><dd>${format(candidate.metrics[drawdown], "pct")}</dd></div>
        <div><dt>Sharpe</dt><dd>${format(candidate.metrics[sharpe])}</dd></div>
        <div><dt>Calmar</dt><dd>${format(candidate.metrics[calmar])}</dd></div>
      </dl>
    </article>`;
  }).join("");
}

function renderFrontierAnchorTable() {
  const payload = state.data.strategy_frontiers;
  const cagr = frontierMetricIndex("cagr");
  const volatility = frontierMetricIndex("annualized_volatility");
  const drawdown = frontierMetricIndex("max_drawdown");
  const rows = payload.anchor_windows.flatMap(anchor => {
    const window = payload.windows[anchor.window_key];
    return Object.entries(FRONTIER_PROFILE_LABELS).map(([profile, label]) => {
      const candidate = frontierCandidate(window.recommendations[profile], window);
      const requested = anchor.requested_start_year === 2010
        ? `2010 demandé · ${monthLabel(anchor.effective_start)} effectif`
        : `${anchor.requested_start_year} → ${anchor.end.slice(0, 4)}`;
      return `<tr>
        <td>${escapeHtml(requested)}</td>
        <td><span class="profile-dot" style="background:${FRONTIER_PROFILE_COLORS[profile]}"></span>${escapeHtml(label)}</td>
        <td class="frontier-composition-cell">${escapeHtml(frontierWeightText(candidate.weights))}</td>
        <td>${format(candidate.metrics[cagr], "pct")}</td>
        <td>${format(candidate.metrics[volatility], "pct")}</td>
        <td>${format(candidate.metrics[drawdown], "pct")}</td>
      </tr>`;
    });
  });
  document.getElementById("frontier-anchor-body").innerHTML = rows.join("");
}

function renderFrontierStudio() {
  renderFrontierRecommendations();
  drawFrontierChart("volatility");
  drawFrontierChart("drawdown");
  renderFrontierAnchorTable();
}

function drawFrontierChart(kind) {
  const window = frontierWindow();
  if (!window) return;
  const isDrawdown = kind === "drawdown";
  const riskField = isDrawdown ? "max_drawdown" : "annualized_volatility";
  const riskIndex = frontierMetricIndex(riskField);
  const cagrIndex = frontierMetricIndex("cagr");
  const frontierIndices = isDrawdown
    ? window.drawdown_frontier_indices
    : window.volatility_frontier_indices;
  const points = window.candidate_metrics.map((metrics, index) => ({
    index,
    x: isDrawdown ? Math.abs(metrics[riskIndex]) : metrics[riskIndex],
    y: metrics[cagrIndex],
    weights: state.data.strategy_frontiers.weights[index],
    kind: "candidate",
  })).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
  const references = Object.entries(window.reference_metrics).map(([name, metrics]) => ({
    name,
    x: isDrawdown ? Math.abs(metrics[riskIndex]) : metrics[riskIndex],
    y: metrics[cagrIndex],
    kind: "reference",
    color: strategyMeta(name).color,
  })).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
  const recommendations = Object.entries(window.recommendations).map(([profile, index]) => ({
    ...points.find(point => point.index === index),
    profile,
    name: FRONTIER_PROFILE_LABELS[profile],
    color: FRONTIER_PROFILE_COLORS[profile],
    kind: "recommendation",
  }));
  const canvas = document.getElementById(`frontier-${kind}-chart`);
  drawFrontierScatter(canvas, {
    points,
    references,
    recommendations,
    frontier: frontierIndices.map(index => points.find(point => point.index === index)).filter(Boolean),
    xLabel: isDrawdown ? "Max drawdown absolu" : "Volatilité annualisée",
  });
}

function drawFrontierScatter(element, model) {
  const describe = point => ({
    ...point,
    description: point.weights ? frontierWeightText(point.weights) : "",
  });
  element._frontierModel = model;
  return AlphaRankStrategyCharts.renderFrontier(element, {
    ...model,
    points: model.points.map(describe),
    recommendations: model.recommendations.map(describe),
  });
}
"""
