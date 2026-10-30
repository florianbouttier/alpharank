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

function drawFrontierScatter(canvas, model) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const height = 330;
  canvas.width = Math.max(520, rect.width * ratio);
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const pad = {l: 58, r: 18, t: 18, b: 44};
  const all = [...model.points, ...model.references];
  let xMin = Math.min(...all.map(point => point.x));
  let xMax = Math.max(...all.map(point => point.x));
  let yMin = Math.min(...all.map(point => point.y));
  let yMax = Math.max(...all.map(point => point.y));
  const xMargin = Math.max((xMax - xMin) * 0.08, 0.005);
  const yMargin = Math.max((yMax - yMin) * 0.10, 0.005);
  xMin = Math.max(0, xMin - xMargin); xMax += xMargin;
  yMin -= yMargin; yMax += yMargin;
  const xScale = value => pad.l + (width - pad.l - pad.r) * (value - xMin) / (xMax - xMin);
  const yScale = value => pad.t + (height - pad.t - pad.b) * (yMax - value) / (yMax - yMin);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px IBM Plex Mono, monospace";
  ctx.fillStyle = cssColor("--muted", "#617087");
  ctx.strokeStyle = cssColor("--grid", "#e1e7ee");
  for (let index = 0; index < 5; index += 1) {
    const y = pad.t + (height - pad.t - pad.b) * index / 4;
    const yValue = yMax - (yMax - yMin) * index / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y); ctx.stroke();
    ctx.fillText(`${(100 * yValue).toFixed(0)}%`, 4, y + 4);
    const x = pad.l + (width - pad.l - pad.r) * index / 4;
    const xValue = xMin + (xMax - xMin) * index / 4;
    ctx.textAlign = index === 4 ? "right" : index === 0 ? "left" : "center";
    ctx.fillText(`${(100 * xValue).toFixed(0)}%`, x, height - 23);
  }
  ctx.textAlign = "center";
  ctx.fillText(model.xLabel, pad.l + (width - pad.l - pad.r) / 2, height - 5);
  model.points.forEach(point => {
    ctx.beginPath(); ctx.arc(xScale(point.x), yScale(point.y), 2.2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(97,112,135,.25)"; ctx.fill();
  });
  ctx.beginPath();
  model.frontier.forEach((point, index) => {
    const x = xScale(point.x), y = yScale(point.y);
    index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = cssColor("--navy", "#111d55"); ctx.lineWidth = 2.2; ctx.stroke();
  model.references.forEach(point => {
    const x = xScale(point.x), y = yScale(point.y);
    ctx.fillStyle = point.color; ctx.fillRect(x - 4, y - 4, 8, 8);
  });
  model.recommendations.forEach(point => {
    const x = xScale(point.x), y = yScale(point.y);
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = point.color; ctx.fill(); ctx.strokeStyle = cssColor("--panel", "#fff"); ctx.lineWidth = 2; ctx.stroke();
  });
  ctx.textAlign = "left";
  canvas._frontierModel = {...model, xScale, yScale};
  canvas.onmousemove = showFrontierTooltip;
  canvas.onmouseleave = hideFrontierTooltip;
}

function showFrontierTooltip(event) {
  const canvas = event.currentTarget;
  const model = canvas._frontierModel;
  const candidates = [...model.recommendations, ...model.references, ...model.points];
  const nearest = candidates.reduce((best, point) => {
    const distance = Math.hypot(event.offsetX - model.xScale(point.x), event.offsetY - model.yScale(point.y));
    return !best || distance < best.distance ? {point, distance} : best;
  }, null);
  const tooltip = canvas.parentElement.querySelector(".frontier-tooltip");
  if (!nearest || nearest.distance > 14) { tooltip.hidden = true; return; }
  const point = nearest.point;
  const title = point.name || `Composition ${point.index + 1}`;
  const composition = point.weights ? `<div>${escapeHtml(frontierWeightText(point.weights))}</div>` : "";
  tooltip.innerHTML = `<header>${escapeHtml(title)}</header>${composition}<div><span>Risque</span><strong>${format(point.x, "pct")}</strong></div><div><span>CAGR</span><strong>${format(point.y, "pct")}</strong></div>`;
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(8, Math.min(canvas.clientWidth - 250, event.offsetX + 14))}px`;
  tooltip.style.top = `${Math.max(8, event.offsetY - 36)}px`;
}

function hideFrontierTooltip(event) {
  event.currentTarget.parentElement.querySelector(".frontier-tooltip").hidden = true;
}
"""
