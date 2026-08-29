PERFORMANCE_REPORT_SCRIPT = r"""
const METRICS = {
  total_return: ["Rendement total", "pct", "Produit des rendements mensuels nets."],
  cagr: ["CAGR", "pct", "Taux de croissance annuel composé."],
  annualized_volatility: ["Volatilité annualisée", "pct", "Écart-type mensuel échantillon × √12."],
  sharpe: ["Sharpe", "num", "(CAGR − taux sans risque 2 %) / volatilité."],
  max_drawdown: ["Max drawdown", "pct", "Perte maximale depuis un plus-haut de la courbe composée."],
  positive_month_rate: ["Mois positifs", "pct", "Part des rendements mensuels strictement positifs."],
  sortino: ["Sortino", "num", "CAGR excédentaire rapporté au risque baissier."],
  calmar: ["Calmar", "num", "CAGR rapporté à la valeur absolue du max drawdown."],
  annualized_excess_return: ["Excès annualisé", "pct", "CAGR stratégie moins CAGR SPY."],
  tracking_error: ["Tracking error", "pct", "Volatilité annualisée du rendement actif."],
  information_ratio: ["Information ratio", "num", "Rendement actif annualisé rapporté au tracking error."],
  beta: ["Bêta", "num", "Covariance avec SPY divisée par variance SPY."],
  alpha: ["Alpha annualisé", "pct", "Alpha CAPM arithmétique annualisé, taux sans risque 2 %."],
  correlation: ["Corrélation SPY", "num", "Corrélation mensuelle avec SPY."],
  benchmark_hit_rate: ["Mois > SPY", "pct", "Part des mois où la stratégie bat SPY."],
  var_95: ["VaR 95 % mensuelle", "pct", "Quantile mensuel à 5 %."],
  cvar_95: ["CVaR 95 % mensuelle", "pct", "Moyenne des rendements sous la VaR 95 %."],
  omega: ["Omega", "num", "Somme des gains mensuels divisée par la somme absolue des pertes."],
  up_capture: ["Capture haussière", "pct", "Rendement moyen quand SPY monte, relatif à SPY."],
  down_capture: ["Capture baissière", "pct", "Rendement moyen quand SPY baisse, relatif à SPY."],
  skewness: ["Asymétrie", "num", "Moment centré standardisé d’ordre 3."],
  excess_kurtosis: ["Kurtosis excédentaire", "num", "Moment standardisé d’ordre 4 moins 3."],
  average_monthly_turnover: ["Turnover mensuel moyen", "pct", "Turnover moyen facturé par le moteur commun."],
  annualized_turnover: ["Turnover annualisé", "pct", "Turnover mensuel moyen × 12."],
  total_transaction_cost: ["Coûts cumulés", "pct", "Somme des taux de coûts mensuels facturés."],
  annualized_transaction_cost: ["Coût annualisé", "pct", "Coût mensuel moyen × 12."],
  average_positions: ["Positions moyennes", "num", "Nombre moyen de positions mensuelles."],
  minimum_positions: ["Positions minimum", "int", "Plus petit nombre de positions mensuelles."],
  maximum_positions: ["Positions maximum", "int", "Plus grand nombre de positions mensuelles."],
  average_maximum_position_weight: ["Poids max moyen", "pct", "Moyenne mensuelle du poids de la première ligne."],
  maximum_single_name_weight: ["Poids individuel maximal", "pct", "Plus grand poids mensuel observé."],
  average_maximum_sector_weight: ["Poids secteur max moyen", "pct", "Moyenne du secteur le plus concentré chaque mois."],
  maximum_sector_weight: ["Poids secteur maximal", "pct", "Concentration sectorielle mensuelle maximale."],
};
const CORE_METRICS = ["cagr", "total_return", "annualized_volatility", "max_drawdown", "sharpe", "sortino"];
const BENCHMARK_STRATEGY = "SPY · Total return";
const DEFAULT_CURVE_STRATEGIES = ["Legacy · Frequency", "Boosting tendance · Top 5", BENCHMARK_STRATEGY];
const METRIC_DIRECTIONS = {
  total_return: "higher", cagr: "higher", annualized_volatility: "lower",
  sharpe: "higher", max_drawdown: "higher", positive_month_rate: "higher",
  sortino: "higher", calmar: "higher", annualized_excess_return: "higher",
  tracking_error: "lower", information_ratio: "higher", alpha: "higher",
  benchmark_hit_rate: "higher", var_95: "higher", cvar_95: "higher",
  omega: "higher", up_capture: "higher", down_capture: "lower",
  average_monthly_turnover: "lower", annualized_turnover: "lower",
  total_transaction_cost: "lower", annualized_transaction_cost: "lower",
  average_maximum_position_weight: "lower", maximum_single_name_weight: "lower",
  average_maximum_sector_weight: "lower", maximum_sector_weight: "lower",
};
const VIRIDIS = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
const state = { data: null, start: null, end: null, curves: [], matrixMetric: "cagr", page: 0 };

async function decodePayload() {
  const bytes = Uint8Array.from(atob(PAYLOAD_GZIP_BASE64), value => value.charCodeAt(0));
  if (!("DecompressionStream" in window)) throw new Error("Ce navigateur ne sait pas décompresser le payload gzip.");
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

function format(value, type="num") {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (type === "pct") return `${(100 * value).toLocaleString("fr-FR", {minimumFractionDigits: 2, maximumFractionDigits: 2})} %`;
  if (type === "int") return Math.round(value).toLocaleString("fr-FR");
  return value.toLocaleString("fr-FR", {minimumFractionDigits: 2, maximumFractionDigits: 3});
}

function option(value, label=value) { return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[char])); }
function monthLabel(value) { return new Date(`${value.slice(0,7)}-15T12:00:00Z`).toLocaleDateString("fr-FR", {month:"short", year:"numeric"}); }
function strategyMeta(label) { return state.data.strategies.find(item => item.label === label); }
function windowKey() { return `${state.start}|${state.end}`; }
function currentMetricRows() {
  const rows = state.data.metric_windows[windowKey()];
  return rows || [];
}
function metricValue(strategy, field, rows=currentMetricRows()) {
  const strategyIndex = state.data.strategy_order.indexOf(strategy);
  const metricIndex = state.data.metric_fields.indexOf(field);
  return rows[strategyIndex]?.[metricIndex];
}

function comparisonState(field, strategy, value) {
  if (strategy === BENCHMARK_STRATEGY) return "benchmark";
  const direction = METRIC_DIRECTIONS[field];
  const benchmark = metricValue(BENCHMARK_STRATEGY, field);
  if (!direction || !Number.isFinite(value) || !Number.isFinite(benchmark)) return "neutral";
  if (Math.abs(value - benchmark) < 1e-12) return "equal";
  const beats = direction === "higher" ? value > benchmark : value < benchmark;
  return beats ? "beats" : "trails";
}

function comparisonMark(status) {
  return status === "beats" ? "↑ SPY" : status === "trails" ? "↓ SPY" : status === "equal" ? "= SPY" : status === "benchmark" ? "Référence" : "";
}

function initializeControls() {
  const D = state.data;
  state.start = D.calendar.start;
  state.end = D.calendar.end;
  state.curves = DEFAULT_CURVE_STRATEGIES.filter(strategy => D.strategy_order.includes(strategy));
  document.getElementById("start-month").innerHTML = D.calendar.available_start_months.map(value => option(value, value.slice(0,4))).join("");
  document.getElementById("end-month").innerHTML = D.calendar.available_end_months.map(value => option(value, value.slice(0,4))).join("");
  document.getElementById("portfolio-strategy").innerHTML = D.strategy_order.filter(value => value !== "SPY · Total return").map(value => option(value)).join("");
  document.getElementById("portfolio-month").innerHTML = D.calendar.available_months.map(value => option(value, monthLabel(value))).join("");
  document.getElementById("end-month").value = state.end;
  document.getElementById("portfolio-month").value = state.end;
  for (const id of ["start-month","end-month"]) document.getElementById(id).addEventListener("change", updatePeriod);
  for (const id of ["portfolio-strategy","portfolio-month","ticker-search"]) document.getElementById(id).addEventListener(id === "ticker-search" ? "input" : "change", () => { state.page=0; renderHoldings(); });
  renderCurveOptions();
  document.getElementById("select-all-curves").addEventListener("click", () => setCurveStrategies(D.strategy_order));
  document.getElementById("select-reference-curves").addEventListener("click", () => setCurveStrategies(["Legacy · Frequency", BENCHMARK_STRATEGY]));
  document.getElementById("reset-window").addEventListener("click", resetWindow);
  document.getElementById("export-holdings").addEventListener("click", exportHoldings);
  document.getElementById("page-prev").addEventListener("click", () => { state.page=Math.max(0,state.page-1); renderHoldings(); });
  document.getElementById("page-next").addEventListener("click", () => { state.page+=1; renderHoldings(); });
  document.querySelectorAll("[data-matrix-metric]").forEach(button => button.addEventListener("click", () => {
    state.matrixMetric = button.dataset.matrixMetric;
    document.querySelectorAll("[data-matrix-metric]").forEach(item => item.classList.toggle("is-active", item === button));
    renderMatrices();
  }));
}

function renderCurveOptions() {
  const container = document.getElementById("curve-options");
  container.innerHTML = state.data.strategies.map((item, index) => `
    <label class="curve-option">
      <input type="checkbox" data-curve-index="${index}" ${state.curves.includes(item.label) ? "checked" : ""}>
      <i style="background:${item.color}"></i><span>${escapeHtml(item.label)}</span>
    </label>`).join("");
  container.querySelectorAll("input").forEach(input => input.addEventListener("change", () => {
    const label = state.data.strategies[Number(input.dataset.curveIndex)].label;
    const next = input.checked ? [...state.curves, label] : state.curves.filter(value => value !== label);
    if (!next.length) { input.checked = true; return; }
    setCurveStrategies(next);
  }));
  updateCurveSummary();
}

function setCurveStrategies(strategies) {
  const selected = new Set(strategies);
  state.curves = state.data.strategy_order.filter(strategy => selected.has(strategy));
  document.querySelectorAll("[data-curve-index]").forEach(input => {
    input.checked = state.curves.includes(state.data.strategies[Number(input.dataset.curveIndex)].label);
  });
  updateCurveSummary();
  drawPerformance();
  drawDrawdown();
}

function updateCurveSummary() {
  document.getElementById("curve-select-label").textContent = `${state.curves.length} / ${state.data.strategy_order.length} stratégies`;
}

function updatePeriod() {
  const start = document.getElementById("start-month").value;
  let end = document.getElementById("end-month").value;
  if (start > end) {
    end = state.data.calendar.available_end_months.find(value => value >= start) || state.data.calendar.end;
    document.getElementById("end-month").value = end;
  }
  state.end = end;
  state.start = start;
  renderPeriod();
}

function resetWindow() {
  state.start = state.data.calendar.start;
  state.end = state.data.calendar.end;
  document.getElementById("start-month").value = state.start;
  document.getElementById("end-month").value = state.end;
  renderPeriod();
}

function renderPeriod() {
  document.getElementById("window-label").textContent = `${monthLabel(state.start)} → ${monthLabel(state.end)}`;
  renderKpis();
  renderMetricTable();
  drawPerformance();
  drawDrawdown();
  renderMatrices();
}

function renderKpis() {
  document.getElementById("kpi-grid").innerHTML = CORE_METRICS.map(field => {
    const [label,type] = METRICS[field];
    const strategies = state.data.strategy_order.map(strategy => {
      const value = metricValue(strategy, field);
      const status = comparisonState(field, strategy, value);
      return `<div class="kpi-strategy-row comparison-${status}">
        <span class="strategy-name"><i style="background:${strategyMeta(strategy).color}"></i>${escapeHtml(strategy)}</span>
        <strong>${format(value,type)}</strong><small>${comparisonMark(status)}</small>
      </div>`;
    }).join("");
    return `<article class="kpi-card"><header><span>${label}</span><small>Référence SPY</small></header><div class="kpi-strategy-list">${strategies}</div></article>`;
  }).join("");
}

function renderMetricTable() {
  const rows = currentMetricRows();
  document.getElementById("metric-head").innerHTML = `<th>KPI</th>${state.data.strategy_order.map(strategy => `<th class="${strategy===BENCHMARK_STRATEGY?"benchmark-head":""}"><i style="background:${strategyMeta(strategy).color}"></i>${escapeHtml(strategy)}</th>`).join("")}<th>Définition</th>`;
  document.getElementById("metric-body").innerHTML = state.data.metric_fields.map((field,index) => {
    const [label,type,definition] = METRICS[field] || [field,"num",""];
    const values = state.data.strategy_order.map((strategy,strategyIndex) => {
      const value = rows[strategyIndex]?.[index];
      const status = comparisonState(field, strategy, value);
      return `<td class="metric-value comparison-${status}"><strong>${format(value,type)}</strong><small>${comparisonMark(status)}</small></td>`;
    }).join("");
    return `<tr><td>${escapeHtml(label)}</td>${values}<td class="metric-definition">${escapeHtml(definition)}</td></tr>`;
  }).join("");
}

function periodMonthly(strategy) {
  return state.data.monthly.filter(row => row.strategy === strategy && row.holding_month >= state.start && row.holding_month <= state.end);
}

function chartStrategies() {
  return state.curves;
}

function wealthSeries(strategy) {
  let wealth = 1;
  return periodMonthly(strategy).map(row => ({date:row.holding_month, value:(wealth *= 1 + row.net_return)}));
}

function drawdownSeries(strategy) {
  let wealth = 1, peak = 1;
  return periodMonthly(strategy).map(row => {
    wealth *= 1 + row.net_return; peak = Math.max(peak, wealth);
    return {date:row.holding_month, value:wealth / peak - 1};
  });
}

function drawPerformance() {
  const series = chartStrategies().map(name => ({name, color:strategyMeta(name).color, values:wealthSeries(name)}));
  drawLineChart(document.getElementById("wealth-chart"), series, value => `${value.toFixed(2)}×`, false);
  renderLegend("wealth-legend", series);
}

function drawDrawdown() {
  const series = chartStrategies().map(name => ({name, color:strategyMeta(name).color, values:drawdownSeries(name)}));
  drawLineChart(document.getElementById("drawdown-chart"), series, value => `${(100*value).toFixed(0)}%`, true);
  renderLegend("drawdown-legend", series);
}

function renderLegend(id, series) {
  document.getElementById(id).innerHTML = series.map(item => `<span><i style="background:${item.color}"></i>${escapeHtml(item.name)}</span>`).join("");
}

function drawLineChart(canvas, series, tickFormat, zeroLine) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(600, rect.width * ratio); canvas.height = 330 * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio,ratio);
  const W = canvas.width/ratio, H=330, pad={l:54,r:18,t:16,b:30};
  const values = series.flatMap(item => item.values.map(point => point.value));
  if (!values.length) return;
  let min=Math.min(...values), max=Math.max(...values); if (min===max) {min-=.1; max+=.1;}
  if (zeroLine) max=Math.max(0,max);
  ctx.clearRect(0,0,W,H); ctx.font="11px IBM Plex Mono, monospace"; ctx.fillStyle="#617087";
  ctx.strokeStyle="#e1e7ee"; ctx.lineWidth=1;
  for (let i=0;i<5;i++) { const y=pad.t+(H-pad.t-pad.b)*i/4; const value=max-(max-min)*i/4; ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(tickFormat(value),4,y+4); }
  const length=Math.max(...series.map(item=>item.values.length));
  series.forEach(item => { ctx.strokeStyle=item.color;ctx.lineWidth=item.name===BENCHMARK_STRATEGY?2.4:1.8;ctx.beginPath();item.values.forEach((point,index)=>{const x=pad.l+(W-pad.l-pad.r)*(length===1?0:index/(length-1));const y=pad.t+(H-pad.t-pad.b)*(max-point.value)/(max-min);index?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke(); });
  ctx.fillStyle="#617087";ctx.textAlign="left";ctx.fillText(monthLabel(state.start),pad.l,H-8);ctx.textAlign="right";ctx.fillText(monthLabel(state.end),W-pad.r,H-8);ctx.textAlign="left";
}

function viridis(value) {
  const x=Math.max(0,Math.min(1,value)); const scaled=x*(VIRIDIS.length-1); const i=Math.min(VIRIDIS.length-2,Math.floor(scaled)); const t=scaled-i;
  const rgb=VIRIDIS[i].map((v,k)=>Math.round(v+(VIRIDIS[i+1][k]-v)*t)); return `rgb(${rgb.join(",")})`;
}

function matrixYears() {
  const first = Number(state.start.slice(0,4));
  const last = Number(state.end.slice(0,4));
  return Array.from({length:last-first+1}, (_,index) => first+index);
}

function yearBoundary(year, side) {
  const values = side === "start" ? state.data.calendar.available_start_months : state.data.calendar.available_end_months;
  return values.find(value => Number(value.slice(0,4)) === year);
}

function matrixWindows(mode) {
  const endYear = Number(state.end.slice(0,4));
  return matrixYears().map(year => {
    const start = yearBoundary(year, "start");
    const end = mode === "cumulative" || year === endYear ? state.end : yearBoundary(year, "end");
    if (!start || !end || start > end) return null;
    const rows = state.data.metric_windows[`${start}|${end}`];
    return rows ? {year,start,end,rows} : null;
  }).filter(Boolean);
}

function heatmapValue(window, strategy, field) {
  const strategyIndex = state.data.strategy_order.indexOf(strategy);
  const metricIndex = state.data.metric_fields.indexOf(field);
  return window.rows[strategyIndex]?.[metricIndex];
}

function renderHeatmap(id, windows, field) {
  const values = windows.flatMap(window => state.data.strategy_order.map(strategy => heatmapValue(window,strategy,field)));
  const colorValues = values.map(value => field === "max_drawdown" ? Math.abs(value) : value).filter(Number.isFinite);
  const min = Math.min(...colorValues), max = Math.max(...colorValues);
  let html = `<div class="heatmap-head"></div>${windows.map(window => `<div class="heatmap-head">${window.year}</div>`).join("")}`;
  state.data.strategy_order.forEach(strategy => {
    html += `<div class="heatmap-label">${escapeHtml(strategy)}</div>`;
    windows.forEach(window => {
      const shown = heatmapValue(window,strategy,field);
      const raw = field === "max_drawdown" ? Math.abs(shown) : shown;
      const level = Number.isFinite(raw) && max > min ? (raw-min)/(max-min) : .5;
      const text = level > .62 ? "#172033" : "#fff";
      const title = `${strategy} · ${monthLabel(window.start)} → ${monthLabel(window.end)}`;
      html += `<div class="heatmap-cell" style="background:${viridis(level)};color:${text}" title="${escapeHtml(title)}">${format(shown,"pct")}</div>`;
    });
  });
  const matrix = document.getElementById(id);
  matrix.style.gridTemplateColumns = `220px repeat(${windows.length}, minmax(72px,1fr))`;
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

function filteredHoldings() {
  const strategy=document.getElementById("portfolio-strategy").value, month=document.getElementById("portfolio-month").value, query=document.getElementById("ticker-search").value.trim().toUpperCase();
  return state.data.holdings.filter(row=>row.strategy===strategy&&row.holding_month===month&&(!query||row.ticker.includes(query)));
}

function renderHoldings() {
  const rows=filteredHoldings(), size=100, pages=Math.max(1,Math.ceil(rows.length/size)); state.page=Math.min(state.page,pages-1); const pageRows=rows.slice(state.page*size,(state.page+1)*size);
  const totalWeight=rows.reduce((sum,row)=>sum+row.target_weight,0); document.getElementById("portfolio-summary").textContent=`${rows.length} ligne(s) · poids total ${format(totalWeight,"pct")} · portefeuille détenu en ${monthLabel(document.getElementById("portfolio-month").value)}`;
  document.getElementById("holdings-body").innerHTML=pageRows.map(row=>`<tr><td>${escapeHtml(row.ticker)}</td><td>${row.selection_rank??"—"}</td><td>${format(row.target_weight,"pct")}</td><td>${format(row.score,"num")}</td><td>${escapeHtml(row.sector||"—")}</td><td class="${row.realized_return>=0?"value-positive":"value-negative"}">${format(row.realized_return,"pct")}</td><td>${row.n_models??"—"}</td></tr>`).join("") || `<tr><td colspan="7">Aucune position pour ce filtre.</td></tr>`;
  document.getElementById("page-label").textContent=`Page ${state.page+1} / ${pages}`;
}

function exportHoldings() {
  const rows=filteredHoldings(); if (!rows.length) return;
  const fields=["strategy","decision_month","holding_month","ticker","target_weight","selection_rank","score","sector","realized_return","n_models"];
  const csv=[fields.join(","),...rows.map(row=>fields.map(field=>`"${String(row[field]??"").replaceAll('"','""')}"`).join(","))].join("\n");
  const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));link.download="alpharank_portefeuille_filtre.csv";link.click();URL.revokeObjectURL(link.href);
}

function renderMethodologies() {
  document.getElementById("method-grid").innerHTML=state.data.methodologies.map(card=>`<article class="method-card"><h3>${escapeHtml(card.title)}</h3><span class="method-status">${escapeHtml(card.status)}</span><p>${escapeHtml(card.summary)}</p><ol>${card.pseudo_code.map(step=>`<li>${escapeHtml(step)}</li>`).join("")}</ol></article>`).join("");
}

function renderLineage() {
  const D=state.data; document.getElementById("lineage-contracts").innerHTML=Object.entries(D.contracts).map(([key,value])=>`<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(typeof value==="object"?JSON.stringify(value):value)}</dd>`).join("");
  document.getElementById("lineage-data").innerHTML=`<dt>composition_id</dt><dd>${escapeHtml(D.lineage.composition_id)}</dd><dt>replay_commit</dt><dd>${escapeHtml(D.lineage.replay_git_commit)}</dd><dt>snapshot</dt><dd>${escapeHtml(D.lineage.snapshot_dir)}</dd>${D.lineage.source_files.map(row=>`<dt>${escapeHtml(row.role)}</dt><dd>${escapeHtml(row.sha256)}</dd>`).join("")}`;
}

function renderMetadata() {
  const D=state.data; document.getElementById("report-generated").textContent=D.generated_at_utc; document.getElementById("report-calendar").textContent=`${D.calendar.months} mois · ${D.calendar.start.slice(0,7)} → ${D.calendar.end.slice(0,7)}`;document.getElementById("status-message").textContent=D.status.message;
}

function observeNavigation() {
  const links=[...document.querySelectorAll(".nav-link")]; const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting)links.forEach(link=>link.classList.toggle("is-active",link.hash===`#${entry.target.id}`));}),{rootMargin:"-25% 0px -60%"}); document.querySelectorAll(".section").forEach(section=>observer.observe(section));
}

async function boot() {
  try {
    state.data=await decodePayload(); document.getElementById("loading").hidden=true;document.getElementById("app").hidden=false;
    initializeControls();renderMetadata();renderMethodologies();renderLineage();renderHoldings();renderPeriod();observeNavigation();window.addEventListener("resize",()=>{drawPerformance();drawDrawdown();});
  } catch (error) { document.getElementById("loading").textContent=`Rapport illisible : ${error.message}`; }
}
boot();
"""
