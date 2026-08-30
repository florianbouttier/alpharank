PERFORMANCE_REPORT_COMPOSER_SCRIPT = r"""
const COMPOSER_DEFAULT_STRATEGIES = ["Legacy · Frequency", "Boosting tendance · Top 5"];
const COMPOSER_COLOR = "#0369a1";
const COMPOSER_NAME = "Portefeuille composé";

function initializeComposer() {
  const composer = state.data.portfolio_composer;
  state.composerStrategies = COMPOSER_DEFAULT_STRATEGIES.filter(name => composer.strategy_order.includes(name));
  if (!state.composerStrategies.length) state.composerStrategies = [composer.strategy_order[0]];
  document.getElementById("composer-all").addEventListener("click", () => setComposerStrategies(composer.strategy_order));
  document.getElementById("composer-reference").addEventListener("click", () => setComposerStrategies(COMPOSER_DEFAULT_STRATEGIES));
  renderComposerOptions();
}

function renderComposerOptions() {
  const composer = state.data.portfolio_composer;
  const equalWeight = 1 / state.composerStrategies.length;
  document.getElementById("composer-options").innerHTML = composer.strategy_order.map((strategy, index) => {
    const checked = state.composerStrategies.includes(strategy) ? "checked" : "";
    const weight = checked ? `<strong>${format(equalWeight,"pct")}</strong>` : "";
    return `<label class="composer-option"><input type="checkbox" data-composer-index="${index}" ${checked}><i style="background:${strategyMeta(strategy).color}"></i><span>${escapeHtml(strategy)}</span>${weight}</label>`;
  }).join("");
  document.querySelectorAll("[data-composer-index]").forEach(input => input.addEventListener("change", event => {
    const strategy = composer.strategy_order[Number(event.currentTarget.dataset.composerIndex)];
    const next = event.currentTarget.checked
      ? [...state.composerStrategies, strategy]
      : state.composerStrategies.filter(value => value !== strategy);
    if (!next.length) {
      event.currentTarget.checked = true;
      return;
    }
    setComposerStrategies(next);
  }));
  document.getElementById("composer-summary").textContent = `${state.composerStrategies.length} poche(s) · ${format(equalWeight,"pct")} chacune · rééquilibrage mensuel`;
}

function setComposerStrategies(strategies) {
  const available = state.data.portfolio_composer.strategy_order;
  state.composerStrategies = available.filter(strategy => strategies.includes(strategy));
  if (!state.composerStrategies.length) state.composerStrategies = [available[0]];
  renderComposerOptions();
  renderComposer();
}

function composerMask() {
  return state.data.portfolio_composer.strategy_order.reduce(
    (mask, strategy, index) => state.composerStrategies.includes(strategy) ? mask + 2 ** index : mask,
    0,
  );
}

function composerCombinationIndex() {
  return state.data.portfolio_composer.combination_masks.indexOf(composerMask());
}

function composerMetricValue(field) {
  const composer = state.data.portfolio_composer;
  const rows = composer.metric_windows[windowKey()] || [];
  const metricIndex = composer.metric_fields.indexOf(field);
  return rows[composerCombinationIndex()]?.[metricIndex];
}

function composerComparisonState(field, value) {
  const benchmark = metricValue(BENCHMARK_STRATEGY, field);
  const direction = METRIC_DIRECTIONS[field];
  if (!direction || !Number.isFinite(value) || !Number.isFinite(benchmark)) return "neutral";
  if (Math.abs(value - benchmark) < 1e-12) return "equal";
  const beats = direction === "higher" ? value > benchmark : value < benchmark;
  return beats ? "beats" : "trails";
}

function renderComposerKpis() {
  document.getElementById("composer-kpis").innerHTML = CORE_METRICS.map(field => {
    const [label,type] = METRICS[field];
    const value = composerMetricValue(field);
    const benchmark = metricValue(BENCHMARK_STRATEGY, field);
    const status = composerComparisonState(field, value);
    return `<article class="composer-kpi comparison-${status}"><span>${escapeHtml(label)}</span><strong>${format(value,type)}</strong><small>${comparisonMark(status)} · SPY ${format(benchmark,type)}</small></article>`;
  }).join("");
}

function composerPeriodReturns() {
  const composer = state.data.portfolio_composer;
  const combinationIndex = composerCombinationIndex();
  return composer.months.map((month, monthIndex) => ({
    date: month,
    value: composer.monthly_returns[monthIndex][combinationIndex],
  })).filter(row => row.date >= state.start && row.date <= state.end);
}

function composerWealthSeries() {
  let wealth = 1;
  return composerPeriodReturns().map(row => ({date: row.date, value: (wealth *= 1 + row.value)}));
}

function composerDrawdownSeries() {
  let wealth = 1, peak = 1;
  return composerPeriodReturns().map(row => {
    wealth *= 1 + row.value;
    peak = Math.max(peak, wealth);
    return {date: row.date, value: wealth / peak - 1};
  });
}

function drawComposerCharts() {
  if (!state.data?.portfolio_composer || !state.composerStrategies?.length) return;
  const wealth = [
    {name: COMPOSER_NAME, color: COMPOSER_COLOR, values: composerWealthSeries()},
    {name: BENCHMARK_STRATEGY, color: strategyMeta(BENCHMARK_STRATEGY).color, values: wealthSeries(BENCHMARK_STRATEGY)},
  ];
  const drawdowns = [
    {name: COMPOSER_NAME, color: COMPOSER_COLOR, values: composerDrawdownSeries()},
    {name: BENCHMARK_STRATEGY, color: strategyMeta(BENCHMARK_STRATEGY).color, values: drawdownSeries(BENCHMARK_STRATEGY)},
  ];
  drawLineChart(document.getElementById("composer-wealth-chart"), wealth, value => `${value.toFixed(2)}×`, false);
  drawLineChart(document.getElementById("composer-drawdown-chart"), drawdowns, value => `${(100*value).toFixed(0)}%`, true);
  renderLegend("composer-wealth-legend", wealth);
  renderLegend("composer-drawdown-legend", drawdowns);
}

function renderComposer() {
  renderComposerKpis();
  drawComposerCharts();
}
"""
