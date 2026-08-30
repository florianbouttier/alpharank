PERFORMANCE_REPORT_COMPOSER_SCRIPT = r"""
const COMPOSER_DEFAULT_STRATEGIES = ["Legacy · Frequency", "Boosting tendance · Top 5"];
const COMPOSER_BOOSTING_PAIR_STRATEGIES = [
  "Boosting · Top 5",
  "Boosting tendance · Top 5",
];
const COMPOSER_COLOR = "#0369a1";
const COMPOSER_NAME = "Portefeuille composé";
const COMPOSER_DISPLAY_METRICS = [
  "cagr", "total_return", "annualized_volatility", "max_drawdown",
  "sharpe", "sortino", "correlation",
];

function initializeComposer() {
  const composer = state.data.portfolio_composer;
  state.composerStrategies = COMPOSER_DEFAULT_STRATEGIES.filter(name => composer.strategy_order.includes(name));
  if (!state.composerStrategies.length) state.composerStrategies = [composer.strategy_order[0]];
  document.getElementById("composer-all").addEventListener("click", () => setComposerStrategies(composer.strategy_order));
  document.getElementById("composer-reference").addEventListener("click", () => setComposerStrategies(COMPOSER_DEFAULT_STRATEGIES));
  document.getElementById("composer-boosting-pair").addEventListener(
    "click",
    () => setComposerStrategies(COMPOSER_BOOSTING_PAIR_STRATEGIES),
  );
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
  document.getElementById("composer-kpis").innerHTML = COMPOSER_DISPLAY_METRICS.map(field => {
    const [metricLabel,type] = METRICS[field];
    const label = field === "correlation" ? "Corrélation mensuelle au SPY" : metricLabel;
    const value = composerMetricValue(field);
    const benchmark = metricValue(BENCHMARK_STRATEGY, field);
    const status = composerComparisonState(field, value);
    const comparison = field === "correlation"
      ? "Pearson sur les rendements mensuels"
      : `${comparisonMark(status)} · SPY ${format(benchmark,type)}`;
    return `<article class="composer-kpi comparison-${status}"><span>${escapeHtml(label)}</span><strong>${format(value,type)}</strong><small>${escapeHtml(comparison)}</small></article>`;
  }).join("");
}

function composerCorrelationValue(first, second) {
  const composer = state.data.portfolio_composer;
  const matrix = composer.strategy_correlation_windows[windowKey()] || [];
  const firstIndex = composer.strategy_order.indexOf(first);
  const secondIndex = composer.strategy_order.indexOf(second);
  return matrix[firstIndex]?.[secondIndex];
}

function correlationClass(value) {
  if (!Number.isFinite(value)) return "correlation-unknown";
  if (value <= 0.3) return "correlation-diversifying";
  if (value <= 0.7) return "correlation-moderate";
  return "correlation-high";
}

function renderComposerCorrelation() {
  const strategies = state.composerStrategies;
  const columns = strategies.map(
    strategy => `<th>${escapeHtml(strategy)}</th>`,
  ).join("");
  const header = `<thead><tr><th>Poche</th>${columns}</tr></thead>`;
  const body = strategies.map(first => {
    const cells = strategies.map(second => {
      const value = composerCorrelationValue(first, second);
      return `<td class="correlation-cell ${correlationClass(value)}">${format(value,"num")}</td>`;
    }).join("");
    return `<tr><th>${escapeHtml(first)}</th>${cells}</tr>`;
  }).join("");
  document.getElementById("composer-correlation-matrix").innerHTML = `${header}<tbody>${body}</tbody>`;
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

function composerRelativeWealthSeries() {
  const spyByMonth = new Map(
    periodMonthly(BENCHMARK_STRATEGY).map(
      row => [row.holding_month, row.net_return],
    ),
  );
  let portfolioWealth = 1, spyWealth = 1;
  return composerPeriodReturns().map(row => {
    portfolioWealth *= 1 + row.value;
    spyWealth *= 1 + spyByMonth.get(row.date);
    return {date: row.date, value: portfolioWealth / spyWealth};
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
  const relative = [
    {
      name: `${COMPOSER_NAME} ÷ SPY`,
      color: COMPOSER_COLOR,
      values: composerRelativeWealthSeries(),
    },
    {
      name: "Parité SPY = 1",
      color: strategyMeta(BENCHMARK_STRATEGY).color,
      values: composerPeriodReturns().map(row => ({date: row.date, value: 1})),
    },
  ];
  drawLineChart(document.getElementById("composer-wealth-chart"), wealth, value => `${value.toFixed(2)}×`, false);
  drawLineChart(document.getElementById("composer-drawdown-chart"), drawdowns, value => `${(100*value).toFixed(0)}%`, true);
  drawLineChart(
    document.getElementById("composer-relative-chart"),
    relative,
    value => `${value.toFixed(2)}×`,
    false,
  );
  renderLegend("composer-wealth-legend", wealth);
  renderLegend("composer-drawdown-legend", drawdowns);
  renderLegend("composer-relative-legend", relative);
}

function renderComposer() {
  renderComposerKpis();
  renderComposerCorrelation();
  drawComposerCharts();
}
"""
