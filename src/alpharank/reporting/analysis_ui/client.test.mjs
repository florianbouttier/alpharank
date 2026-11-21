import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { buildSync } from "esbuild";
import {
  sourcePolicy,
  BENCHMARK,
  colorFor,
  contiguousSegments,
  format,
  metricDefinition,
  sampledPoints,
  timeKey,
} from "./format.js";

const rendered = buildSync({
  stdin: {
    contents: `import React from 'react';
      import {renderToStaticMarkup} from 'react-dom/server';
      import {Frontiers} from './charts.jsx';
      import {CoverageNotice} from './coverage.jsx';
      import {PeriodControls} from './controls.jsx';
      import {Correlations, KpiTable} from './tables.jsx';
      export const frontier = data => renderToStaticMarkup(React.createElement(Frontiers, {data, dark:false}));
      export const coverage = props => renderToStaticMarkup(React.createElement(CoverageNotice, props));
      export const controls = props => renderToStaticMarkup(React.createElement(PeriodControls, props));
      export const correlations = props => renderToStaticMarkup(React.createElement(Correlations, props));
      export const kpis = props => renderToStaticMarkup(React.createElement(KpiTable, props));`,
    resolveDir: fileURLToPath(new URL(".", import.meta.url)),
  },
  bundle: true,
  platform: "node",
  format: "cjs",
  write: false,
  define: { "process.env.NODE_ENV": '"production"' },
});
const renderedModule = { exports: {} };
new Function("module", "exports", "require", rendered.outputFiles[0].text)(
  renderedModule,
  renderedModule.exports,
  createRequire(import.meta.url),
);

test("frontier keeps complete references when no mix has observable risk", () => {
  const html = renderedModule.exports.frontier({
    source: "history",
    frontier: {
      points: [
        {
          weights: [0.5, 0.5],
          cagr: 0.2,
          annualized_volatility: null,
          max_drawdown: null,
        },
      ],
      strategy_order: ["Legacy · Frequency", "Boosting · Top 5"],
      recommendations: {},
      reference_metrics: {
        [BENCHMARK]: {
          cagr: 0.1,
          annualized_volatility: 0.15,
          max_drawdown: -0.2,
        },
        "Boosting · Top 5": {
          cagr: 0.2,
          annualized_volatility: 0.3,
          max_drawdown: -0.35,
        },
      },
    },
  });
  assert.equal((html.match(/<svg/g) || []).length, 2);
  assert.equal((html.match(/<rect/g) || []).length, 4);
  assert.match(html, /Aucun mélange admissible/);
});

test("source policy never labels the frozen gross cohort as monthly net rebalancing", () => {
  const recent = sourcePolicy("recent");
  assert.match(recent.composer, /à l’entrée/);
  assert.match(recent.composer, /brut de frais/);
  assert.match(recent.frontier, /sans rééquilibrage/);
  assert.match(recent.sidebar, /sans rééquilibrage/);
  assert.doesNotMatch(
    Object.values(recent).join(" "),
    /chaque mois|mensuel|Frais propres inclus/,
  );
  assert.match(sourcePolicy("history").composer, /chaque mois/);
  assert.match(sourcePolicy("history").composer, /Frais propres inclus/);
  assert.deepEqual(sourcePolicy("monthly_reference"), sourcePolicy("history"));
});

test("monthly risk labels never claim daily observations", () => {
  for (const field of [
    "annualized_volatility",
    "max_drawdown",
    "tracking_error",
    "correlation",
    "var_95",
    "cvar_95",
    "benchmark_hit_rate",
    "omega",
  ]) {
    const definition = metricDefinition(field, "monthly");
    assert.doesNotMatch(definition.join(" "), /quotidien|journali|252/);
    assert.match(definition.join(" "), /mois|mensuel/);
  }
  assert.match(metricDefinition("annualized_volatility", "daily")[2], /252/);
  const html = renderedModule.exports.correlations({
    correlations: {
      frequency: "monthly",
      strategies: [BENCHMARK],
      returns: [[1]],
      active_returns: [[null]],
    },
  });
  assert.match(html, /rendements mensuels/);
  assert.doesNotMatch(html, /quotidien|journalier/);
});

test("monthly KPI table preserves monthly tails and does not expose positive-day rates", () => {
  const html = renderedModule.exports.kpis({
    data: {
      frequency: "monthly",
      series: {
        [BENCHMARK]: {
          metrics: {
            positive_day_rate: 0.5,
            positive_month_rate: 0.6,
            var_95: -0.1,
            cvar_95: -0.2,
          },
        },
      },
    },
    names: [BENCHMARK],
    full: true,
  });
  assert.match(html, /Mois positifs/);
  assert.match(html, /VaR 95 % mensuelle/);
  assert.doesNotMatch(html, /Séances positives|VaR 95 % quotidienne/);
});

test("coverage audit is restricted to displayed sleeves and fallback names exact changed bounds", () => {
  const data = {
    frequency: "daily",
    series: { Visible: { coverage: { missing_closes: 13 } } },
    price_gap_audit_available: true,
    price_gaps: [
      {
        strategy: "Visible",
        ticker: "NLSN",
        holding_month: "2022-10-01",
        first_missing_session: "2022-10-13",
        last_missing_session: "2022-10-31",
        missing_sessions: 13,
        target_weight: 0.2,
        dates: ["2022-10-13"],
      },
      { strategy: "Hidden", ticker: "ABMD" },
    ],
    monthly_fallback: {
      available: true,
      source: "monthly_reference",
      start: "2020-02-01",
      end: "2026-07-31",
      period_changed: true,
    },
  };
  const html = renderedModule.exports.coverage({
    data,
    names: ["Visible"],
    onMonthlyFallback: () => {},
  });
  assert.match(html, /NLSN/);
  assert.doesNotMatch(html, /ABMD/);
  assert.match(html, /apply-monthly-fallback/);
  assert.match(html, /1 févr\. 2020/);
  assert.match(html, /31 juil\. 2026/);
  assert.match(html, /bornes modifiées/);
  const monthly = renderedModule.exports.coverage({
    data: { ...data, frequency: "monthly" },
    names: ["Visible"],
  });
  assert.match(monthly, /12 observations\/an/);
  assert.match(monthly, /ne corrige pas les anciennes hypothèses/);
  assert.match(monthly, /NLSN/);
  assert.doesNotMatch(monthly, /apply-monthly-fallback/);
});

test("date drafts are explicitly unapplied and monthly view offers no daily resolution", () => {
  const props = {
    settings: {
      sources: {
        history: { label: "Backtest" },
        monthly_reference: { label: "Référence mensuelle" },
      },
    },
    source: "history",
    draft: { start: "2020-01-01", end: "2026-02-28" },
    period: { start: "2020-01-01", end: "2026-08-31" },
    onSource: () => {},
    setDraft: () => {},
    onApply: () => {},
    resolution: "D1",
    setResolution: () => {},
  };
  const html = renderedModule.exports.controls(props);
  assert.match(html, /Dates modifiées, non appliquées/);
  assert.match(html, /31 août 2026/);
  const applied = renderedModule.exports.controls({
    ...props,
    draft: props.period,
  });
  assert.doesNotMatch(applied, /unapplied-dates/);
  const monthly = renderedModule.exports.controls({
    ...props,
    source: "monthly_reference",
  });
  assert.match(monthly, /12 observations\/an/);
  assert.match(monthly, /Mois calendaires complets uniquement/);
  assert.doesNotMatch(monthly, /Résolution visuelle|Chaque séance/);
});

test("daily presentation preserves every supplied observation and its anchor", () => {
  const points = [
    { date: "2025-12-31", wealth: 1 },
    { date: "2026-01-02", wealth: 1.03 },
    { date: "2026-01-05", wealth: 1.01 },
  ];
  assert.equal(sampledPoints(points, "D1", "wealth"), points);
  assert.equal(points[0].wealth, 1);
});

test("coarser presentation retains gaps instead of hiding them between weekly points", () => {
  const points = [
    { date: "2025-12-31", wealth: 1 },
    { date: "2026-01-02", wealth: 1.03 },
    { date: "2026-01-05", wealth: null },
    { date: "2026-01-06", wealth: 1.01 },
    { date: "2026-01-09", wealth: 1.02 },
  ];
  for (const resolution of ["W1", "M1"]) {
    const sampled = sampledPoints(points, resolution, "wealth");
    assert.equal(sampled[0], points[0]);
    assert(sampled.includes(points[2]));
    assert.equal(sampled.at(-1), points.at(-1));
  }
});

test("separate line segments cannot bridge unavailable observations", () => {
  const segments = contiguousSegments(
    [
      { date: "2026-01-01", drawdown: 0 },
      { date: "2026-01-02", drawdown: -0.1 },
      { date: "2026-01-03", drawdown: null },
      { date: "2026-01-04", drawdown: -0.05 },
    ],
    "drawdown",
  );
  assert.deepEqual(
    segments.map((segment) => segment.length),
    [2, 1],
  );
  assert.equal(segments[0][0].value, 0);
  assert(!segments.flat().some((point) => point.value === null));
});

test("missing or undefined KPI are not displayed as zero", () => {
  for (const value of [null, undefined, NaN, Infinity])
    assert.equal(format(value, "pct"), "—");
  assert.notEqual(format(0, "pct"), "—");
});

test("native chart dates stay zero padded and benchmark color is stable", () => {
  assert.equal(timeKey({ year: 2026, month: 1, day: 5 }), "2026-01-05");
  assert.equal(timeKey("2026-01-05"), "2026-01-05");
  assert.equal(colorFor(BENCHMARK), "#8a97ab");
});
