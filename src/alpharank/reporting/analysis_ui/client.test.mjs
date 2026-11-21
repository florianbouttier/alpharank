import assert from "node:assert/strict";
import test from "node:test";
import {
  sourcePolicy,
  BENCHMARK,
  colorFor,
  contiguousSegments,
  format,
  sampledPoints,
  timeKey,
} from "./format.js";

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
