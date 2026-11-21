import React, { useEffect, useRef, useState } from "react";
import {
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  createChart,
} from "lightweight-charts";
import {
  BENCHMARK,
  colorFor,
  contiguousSegments,
  dateLabel,
  format,
  sampledPoints,
  shortName,
  sourcePolicy,
  timeKey,
} from "./format.js";

function chartOptions(dark, height, timeVisible) {
  return {
    autoSize: true,
    height,
    layout: {
      background: {
        type: ColorType.Solid,
        color: dark ? "#152234" : "#ffffff",
      },
      textColor: dark ? "#a9b9cb" : "#66778d",
      fontFamily: '"SF Pro Text", "Helvetica Neue", Arial, sans-serif',
      fontSize: 12,
      attributionLogo: !timeVisible,
    },
    grid: {
      vertLines: { visible: false },
      horzLines: { color: dark ? "#233448" : "#edf1f6" },
    },
    crosshair: {
      mode: CrosshairMode.Normal,
      vertLine: { color: "#97a8bf", width: 1, style: LineStyle.Dashed },
      horzLine: { color: "#97a8bf", width: 1, style: LineStyle.Dashed },
    },
    rightPriceScale: {
      borderVisible: false,
      minimumWidth: 82,
      scaleMargins: { top: 0.1, bottom: 0.1 },
    },
    leftPriceScale: { visible: false },
    timeScale: {
      visible: timeVisible,
      borderVisible: false,
      rightOffset: 0,
      fixLeftEdge: true,
      fixRightEdge: true,
      timeVisible: false,
      lockVisibleTimeRangeOnResize: true,
    },
    localization: { locale: "fr-FR", dateFormat: "dd MMM yyyy" },
    handleScroll: {
      mouseWheel: false,
      pressedMouseMove: true,
      horzTouchDrag: true,
      vertTouchDrag: false,
    },
    handleScale: {
      mouseWheel: true,
      pinch: true,
      axisPressedMouseMove: { time: true, price: false },
    },
  };
}

function addLines(chart, names, data, field, resolution) {
  const handles = new Map();
  names.forEach((name) => {
    const points = sampledPoints(
      data.series[name]?.points || [],
      resolution,
      field,
    );
    contiguousSegments(points, field).forEach((segment) => {
      const line = chart.addSeries(LineSeries, {
        color: colorFor(name),
        lineWidth: name === BENCHMARK ? 2 : 3,
        lineStyle: name === BENCHMARK ? LineStyle.Dashed : LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerRadius: 4,
        priceFormat: {
          type: "custom",
          minMove: 0.0001,
          formatter: (value) =>
            field === "drawdown"
              ? format(value, "pct")
              : `${format(value, "wealth")}×`,
        },
      });
      line.setData(segment);
      if (!handles.has(name)) handles.set(name, line);
    });
  });
  chart.timeScale().fitContent();
  return handles;
}

export function FinancialCharts({
  data,
  names,
  relative,
  resolution,
  dark,
  onAnalyzeZoom,
}) {
  const capitalRef = useRef(null),
    drawdownRef = useRef(null),
    charts = useRef(null);
  const [hover, setHover] = useState(null),
    [zoom, setZoom] = useState(null);
  const field = relative ? "relative_wealth" : "wealth";
  useEffect(() => {
    if (!capitalRef.current || !data || !names.length) return;
    const capital = createChart(
      capitalRef.current,
      chartOptions(dark, 290, false),
    );
    const drawdown = createChart(
      drawdownRef.current,
      chartOptions(dark, 112, true),
    );
    const capitalLines = addLines(capital, names, data, field, resolution);
    const drawdownLines = addLines(
      drawdown,
      names,
      data,
      "drawdown",
      resolution,
    );
    const lookup = new Map(
      names.map((name) => [
        name,
        new Map(
          (data.series[name]?.points || []).map((point) => [point.date, point]),
        ),
      ]),
    );
    let syncing = false,
      ready = false;
    const move = (other, handles, targetField) => (param) => {
      if (syncing) return;
      const day = timeKey(param.time);
      syncing = true;
      if (!day || !param.point) {
        setHover(null);
        other.clearCrosshairPosition();
      } else {
        const rows = names.map((name) => ({
          name,
          point: lookup.get(name)?.get(day),
        }));
        setHover({ day, rows });
        const valid = rows.find(
          (row) =>
            typeof row.point?.[targetField] === "number" &&
            handles.has(row.name),
        );
        if (valid)
          other.setCrosshairPosition(
            valid.point[targetField],
            param.time,
            handles.get(valid.name),
          );
        else other.clearCrosshairPosition();
      }
      syncing = false;
    };
    const synchronize = (other, reportZoom) => (range) => {
      if (syncing || !range) return;
      syncing = true;
      other.timeScale().setVisibleRange(range);
      syncing = false;
      if (ready && reportZoom) {
        const from = timeKey(range.from),
          to = timeKey(range.to);
        const points = data.series[names[0]]?.points || [];
        setZoom(
          from > points[0]?.date || to < points.at(-1)?.date
            ? { from, to }
            : null,
        );
      }
    };
    capital.subscribeCrosshairMove(move(drawdown, drawdownLines, "drawdown"));
    drawdown.subscribeCrosshairMove(move(capital, capitalLines, field));
    capital
      .timeScale()
      .subscribeVisibleTimeRangeChange(synchronize(drawdown, true));
    drawdown
      .timeScale()
      .subscribeVisibleTimeRangeChange(synchronize(capital, true));
    const frame = requestAnimationFrame(() => {
      ready = true;
    });
    charts.current = { capital, drawdown };
    setHover(null);
    setZoom(null);
    capitalRef.current.dataset.chartEngine = "lightweight-charts";
    capitalRef.current._analysisView = { data, names, field, resolution };
    capitalRef.current._chartApi = capital;
    return () => {
      cancelAnimationFrame(frame);
      charts.current = null;
      capital.remove();
      drawdown.remove();
    };
  }, [data, names.join("|"), field, resolution, dark]);
  const selectedRows =
    hover?.rows ||
    names.map((name) => ({ name, point: data.series[name]?.points.at(-1) }));
  const reset = () => {
    charts.current?.capital.timeScale().fitContent();
    charts.current?.drawdown.timeScale().fitContent();
    setZoom(null);
  };
  return (
    <section className="financial-chart" aria-label="Courbes journalières">
      <header className="chart-title-row">
        <h2>{relative ? "Capital relatif au SPY" : "Capital, base 1"}</h2>
        <span className="chart-date" id="hover-date">
          {dateLabel(hover?.day || data.end_date)}
        </span>
      </header>
      <div className="live-legend" id="live-legend" aria-live="off">
        {selectedRows.map(({ name, point }) => (
          <div className="live-series" key={name} title={name}>
            <i style={{ background: colorFor(name) }} />
            <span>{shortName(name)}</span>
            <strong>
              {point?.[field] == null
                ? "indisponible"
                : `${format(point[field], "wealth")}×`}
            </strong>
          </div>
        ))}
      </div>
      <div
        className="capital-chart"
        id="capital-chart"
        ref={capitalRef}
        role="img"
        aria-label="Capital quotidien des stratégies sélectionnées"
      />
      <div className="drawdown-label">
        <h3>Drawdown</h3>
        <span>Depuis le plus-haut de cette fenêtre</span>
      </div>
      <div
        className="drawdown-chart"
        id="daily-drawdown-chart"
        ref={drawdownRef}
        role="img"
        aria-label="Drawdown quotidien, même fenêtre et même sélection"
      />
      <footer className="chart-foot">
        <span>Molette : zoom · glisser : déplacer</span>
        <div>
          {zoom && (
            <button
              className="text-action"
              id="analyze-zoom"
              onClick={() => onAnalyzeZoom(zoom)}
            >
              Analyser cette plage
            </button>
          )}
          <button className="text-action" id="reset-chart-zoom" onClick={reset}>
            Réinitialiser le zoom
          </button>
        </div>
      </footer>
      {zoom && (
        <p className="zoom-notice">
          Zoom d’affichage. Les KPI restent sur la période choisie jusqu’à «
          Analyser cette plage ».
        </p>
      )}
      <p className="chart-attribution">
        Graphiques{" "}
        <a
          href="https://www.tradingview.com/"
          target="_blank"
          rel="noopener noreferrer"
        >
          TradingView Lightweight Charts™
        </a>{" "}
        · valeurs calculées par AlphaRank
      </p>
    </section>
  );
}

const PROFILE_LABELS = {
  minimum_volatility: "Volatilité minimale",
  maximum_sharpe: "Sharpe maximal",
  maximum_calmar: "Calmar maximal",
};
function weightsText(weights, order) {
  return weights
    .map(
      (weight, index) => `${shortName(order[index])} ${format(weight, "pct")}`,
    )
    .join(" · ");
}

function FrontierScatter({ frontier, kind, dark }) {
  const [active, setActive] = useState(null);
  const field =
    kind === "volatility" ? "annualized_volatility" : "max_drawdown";
  const points = (frontier.points || [])
    .map((point, index) => ({
      ...point,
      index,
      x:
        typeof point[field] === "number"
          ? field === "max_drawdown"
            ? Math.abs(point[field])
            : point[field]
          : null,
      y: point.cagr,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const references = Object.entries(frontier.reference_metrics || {})
    .map(([name, point]) => ({
      ...point,
      name,
      x:
        typeof point[field] === "number"
          ? field === "max_drawdown"
            ? Math.abs(point[field])
            : point[field]
          : null,
      y: point.cagr,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!points.length)
    return (
      <div className="empty-state">
        Pas de frontière calculable : la fenêtre doit avoir des clôtures
        complètes.
      </div>
    );
  const all = [...points, ...references];
  const xmin = Math.min(...all.map((p) => p.x)),
    xmax = Math.max(...all.map((p) => p.x));
  const ymin = Math.min(...all.map((p) => p.y)),
    ymax = Math.max(...all.map((p) => p.y));
  const xrange = Math.max(0.01, xmax - xmin),
    yrange = Math.max(0.01, ymax - ymin);
  const x = (value) =>
    62 + ((value - xmin + xrange * 0.08) / (xrange * 1.16)) * 438;
  const y = (value) =>
    270 - ((value - ymin + yrange * 0.08) / (yrange * 1.16)) * 232;
  const indices =
    kind === "volatility"
      ? frontier.volatility_frontier_indices
      : frontier.drawdown_frontier_indices;
  const line = (indices || [])
    .map((index) => points.find((point) => point.index === index))
    .filter(Boolean)
    .sort((a, b) => a.x - b.x);
  return (
    <article className="frontier-chart-card">
      <h3>
        {kind === "volatility"
          ? "Rendement / volatilité"
          : "Rendement / drawdown"}
      </h3>
      <p>Plus haut : plus de rendement. Plus à gauche : moins de risque.</p>
      <svg
        viewBox="0 0 530 322"
        role="img"
        aria-label={`Frontière rendement ${kind === "volatility" ? "volatilité" : "drawdown"}`}
      >
        {[0, 1, 2, 3, 4].map((index) => {
          const xv = xmin + (xrange * index) / 4,
            yv = ymin + (yrange * index) / 4;
          return (
            <g key={index}>
              <line
                x1="62"
                x2="500"
                y1={y(yv)}
                y2={y(yv)}
                stroke={dark ? "#293a4d" : "#e8eef5"}
              />
              <text x="53" y={y(yv) + 4} textAnchor="end">
                {format(yv, "pct")}
              </text>
              <text x={x(xv)} y="293" textAnchor="middle">
                {format(xv, "pct")}
              </text>
            </g>
          );
        })}
        <polyline
          points={line.map((point) => `${x(point.x)},${y(point.y)}`).join(" ")}
          fill="none"
          stroke="#315cf4"
          strokeWidth="2.5"
        />
        {points.map((point) => (
          <circle
            key={point.index}
            cx={x(point.x)}
            cy={y(point.y)}
            r={active?.index === point.index ? 6 : 3.5}
            fill={active?.index === point.index ? "#315cf4" : "#a9b9cc"}
            onMouseEnter={() => setActive(point)}
            onFocus={() => setActive(point)}
            tabIndex="0"
            role="button"
            aria-label={`${weightsText(point.weights, frontier.strategy_order)}, CAGR ${format(point.cagr, "pct")}`}
          >
            <title>{weightsText(point.weights, frontier.strategy_order)}</title>
          </circle>
        ))}
        {references.map((point) => (
          <rect
            key={point.name}
            x={x(point.x) - 4}
            y={y(point.y) - 4}
            width="8"
            height="8"
            fill={colorFor(point.name)}
            onMouseEnter={() => setActive(point)}
          >
            <title>{shortName(point.name)}</title>
          </rect>
        ))}
        <text x="20" y="20">
          CAGR
        </text>
        <text x="280" y="319" textAnchor="middle">
          {kind === "volatility" ? "Volatilité annualisée" : "Drawdown absolu"}
        </text>
      </svg>
      <div className="frontier-hover">
        {active ? (
          <>
            <strong>
              {active.name
                ? shortName(active.name)
                : weightsText(active.weights, frontier.strategy_order)}
            </strong>
            <span>
              CAGR {format(active.cagr, "pct")} · risque{" "}
              {format(active.x, "pct")}
            </span>
          </>
        ) : (
          "Survolez un point pour lire sa composition et ses mesures."
        )}
      </div>
    </article>
  );
}

export function Frontiers({ data, dark }) {
  const frontier = data?.frontier;
  if (!frontier)
    return (
      <div className="empty-state">
        Les frontières seront calculées pour la période sélectionnée.
      </div>
    );
  return (
    <section className="frontier-view">
      <p className="method-note">
        Diagnostic post-hoc : trois poches, poids longs par pas de 5 %, plafond
        de 70 %, {sourcePolicy(data.source).frontier}. Ces profils ne sont pas
        des recommandations causales.
      </p>
      <div className="profile-strip">
        {Object.entries(frontier.recommendations || {}).map(([key, index]) => {
          const point = frontier.points?.[index];
          return (
            point && (
              <article key={key}>
                <span>{PROFILE_LABELS[key] || key}</span>
                <strong>
                  {format(point.cagr, "pct")} <small>CAGR</small>
                </strong>
                <p>{weightsText(point.weights, frontier.strategy_order)}</p>
                <div>
                  Vol. {format(point.annualized_volatility, "pct")} · DD{" "}
                  {format(point.max_drawdown, "pct")}
                </div>
              </article>
            )
          );
        })}
      </div>
      <div className="frontier-panels">
        <FrontierScatter frontier={frontier} kind="volatility" dark={dark} />
        <FrontierScatter frontier={frontier} kind="drawdown" dark={dark} />
      </div>
    </section>
  );
}
