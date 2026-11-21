import React, { useState } from "react";
import {
  BENCHMARK,
  METRICS,
  PRIMARY_METRICS,
  colorFor,
  format,
  shortName,
  viridis,
} from "./format.js";

function metricCell(data, name, field) {
  const value = data.series[name]?.metrics[field];
  const benchmark = data.series[BENCHMARK]?.metrics[field];
  const [, type, , direction] = METRICS[field] || [field, "num"];
  const finite = typeof value === "number" && Number.isFinite(value);
  const comparable =
    name !== BENCHMARK &&
    direction &&
    finite &&
    typeof benchmark === "number" &&
    Number.isFinite(benchmark);
  const delta = comparable ? value - benchmark : null;
  const status =
    !comparable || Math.abs(delta) < 1e-12
      ? "neutral"
      : (direction === "higher" ? delta > 0 : delta < 0)
        ? "better"
        : "worse";
  const text =
    type === "years"
      ? finite && data.series[name]?.metrics.complete_years > 0
        ? `${format(value, "int")} / ${format(data.series[name]?.metrics.complete_years, "int")}`
        : "—"
      : format(value, type);
  const comparison =
    delta !== null && type !== "years"
      ? `${delta > 0 ? "+" : ""}${format(type === "pct" ? delta * 100 : delta)} ${type === "pct" ? "points" : ""} vs SPY`
      : name === BENCHMARK
        ? "Référence SPY"
        : "";
  return (
    <td
      className={`${name === BENCHMARK ? "benchmark-cell" : ""} ${status}`}
      title={`${text}${comparison ? ` ; ${comparison}` : ""}`}
    >
      <strong>{text}</strong>
      {comparison && <small>{comparison}</small>}
    </td>
  );
}

export function KpiTable({ data, names, full = false }) {
  if (!full)
    return (
      <div className="table-scroll primary-kpis">
        <table className="primary-summary-table" id="primary-kpi-table">
          <thead>
            <tr>
              <th>Stratégie</th>
              {PRIMARY_METRICS.map((field) => (
                <th key={field}>
                  <abbr title={METRICS[field][2]}>
                    {{
                      total_return: "Rendement",
                      annualized_volatility: "Volatilité",
                      worst_year_return: "Pire année",
                      worst_year_excess: "Pire année vs SPY",
                      years_beating_benchmark: "Années devant SPY",
                    }[field] || METRICS[field][0]}
                  </abbr>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name}>
                <th>
                  <i
                    className="series-dot"
                    style={{ background: colorFor(name) }}
                  />
                  {shortName(name)}
                </th>
                {PRIMARY_METRICS.map((field) => (
                  <React.Fragment key={field}>
                    {metricCell(data, name, field)}
                  </React.Fragment>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  const fields = full
    ? [
        ...new Set([
          ...PRIMARY_METRICS,
          ...Object.keys(data.series[names[0]]?.metrics || {}).filter(
            (field) => METRICS[field],
          ),
        ]),
      ]
    : PRIMARY_METRICS;
  return (
    <div className={`table-scroll ${full ? "full-kpis" : "primary-kpis"}`}>
      <table
        className="kpi-table"
        id={full ? "full-kpi-table" : "primary-kpi-table"}
      >
        <thead>
          <tr>
            <th>Mesure</th>
            {names.map((name) => (
              <th
                key={name}
                className={name === BENCHMARK ? "benchmark-cell" : ""}
              >
                <i
                  className="series-dot"
                  style={{ background: colorFor(name) }}
                />
                {shortName(name)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fields.map((field) => (
            <tr key={field}>
              <th>
                <abbr title={METRICS[field]?.[2]}>
                  {METRICS[field]?.[0] || field}
                </abbr>
              </th>
              {names.map((name) => (
                <React.Fragment key={name}>
                  {metricCell(data, name, field)}
                </React.Fragment>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Matrix({ rows, names, field, title, incremental }) {
  const years = [
    ...new Set(
      names.flatMap((name) => (rows[name] || []).map((row) => row.year)),
    ),
  ].sort((a, b) => a - b);
  const lookup = new Map(
    names.map((name) => [
      name,
      new Map((rows[name] || []).map((row) => [row.year, row])),
    ]),
  );
  const fieldFor = incremental && field === "cagr" ? "return" : field;
  const numeric = names
    .flatMap((name) => (rows[name] || []).map((row) => row[fieldFor]))
    .filter((value) => typeof value === "number" && Number.isFinite(value));
  const values = field === "max_drawdown" ? numeric.map(Math.abs) : numeric;
  const min = Math.min(...values),
    max = Math.max(...values);
  return (
    <section className="matrix-block">
      <h3>{title}</h3>
      <div className="table-scroll">
        <table className="heatmap-table">
          <thead>
            <tr>
              <th>Stratégie</th>
              {years.map((year) => (
                <th key={year}>{year}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name}>
                <th>
                  <i
                    className="series-dot"
                    style={{ background: colorFor(name) }}
                  />
                  {shortName(name)}
                </th>
                {years.map((year) => {
                  const row = lookup.get(name).get(year),
                    value = row?.[fieldFor];
                  const finite =
                    typeof value === "number" && Number.isFinite(value);
                  const measured =
                    field === "max_drawdown" ? Math.abs(value) : value;
                  const intensity = finite
                    ? max === min
                      ? 0.5
                      : (measured - min) / (max - min)
                    : 0;
                  return (
                    <td
                      className="heatmap-cell"
                      key={year}
                      style={
                        finite
                          ? {
                              background: viridis(intensity),
                              color: intensity > 0.55 ? "#142c29" : "#fff",
                            }
                          : undefined
                      }
                      title={`${name}, ${year}${row?.partial ? ", année partielle" : ""}`}
                    >
                      <strong>{format(value, "pct")}</strong>
                      {row?.partial && (
                        <span aria-label="année partielle">*</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function ModelCards({ data, names }) {
  const [field, setField] = useState("cagr");
  const annual = Object.fromEntries(
    names.map((name) => [name, data.series[name]?.annual || []]),
  );
  const cumulative = Object.fromEntries(
    names.map((name) => [
      name,
      (data.series[name]?.cumulative || []).map((row) => ({
        ...row,
        partial: annual[name]?.find((year) => year.year === row.year)?.partial,
      })),
    ]),
  );
  return (
    <div className="model-cards">
      <div className="inline-tabs" aria-label="Indicateur des model cards">
        {[
          ["cagr", "Rendement / CAGR"],
          ["annualized_volatility", "Volatilité"],
          ["max_drawdown", "Max drawdown"],
        ].map(([value, label]) => (
          <button
            key={value}
            className={field === value ? "active" : ""}
            aria-pressed={field === value}
            onClick={() => setField(value)}
          >
            {label}
          </button>
        ))}
      </div>
      <Matrix
        rows={cumulative}
        names={names}
        field={field}
        title="Depuis chaque année jusqu’à la fin choisie"
      />
      <Matrix
        rows={annual}
        names={names}
        field={field}
        incremental
        title="Chaque année isolée"
      />
      <p className="fine-print">
        Viridis encode les valeurs, qui restent écrites. * Année partielle ; le
        rendement annuel n’est pas annualisé. Une cellule vide signale une
        mesure indisponible.
      </p>
    </div>
  );
}

export function Correlations({ correlations }) {
  const [active, setActive] = useState(false);
  if (!correlations?.strategies?.length)
    return (
      <p className="fine-print">
        Les corrélations sont disponibles après le calcul d’une composition.
      </p>
    );
  const matrix = active ? correlations.active_returns : correlations.returns;
  return (
    <div>
      <div className="inline-tabs">
        <button
          className={!active ? "active" : ""}
          onClick={() => setActive(false)}
        >
          Rendements
        </button>
        <button
          className={active ? "active" : ""}
          onClick={() => setActive(true)}
        >
          Rendements actifs vs SPY
        </button>
      </div>
      <p className="fine-print">
        Pearson sur les rendements journaliers de la fenêtre, jamais sur les
        niveaux de capital. Le rendement actif est la différence quotidienne au
        SPY.
      </p>
      <div className="table-scroll">
        <table className="correlation-table">
          <thead>
            <tr>
              <th>Poche</th>
              {correlations.strategies.map((name) => (
                <th key={name}>{shortName(name)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {correlations.strategies.map((name, i) => (
              <tr key={name}>
                <th>{shortName(name)}</th>
                {correlations.strategies.map((_, j) => (
                  <td key={j}>{format(matrix?.[i]?.[j])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function AnnualTable({ data, names }) {
  const years = [
    ...new Set(
      names.flatMap((name) =>
        (data.series[name]?.annual || []).map((row) => row.year),
      ),
    ),
  ].sort((a, b) => b - a);
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Année</th>
            {names.map((name) => (
              <th key={name}>{shortName(name)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {years.map((year) => (
            <tr key={year}>
              <th>{year}</th>
              {names.map((name) => {
                const row = data.series[name]?.annual?.find(
                  (row) => row.year === year,
                );
                return (
                  <td key={name}>
                    {format(row?.return, "pct")}
                    {row?.partial ? " *" : ""}
                    <small>
                      {name !== BENCHMARK && row?.excess_return != null
                        ? `${row.excess_return > 0 ? "+" : ""}${format(row.excess_return * 100)} pts vs SPY`
                        : ""}
                    </small>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
