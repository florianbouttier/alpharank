import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { FinancialCharts, Frontiers } from "./charts.jsx";
import { AnnualTable, Correlations, KpiTable, ModelCards } from "./tables.jsx";
import { Holdings, Methods, Sources } from "./audit.jsx";
import {
  BENCHMARK,
  COMPOSED,
  DEFAULT_STRATEGIES,
  colorFor,
  dateLabel,
  format,
  groupFor,
  shortName,
} from "./format.js";

function initialTheme() {
  try {
    return localStorage.getItem("alpharank-report-theme") === "dark";
  } catch {
    return false;
  }
}

function useAnalysis(payload, query) {
  const settings = payload.analysis;
  const [data, setData] = useState(settings?.initial_view || null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(null);
  const signature = JSON.stringify(query);
  useEffect(() => {
    if (!settings) return;
    const initial = settings.initial_view;
    const matches =
      initial &&
      query.start === (initial.requested_start || settings.default_start) &&
      query.end === (initial.requested_end || settings.default_end) &&
      query.source === (settings.default_source || "history") &&
      !query.sleeves.length &&
      !query.frontier &&
      !query.retry;
    if (matches) {
      setData(initial);
      setError(null);
      setBusy(false);
      return;
    }
    if (location.protocol === "file:") {
      setError(
        "Les dates libres utilisent le moteur Python local. Ouvrez ce rapport depuis le serveur AlphaRank pour calculer cette fenêtre ; aucun ancien KPI n’est affiché à sa place.",
      );
      setBusy(false);
      return;
    }
    const abort = new AbortController();
    const url = new URL(settings.endpoint || "/api/analysis", location.href);
    if (url.origin !== location.origin) {
      setError(
        "Le moteur d’analyse doit être servi par la même origine locale que le rapport.",
      );
      return;
    }
    url.searchParams.set("start", query.start);
    url.searchParams.set("end", query.end);
    url.searchParams.set("source", query.source);
    query.sleeves.forEach((name) => url.searchParams.append("sleeve", name));
    if (query.frontier) url.searchParams.set("frontier", "1");
    setBusy(true);
    setError(null);
    fetch(url, {
      signal: abort.signal,
      credentials: "same-origin",
      cache: "no-store",
      headers: { Accept: "application/json" },
    })
      .then(async (response) => {
        if (!response.headers.get("content-type")?.includes("application/json"))
          throw new Error(
            "Ce serveur ne fournit pas le moteur d’analyse. Ouvrez le rapport avec le serveur local AlphaRank pour utiliser les dates libres.",
          );
        const result = await response.json();
        if (!response.ok || result.error)
          throw new Error(
            result.error || `Échec du calcul (${response.status}).`,
          );
        return result;
      })
      .then((result) => {
        if (!abort.signal.aborted) setData(result);
      })
      .catch((reason) => {
        if (!abort.signal.aborted)
          setError(
            reason.message ||
              "Le moteur local ne répond pas. Vérifiez que le serveur AlphaRank est démarré.",
          );
      })
      .finally(() => {
        if (!abort.signal.aborted) setBusy(false);
      });
    return () => abort.abort();
  }, [signature, settings]);
  return { data, busy, error };
}

function StrategySelector({
  payload,
  selected,
  onChange,
  mode,
  sleeves,
  setSleeves,
  source,
}) {
  const [search, setSearch] = useState("");
  const available = payload.strategy_order.filter((name) => name !== BENCHMARK);
  const toggle = (name) =>
    onChange(
      selected.includes(name)
        ? selected.filter((item) => item !== name)
        : [...selected, name],
    );
  const composerOrder = payload.portfolio_composer?.strategy_order || available;
  const selectSleeve = (name) => {
    const next = sleeves.includes(name)
      ? sleeves.filter((item) => item !== name)
      : [...sleeves, name];
    if (next.length) setSleeves(next);
  };
  return (
    <aside className="series-sidebar">
      <details id="series-selector" open={window.innerWidth >= 760}>
        <summary>
          {mode === "composer" ? "Mon portefeuille" : "Stratégies"}
          <span>
            {mode === "composer"
              ? `${sleeves.length} poches`
              : `${selected.length} sélectionnées`}
          </span>
        </summary>
        {mode === "composer" ? (
          <div className="sidebar-content">
            <p className="sidebar-help">
              {source === "recent"
                ? "Même poids à l’entrée, puis maintien sans rééquilibrage."
                : "Chaque poche reçoit le même poids, rééquilibré tous les mois."}
            </p>
            <button
              className="pair-button"
              id="composer-boosting-pair"
              onClick={() =>
                setSleeves(
                  ["Boosting · Top 5", "Boosting tendance · Top 5"].filter(
                    (name) => composerOrder.includes(name),
                  ),
                )
              }
            >
              50/50 des deux Boosting
            </button>
            <div className="strategy-options" id="composer-options">
              {composerOrder.map((name) => (
                <label
                  className={`strategy-choice ${sleeves.includes(name) ? "checked" : ""}`}
                  key={name}
                >
                  <input
                    type="checkbox"
                    checked={sleeves.includes(name)}
                    onChange={() => selectSleeve(name)}
                  />
                  <i style={{ background: colorFor(name) }} />
                  <span>{shortName(name)}</span>
                  {sleeves.includes(name) && (
                    <strong>{format(1 / sleeves.length, "pct")}</strong>
                  )}
                </label>
              ))}
            </div>
            <p className="sidebar-help">
              Total : 100 %. Au moins une poche. SPY reste la référence.
            </p>
          </div>
        ) : (
          <div className="sidebar-content">
            <input
              id="strategy-search"
              type="search"
              placeholder="Rechercher une stratégie"
              aria-label="Rechercher une stratégie"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <div className="selection-actions">
              <button
                id="select-all-curves"
                onClick={() => onChange(available)}
              >
                Tout cocher
              </button>
              <button id="select-reference-curves" onClick={() => onChange([])}>
                Tout retirer
              </button>
            </div>
            {["Legacy", "Boosting", "Boosting tendance", "Sectorielles"].map(
              (group) => {
                const choices = available.filter(
                  (name) =>
                    groupFor(name) === group &&
                    name.toLowerCase().includes(search.toLowerCase()),
                );
                return (
                  choices.length > 0 && (
                    <fieldset className="strategy-group" key={group}>
                      <legend>{group}</legend>
                      {choices.map((name) => (
                        <label
                          key={name}
                          className={`strategy-choice ${selected.includes(name) ? "checked" : ""}`}
                          title={name}
                        >
                          <input
                            type="checkbox"
                            checked={selected.includes(name)}
                            onChange={() => toggle(name)}
                            data-strategy={name}
                          />
                          <i style={{ background: colorFor(name) }} />
                          <span>
                            {group === "Legacy"
                              ? name
                                  .replace("Legacy · Frequency", "Fréquence")
                                  .replace("Legacy · Equal", "Équipondéré")
                              : group === "Sectorielles"
                                ? name.includes("tendance")
                                  ? "Tendance · Top 5"
                                  : "Natif · Top 5"
                                : name.split(" · ")[1]}
                          </span>
                        </label>
                      ))}
                    </fieldset>
                  )
                );
              },
            )}
            <div className="fixed-benchmark">
              <i />
              SPY total return
            </div>
          </div>
        )}
      </details>
    </aside>
  );
}

function PeriodControls({
  settings,
  source,
  onSource,
  draft,
  setDraft,
  onApply,
  resolution,
  setResolution,
  busy,
}) {
  const bounds = settings?.sources?.[source];
  return (
    <form
      className="period-bar"
      onSubmit={(event) => {
        event.preventDefault();
        onApply(draft);
      }}
    >
      <label className="source-picker">
        Données
        <select
          id="analysis-source"
          disabled={!settings}
          value={source}
          onChange={(event) => onSource(event.target.value)}
        >
          {Object.entries(
            settings?.sources || { history: { label: "Backtest" } },
          ).map(([value, description]) => (
            <option key={value} value={value}>
              {description.label || value}
            </option>
          ))}
        </select>
      </label>
      <div className="date-pair">
        <label>
          Début
          <input
            id="start-date"
            disabled={!settings}
            type="date"
            required
            min={bounds?.min_date}
            max={bounds?.max_date}
            value={draft.start}
            onChange={(event) =>
              setDraft({ ...draft, start: event.target.value })
            }
          />
        </label>
        <span className="date-separator">à</span>
        <label>
          Fin
          <input
            id="end-date"
            disabled={!settings}
            type="date"
            required
            min={bounds?.min_date}
            max={bounds?.max_date}
            value={draft.end}
            onChange={(event) =>
              setDraft({ ...draft, end: event.target.value })
            }
          />
        </label>
      </div>
      <button
        className="primary-button"
        id="apply-period"
        type="submit"
        disabled={!settings || busy}
      >
        {busy ? "Calcul…" : "Analyser"}
      </button>
      <button
        className="text-action all-period"
        disabled={!settings}
        type="button"
        onClick={() => {
          const next = { start: bounds?.min_date, end: bounds?.max_date };
          setDraft(next);
          onApply(next);
        }}
      >
        Tout l’historique
      </button>
      <div
        className="resolution-control"
        role="group"
        aria-label="Résolution visuelle"
      >
        {["D1", "W1", "M1"].map((value) => (
          <button
            type="button"
            key={value}
            disabled={!settings}
            title={
              value === "D1"
                ? "Chaque séance"
                : value === "W1"
                  ? "Dernier point de chaque semaine"
                  : "Dernier point de chaque mois"
            }
            className={resolution === value ? "active" : ""}
            aria-pressed={resolution === value}
            onClick={() => setResolution(value)}
          >
            {value === "D1" ? "Jour" : value === "W1" ? "Semaine" : "Mois"}
          </button>
        ))}
      </div>
    </form>
  );
}

function CoverageNotice({ data, names }) {
  const incomplete = names.filter(
    (name) => (data.series[name]?.coverage?.missing_closes || 0) > 0,
  );
  if (!incomplete.length) return null;
  return (
    <details className="coverage-warning">
      <summary>
        Clôtures manquantes : certaines mesures quotidiennes sont indisponibles
      </summary>
      <p>
        Les trous sont conservés dans les courbes. Les risques ne sont pas
        calculés sur une série artificiellement complétée.
      </p>
      <ul>
        {incomplete.map((name) => (
          <li key={name}>
            {shortName(name)} : {data.series[name].coverage.missing_closes}{" "}
            clôture(s) manquante(s)
          </li>
        ))}
      </ul>
    </details>
  );
}

function Analysis({ payload, dark }) {
  const settings = payload.analysis;
  const sourceDefault = settings?.default_source || "history";
  const [source, setSource] = useState(sourceDefault),
    [mode, setMode] = useState("compare"),
    [relative, setRelative] = useState(false),
    [resolution, setResolution] = useState("D1");
  const [period, setPeriod] = useState({
    start: settings?.default_start || payload.calendar.start,
    end: settings?.default_end || payload.calendar.end,
  });
  const [draft, setDraft] = useState(period),
    [formError, setFormError] = useState(null),
    [retry, setRetry] = useState(0);
  const [selected, setSelected] = useState(
    DEFAULT_STRATEGIES.filter(
      (name) => name !== BENCHMARK && payload.strategy_order.includes(name),
    ),
  );
  const [sleeves, setSleeves] = useState(
    ["Boosting · Top 5", "Boosting tendance · Top 5"].filter((name) =>
      payload.portfolio_composer?.strategy_order.includes(name),
    ),
  );
  const query = useMemo(
    () => ({
      ...period,
      source,
      sleeves: mode === "composer" ? sleeves : [],
      frontier: mode === "frontier",
      retry,
    }),
    [period, source, sleeves, mode, retry],
  );
  const { data, busy, error } = useAnalysis(payload, query);
  const names = useMemo(
    () =>
      mode === "composer"
        ? [COMPOSED, BENCHMARK]
        : [
            ...payload.strategy_order.filter((name) => selected.includes(name)),
            BENCHMARK,
          ],
    [mode, selected, payload.strategy_order],
  );
  const apply = (next) => {
    if (!next.start || !next.end || next.start > next.end) {
      setFormError("Choisissez un début antérieur ou égal à la fin.");
      return;
    }
    setFormError(null);
    setPeriod(next);
  };
  const changeSource = (value) => {
    setSource(value);
    const bounds = settings.sources[value];
    const next = { start: bounds.min_date, end: bounds.max_date };
    setDraft(next);
    setPeriod(next);
    setFormError(null);
  };
  const analyzeZoom = (range) => {
    const next = { start: range.from, end: range.to };
    setDraft(next);
    apply(next);
  };
  const missing = !settings;
  const ready = data && names.every((name) => data.series?.[name]);
  return (
    <section className="analysis-view">
      <div className="analysis-heading">
        <div>
          <h1>Analyse des performances</h1>
          <p>
            Choisissez les stratégies et les dates. Comparez leur parcours au
            SPY, séance par séance.
          </p>
        </div>
        <span className="data-status">
          {settings ? "Valorisations journalières" : "Archive mensuelle"}
        </span>
      </div>
      <PeriodControls
        settings={settings}
        source={source}
        onSource={changeSource}
        draft={draft}
        setDraft={setDraft}
        onApply={apply}
        resolution={resolution}
        setResolution={setResolution}
        busy={busy}
      />
      <div className="analysis-tabs" role="tablist" aria-label="Mode d’analyse">
        {[
          ["compare", "Comparer"],
          ["composer", "Composer un portefeuille"],
          ["frontier", "Frontières efficientes"],
        ].map(([value, label]) => (
          <button
            role="tab"
            key={value}
            aria-selected={mode === value}
            className={mode === value ? "active" : ""}
            data-analysis-mode={value}
            onClick={() => setMode(value)}
          >
            {label}
          </button>
        ))}
      </div>
      {settings?.sources?.[source]?.warning && (
        <div className="source-notice recent-notice">
          {settings.sources[source].warning}
        </div>
      )}
      {mode === "composer" && (
        <div className="composer-notice">
          Poches équipondérées chaque mois. Frais propres inclus ; aucun coût
          supplémentaire entre poches. Un titre commun à deux poches reste
          exposé dans chacune.
        </div>
      )}
      <div
        className={`analysis-layout ${mode === "frontier" ? "frontier-layout" : ""}`}
      >
        {mode !== "frontier" && (
          <StrategySelector
            payload={payload}
            source={source}
            selected={selected}
            onChange={setSelected}
            mode={mode}
            sleeves={sleeves}
            setSleeves={setSleeves}
          />
        )}
        <div className="analysis-workspace" aria-busy={busy}>
          {missing ? (
            <div className="empty-state">
              <h2>Le journalier n’est pas préparé dans cette archive</h2>
              <p>
                Aucune courbe quotidienne n’est inventée à partir des rendements
                mensuels. Les portefeuilles, les méthodes et les KPI mensuels
                scellés restent disponibles dans les autres vues.
              </p>
            </div>
          ) : formError || error ? (
            <div className="analysis-error" role="alert">
              <h2>Cette fenêtre n’est pas calculée</h2>
              <p>{formError || error}</p>
              {!formError && (
                <button
                  className="secondary-button"
                  onClick={() => setRetry(retry + 1)}
                >
                  Réessayer le calcul
                </button>
              )}
            </div>
          ) : (
            <>
              <div className="window-context">
                <span id="window-label">
                  {data
                    ? `${dateLabel(data.start_date)} – ${dateLabel(data.end_date)}`
                    : "Préparation de la fenêtre…"}
                </span>
                <span>
                  {busy
                    ? "Actualisation du moteur Python…"
                    : data
                      ? `Base 1 à la clôture du ${dateLabel(data.anchor_date)}`
                      : ""}
                </span>
              </div>
              {busy && (
                <div className="calculation-indicator" role="status">
                  Calcul des mesures sur les dates choisies…
                </div>
              )}
              {mode === "frontier" ? (
                <Frontiers data={data} dark={dark} />
              ) : ready ? (
                <div className={busy ? "results calculating" : "results"}>
                  <CoverageNotice data={data} names={names} />
                  <div className="chart-surface">
                    <div
                      className="chart-view-tabs"
                      role="group"
                      aria-label="Valeur du graphique"
                    >
                      <button
                        id="view-wealth"
                        className={!relative ? "active" : ""}
                        aria-pressed={!relative}
                        onClick={() => setRelative(false)}
                      >
                        Capital
                      </button>
                      <button
                        id="view-relative"
                        className={relative ? "active" : ""}
                        aria-pressed={relative}
                        onClick={() => setRelative(true)}
                      >
                        Vs SPY
                      </button>
                    </div>
                    <FinancialCharts
                      data={data}
                      names={names}
                      relative={relative}
                      resolution={resolution}
                      dark={dark}
                      onAnalyzeZoom={analyzeZoom}
                    />
                    <div className="kpi-heading">
                      <h2>La même période, en chiffres</h2>
                      <span>252 séances/an · CAGR en durée réelle</span>
                    </div>
                    <KpiTable data={data} names={names} />
                  </div>
                  <div className="analysis-details">
                    <details className="disclosure" id="full-kpis-drawer">
                      <summary>
                        Tous les indicateurs{" "}
                        <span>
                          Performance, risques, coûts et concentration
                        </span>
                      </summary>
                      <div className="disclosure-body">
                        <KpiTable data={data} names={names} full />
                      </div>
                    </details>
                    <details className="disclosure" id="model-cards-drawer">
                      <summary>
                        Model cards par année{" "}
                        <span>Depuis chaque année et années isolées</span>
                      </summary>
                      <div className="disclosure-body">
                        <ModelCards data={data} names={names} />
                      </div>
                    </details>
                    <details className="disclosure">
                      <summary>
                        Rendements annuels{" "}
                        <span>Écarts au SPY et années partielles</span>
                      </summary>
                      <div className="disclosure-body">
                        <AnnualTable data={data} names={names} />
                      </div>
                    </details>
                    {mode === "composer" && (
                      <details className="disclosure" id="composer-correlation">
                        <summary>Corrélations entre les poches</summary>
                        <div className="disclosure-body">
                          <Correlations correlations={data.correlations} />
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  Préparation des séries sélectionnées…
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function App({ payload }) {
  const [page, setPage] = useState("analyse"),
    [dark, setDark] = useState(initialTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try {
      localStorage.setItem("alpharank-report-theme", dark ? "dark" : "light");
    } catch {
      /* Storage is optional for offline reports. */
    }
  }, [dark]);
  const lastPrice =
    payload.recent_valuation?.latest_close ||
    payload.current_portfolio?.as_of_date;
  return (
    <div className="ar-app" data-theme={dark ? "dark" : "light"}>
      <a className="skip-link" href="#analysis-main">
        Aller à l’analyse
      </a>
      <header className="app-header">
        <div className="brand">
          <span className="brand-symbol">AR</span>
          <div>
            <strong>AlphaRank</strong>
            <span>Performance</span>
          </div>
        </div>
        <nav aria-label="Navigation du rapport">
          {[
            ["analyse", "Analyse"],
            ["portefeuilles", "Portefeuilles"],
            ["methodes", "Méthodes"],
            ["sources", "Sources"],
          ].map(([value, label]) => (
            <button
              key={value}
              className={page === value ? "active" : ""}
              aria-current={page === value ? "page" : undefined}
              onClick={() => setPage(value)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="header-actions">
          <span className="header-freshness">
            Prix au {dateLabel(lastPrice)}
          </span>
          <button
            className="theme-button"
            id="theme-toggle"
            aria-label={
              dark ? "Activer le thème clair" : "Activer le thème sombre"
            }
            aria-pressed={dark}
            onClick={() => setDark(!dark)}
          >
            {dark ? "Clair" : "Sombre"}
            <span aria-hidden="true">{dark ? "☀" : "◐"}</span>
          </button>
        </div>
      </header>
      <main id="analysis-main">
        <div hidden={page !== "analyse"}>
          <Analysis payload={payload} dark={dark} />
        </div>
        <div hidden={page !== "portefeuilles"}>
          <Holdings payload={payload} />
        </div>
        <div hidden={page !== "methodes"}>
          <Methods payload={payload} />
        </div>
        <div hidden={page !== "sources"}>
          <Sources payload={payload} />
        </div>
      </main>
      <footer className="app-footer">
        <span>
          {payload.strategy_order.length} séries ·{" "}
          {format(payload.holdings.length, "int")} positions historiques
        </span>
        <span>
          Recherche sur des paniers archivés, sans ordre ni nouveau signal
        </span>
      </footer>
    </div>
  );
}

globalThis.AlphaRankAnalysis = Object.freeze({
  mount(element, payload) {
    const root = createRoot(element);
    root.render(<App payload={payload} />);
    return root;
  },
  version: "react-19.3.1-lightweight-charts-5.3.0",
});
