import React, { useEffect, useState } from "react";
import {
  dateLabel,
  format,
  metricDefinition,
  METRICS,
  shortName,
} from "./format.js";

export function versionLabel(version) {
  const label = version?.replay_at_utc
    ? `Replay du ${dateLabel(version.replay_at_utc.slice(0, 10))} · ${version.strategies.length} séries`
    : `Version ${version?.id?.slice(-8) || "non renseignée"}`;
  return `${label} · ${version?.valuation_policy === "last_observed_mark_v1" ? "Repli quotidien (estimé)" : "Strict"}`;
}

export function versionHref(context, identity, href) {
  if (!context.entries.some((entry) => entry.id === identity))
    throw new Error("Version absente du catalogue capturé.");
  const current = new URL(href);
  let target;
  if (current.protocol === "file:") {
    const root = new URL(
      context.current_version_id === context.default_version_id
        ? "./"
        : "../../",
      current,
    );
    target = new URL(
      identity === context.default_version_id
        ? "backtest_performance_report.html"
        : `versions/${identity}/backtest_performance_report.html`,
      root,
    );
  } else {
    target = new URL("/backtest_performance_report.html", current);
    target.searchParams.set("version", identity);
  }
  target.hash = current.hash;
  return target.href;
}

export function VersionBar({
  payload,
  onVersion,
  onCompare,
  onFollowup,
  followup,
}) {
  const context = payload.analysis?.versions;
  const entries = context?.entries || [];
  const current = entries.find(
    (entry) => entry.id === context.current_version_id,
  );
  const decisionMonth = payload.current_portfolio?.decision_month;
  return (
    <section className="version-bar" aria-label="Version du backtest">
      <div className="version-selection">
        <label>
          Version du backtest
          <select
            id="backtest-version"
            value={context?.current_version_id || ""}
            disabled={!entries.length}
            onChange={(event) => onVersion(event.target.value)}
          >
            {!entries.length && <option value="">Version non préparée</option>}
            {entries.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {versionLabel(entry)}
              </option>
            ))}
          </select>
        </label>
        <div className="version-coverage" id="version-coverage">
          <strong>
            {current?.last_close ||
            payload.analysis?.sources?.history?.last_close
              ? `Historique calculé jusqu’au ${dateLabel(current?.last_close || payload.analysis.sources.history.last_close)}`
              : `Historique mensuel jusqu’à ${dateLabel(payload.calendar?.end, { day: undefined })}`}
          </strong>
          <span>
            Backtest de recherche
            {current?.composition_id && (
              <>
                {" "}
                · snapshot <code>{current.composition_id.slice(0, 12)}</code>
              </>
            )}
            {current?.price_as_of_date && (
              <> · cours du snapshot au {dateLabel(current.price_as_of_date)}</>
            )}
          </span>
        </div>
        <button
          id="compare-versions-open"
          className="text-action"
          onClick={onCompare}
        >
          Détails / comparer
        </button>
      </div>
      {payload.recent_valuation && (
        <details className="version-followup" open={followup}>
          <summary>
            Panier figé · mois détenu{" "}
            {dateLabel(payload.current_portfolio.holding_month, {
              day: undefined,
              month: "long",
            })}
            {decisionMonth && (
              <>
                {" "}
                · décision en{" "}
                {dateLabel(decisionMonth, { day: undefined, month: "long" })}
              </>
            )}{" "}
            · valorisé au {dateLabel(payload.recent_valuation.latest_close)}
          </summary>
          <div className="version-followup-body">
            <p>Aucun rééquilibrage après ce mois n’est inclus.</p>
            <button
              id="toggle-frozen-followup"
              className="text-action"
              aria-pressed={!!followup}
              onClick={onFollowup}
            >
              {followup
                ? "Revenir au backtest"
                : "Voir ce panier conservé · hors backtest"}
            </button>
          </div>
        </details>
      )}
    </section>
  );
}

export function VersionFacts({ version }) {
  if (!version) return null;
  const bps = version.cost_policy?.bps_times_turnover;
  return (
    <article className="version-facts">
      <h2>{versionLabel(version)}</h2>
      <dl>
        <div>
          <dt>Historique mesuré</dt>
          <dd>
            {dateLabel(version.start_date)} – {dateLabel(version.last_close)}
          </dd>
        </div>
        <div>
          <dt>Snapshot composé le</dt>
          <dd>{dateLabel(version.snapshot_at_utc?.slice(0, 10))}</dd>
        </div>
        <div>
          <dt>Prix du snapshot jusqu’au</dt>
          <dd>{dateLabel(version.price_as_of_date)}</dd>
        </div>
        <div>
          <dt>Frais de référence</dt>
          <dd>
            {bps == null
              ? "Voir les conventions archivées"
              : `${format(bps, "int")} pb × turnover`}
          </dd>
        </div>
        <div>
          <dt>Frais archivés nuls</dt>
          <dd>
            {version.zero_cost_strategies?.length
              ? version.zero_cost_strategies.map(shortName).join(", ")
              : "Aucune stratégie hors SPY"}
          </dd>
        </div>
        <div>
          <dt>Code du replay central</dt>
          <dd>
            <code>
              {version.replay_git_commit?.slice(0, 12) || "Non renseigné"}
            </code>
          </dd>
        </div>
        <div>
          <dt>Snapshot</dt>
          <dd>
            <code>{version.composition_id.slice(0, 16)}</code>
          </dd>
        </div>
        <div>
          <dt>Valorisation quotidienne</dt>
          <dd>
            {version.valuation_policy === "last_observed_mark_v1"
              ? "Dernier cours conservé · scénario estimé, pas de nouveau signal"
              : "Stricte · clôtures absentes conservées comme trous"}
          </dd>
        </div>
        <div>
          <dt>Statut</dt>
          <dd>Recherche non promue</dd>
        </div>
      </dl>
    </article>
  );
}

export function comparisonMatches(result, query) {
  return (
    !!result &&
    result.left_version_id === query.left &&
    result.right_version_id === query.right &&
    result.requested_start === query.start &&
    result.requested_end === query.end &&
    result.frequency === query.frequency
  );
}

function useVersionComparison(query, enabled, attempt) {
  const [result, setResult] = useState(null),
    [error, setError] = useState(null),
    [busy, setBusy] = useState(false);
  const signature = JSON.stringify(query);
  useEffect(() => {
    if (!enabled) return;
    setResult(null);
    setError(null);
    if (location.protocol === "file:") {
      setError(
        "La comparaison à dates libres nécessite le serveur local AlphaRank. Les versions et leurs archives restent consultables séparément.",
      );
      return;
    }
    const abort = new AbortController();
    const url = new URL("/api/version-comparison", location.href);
    Object.entries(query).forEach(([key, value]) =>
      url.searchParams.set(key, value),
    );
    setBusy(true);
    fetch(url, {
      signal: abort.signal,
      credentials: "same-origin",
      cache: "no-store",
      headers: { Accept: "application/json" },
    })
      .then(async (response) => {
        if (!response.headers.get("content-type")?.includes("application/json"))
          throw new Error(
            "Le serveur local de comparaison n’est pas disponible.",
          );
        const data = await response.json();
        if (!response.ok || data.error)
          throw new Error(data.error || "Comparaison indisponible.");
        if (!comparisonMatches(data, query))
          throw new Error(
            "La réponse ne correspond pas aux versions, dates et fréquence demandées.",
          );
        return data;
      })
      .then((data) => {
        if (!abort.signal.aborted) setResult(data);
      })
      .catch((reason) => {
        if (!abort.signal.aborted) setError(reason.message);
      })
      .finally(() => {
        if (!abort.signal.aborted) setBusy(false);
      });
    return () => abort.abort();
  }, [signature, enabled, attempt]);
  const ready = comparisonMatches(result, query);
  return {
    result: ready ? result : null,
    error,
    busy: busy || (enabled && !error && !ready),
  };
}

export function VersionMetrics({ result, field, left, right }) {
  const definition = metricDefinition(field, result.frequency);
  return (
    <>
      <div className="window-context">
        <span>
          {dateLabel(result.start_date)} – {dateLabel(result.end_date)}
        </span>
        <span>
          {result.frequency === "monthly"
            ? "Fins de mois · 12 observations/an"
            : "Séances · 252 observations/an"}
        </span>
      </div>
      <p className="fine-print">{definition[2]}</p>
      <div className="table-scroll">
        <table id="version-metrics">
          <thead>
            <tr>
              <th>{definition[0]}</th>
              <th>{versionLabel(left)}</th>
              <th>{versionLabel(right)}</th>
            </tr>
          </thead>
          <tbody>
            {result.metrics.map((row) => (
              <tr key={row.strategy}>
                <th>{row.strategy}</th>
                <td>{format(row.left[field], definition[1])}</td>
                <td>{format(row.right[field], definition[1])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!!(
        result.left_only_strategies.length ||
        result.right_only_strategies.length
      ) && (
        <p className="version-warning">
          Hors comparaison, car absentes de l’autre version :{" "}
          {[
            ...result.left_only_strategies,
            ...result.right_only_strategies,
          ].join(", ")}
          .
        </p>
      )}
      <details className="disclosure" id="version-holdings-diff">
        <summary>
          Titres et poids historiques{" "}
          <span>
            {format(result.holdings.changed_positions, "int")} lignes
            différentes dans {format(result.holdings.changed_baskets, "int")}{" "}
            paniers
          </span>
        </summary>
        <div className="disclosure-body">
          <p className="fine-print">
            Clés stratégie × mois × titre et poids cibles des seules stratégies
            communes, pour les mois touchés par la fenêtre. Un changement de
            titre compte une sortie et une entrée. Ni scores ni causalité ne
            sont certifiés par ce rapprochement.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Mois détenu</th>
                  <th>Stratégie</th>
                  <th>Lignes différentes</th>
                </tr>
              </thead>
              <tbody>
                {result.holdings.by_month.map((row) => (
                  <tr key={`${row.strategy}:${row.holding_month}`}>
                    <th>{dateLabel(row.holding_month, { day: undefined })}</th>
                    <td>{row.strategy}</td>
                    <td>{format(row.changed_positions, "int")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!result.holdings.by_month.length && (
            <p>Aucune différence de titres ou poids sur ces mois.</p>
          )}
        </div>
      </details>
    </>
  );
}

export function VersionComparison({ payload }) {
  const context = payload.analysis?.versions;
  const entries = context?.entries || [];
  const current = entries.find(
    (entry) => entry.id === context.current_version_id,
  );
  const initial = {
    left: current?.id || "",
    right:
      entries.find((entry) => entry.id !== current?.id)?.id ||
      current?.id ||
      "",
    start: payload.analysis?.default_start || "",
    end: payload.analysis?.default_end || "",
    frequency: "daily",
  };
  const [draft, setDraft] = useState(initial),
    [applied, setApplied] = useState(initial),
    [field, setField] = useState("cagr");
  const [formError, setFormError] = useState(null),
    [attempt, setAttempt] = useState(0);
  const { result, error, busy } = useVersionComparison(
    applied,
    entries.length > 1,
    attempt,
  );
  const left = entries.find((entry) => entry.id === applied.left),
    right = entries.find((entry) => entry.id === applied.right);
  const pending = JSON.stringify(draft) !== JSON.stringify(applied);
  const update = (key, value) => setDraft({ ...draft, [key]: value });
  return (
    <section className="audit-view version-comparison" id="version-comparison">
      <header className="section-heading">
        <div>
          <h1>Versions du backtest</h1>
          <p>
            Un run cohérent, ses résultats et ses limites. La date du HTML n’est
            pas celle du calcul modèle.
          </p>
        </div>
      </header>
      <details
        className="disclosure version-provenance"
        id="version-provenance"
      >
        <summary>
          Sources et conventions des versions{" "}
          <span>Dates, frais, code et snapshots</span>
        </summary>
        <div className="disclosure-body version-pair">
          <VersionFacts version={left || current} />
          <VersionFacts version={right} />
        </div>
      </details>
      {entries.length > 1 ? (
        <>
          <form
            className="audit-controls version-comparison-controls"
            onSubmit={(event) => {
              event.preventDefault();
              if (!draft.start || !draft.end || draft.start > draft.end) {
                setFormError("Choisissez un début antérieur ou égal à la fin.");
                return;
              }
              setFormError(null);
              setApplied({ ...draft });
              setAttempt((value) => value + 1);
            }}
          >
            {[
              ["left", "Version A"],
              ["right", "Version B"],
            ].map(([key, label]) => (
              <label key={key}>
                {label}
                <select
                  id={`compare-version-${key}`}
                  value={draft[key]}
                  onChange={(event) => update(key, event.target.value)}
                >
                  {entries.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {versionLabel(entry)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            <label>
              Début
              <input
                id="version-start"
                type="date"
                required
                value={draft.start}
                onChange={(event) => update("start", event.target.value)}
              />
            </label>
            <label>
              Fin
              <input
                id="version-end"
                type="date"
                required
                value={draft.end}
                onChange={(event) => update("end", event.target.value)}
              />
            </label>
            <label>
              Observations du risque
              <select
                id="version-frequency"
                value={draft.frequency}
                onChange={(event) => update("frequency", event.target.value)}
              >
                <option value="daily">Quotidiennes</option>
                <option value="monthly">Fins de mois</option>
              </select>
            </label>
            <button
              className="primary-button"
              id="apply-version-comparison"
              disabled={busy}
              type="submit"
            >
              {busy ? "Calcul…" : "Comparer ces versions"}
            </button>
          </form>
          {pending && (
            <p className="draft-notice" role="status">
              Choix modifiés, non appliqués. Les résultats et fiches restent
              ceux des versions précédemment comparées.
            </p>
          )}
          {draft.frequency === "monthly" && (
            <p className="period-requirement">
              Fins de mois : mois calendaires complets uniquement, sans
              recadrage automatique.
            </p>
          )}
          {formError || error ? (
            <div className="analysis-error" role="alert">
              <h2>Cette comparaison n’est pas calculée</h2>
              <p>{formError || error}</p>
            </div>
          ) : busy ? (
            <p className="calculation-indicator" role="status">
              Calcul des deux versions aux mêmes dates…
            </p>
          ) : (
            result && (
              <>
                <div className="version-warning">
                  <strong>
                    Comparaison descriptive, pas un audit de fuite.
                  </strong>
                  <p>
                    {result.snapshot_changed
                      ? "Snapshots différents. "
                      : "Même snapshot. "}
                    {result.code_changed
                      ? "Codes de replay différents. "
                      : "Même code de replay déclaré. "}
                    {!!result.changed_contract_fields.length &&
                      "Conventions différentes : " +
                        result.changed_contract_fields
                          .map(
                            (key) =>
                              ({
                                transaction_cost: "frais",
                                sector_release: "historique sectoriel",
                                sector_taxonomy: "classification des secteurs",
                                timing: "dates de décision et détention",
                                benchmark: "benchmark",
                                missing_return: "rendements manquants",
                                kpi_engine: "moteur de mesure",
                                daily_valuation:
                                  "repli de valorisation quotidienne",
                              })[key] || key.replaceAll("_", " "),
                          )
                          .join(", ") +
                        "."}
                  </p>
                  {(!!left.zero_cost_strategies?.length ||
                    !!right.zero_cost_strategies?.length) && (
                    <p>
                      Frais archivés nuls — version A :{" "}
                      {left.zero_cost_strategies.length
                        ? left.zero_cost_strategies.map(shortName).join(", ")
                        : "aucune stratégie hors SPY"}{" "}
                      ; version B :{" "}
                      {right.zero_cost_strategies.length
                        ? right.zero_cost_strategies.map(shortName).join(", ")
                        : "aucune stratégie hors SPY"}
                      .
                    </p>
                  )}
                  <details>
                    <summary>Limites du rapprochement</summary>
                    <p>{result.warning}</p>
                  </details>
                </div>
                <label className="version-metric-picker">
                  Indicateur comparé
                  <select
                    id="version-kpi"
                    value={field}
                    onChange={(event) => setField(event.target.value)}
                  >
                    {Object.keys(METRICS)
                      .filter((name) =>
                        result.metrics.some(
                          (row) =>
                            Object.hasOwn(row.left, name) ||
                            Object.hasOwn(row.right, name),
                        ),
                      )
                      .map((name) => (
                        <option key={name} value={name}>
                          {metricDefinition(name, result.frequency)[0]}
                        </option>
                      ))}
                  </select>
                </label>
                <VersionMetrics
                  result={result}
                  field={field}
                  left={left}
                  right={right}
                />
              </>
            )
          )}
        </>
      ) : (
        <p className="empty-state">
          Une seule version est capturée dans ce rapport. Aucun ancien run n’est
          déduit d’un dossier ou inventé.
        </p>
      )}
    </section>
  );
}
