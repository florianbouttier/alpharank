import React, { useMemo, useState } from "react";
import {
  BENCHMARK,
  METRICS,
  dateLabel,
  downloadCsv,
  format,
  shortName,
} from "./format.js";

const monthlyLabel = (field) =>
  (METRICS[field]?.[0] || field)
    .replace("quotidienne", "mensuelle")
    .replace("Séances", "Mois");

export function Holdings({ payload }) {
  const [current, setCurrent] = useState(true),
    [strategy, setStrategy] = useState(
      payload.strategy_order.find((name) => name === "Boosting · Top 5") ||
        payload.strategy_order[0],
    );
  const [month, setMonth] = useState(payload.calendar.end),
    [query, setQuery] = useState(""),
    [page, setPage] = useState(0);
  const source = current
    ? payload.current_portfolio?.holdings || []
    : payload.holdings || [];
  const rows = useMemo(
    () =>
      source
        .filter(
          (row) =>
            row.strategy === strategy &&
            (current || row.holding_month === month) &&
            row.ticker.toLowerCase().includes(query.trim().toLowerCase()),
        )
        .sort(
          (a, b) =>
            (a.selection_rank ?? 999) - (b.selection_rank ?? 999) ||
            a.ticker.localeCompare(b.ticker),
        ),
    [source, strategy, current, month, query],
  );
  const update = (setter, value) => {
    setter(value);
    setPage(0);
  };
  return (
    <section className="audit-view">
      <header className="section-heading">
        <div>
          <h1>Les portefeuilles</h1>
          <p>
            Positions et poids archivés, sans nouveau classement ni ordre
            envoyé.
          </p>
        </div>
      </header>
      <div className="audit-controls">
        <div className="inline-tabs">
          <button
            className={current ? "active" : ""}
            onClick={() => update(setCurrent, true)}
          >
            Dernier panier archivé
          </button>
          <button
            className={!current ? "active" : ""}
            onClick={() => update(setCurrent, false)}
          >
            Historique complet
          </button>
        </div>
        <label>
          Stratégie
          <select
            id="portfolio-strategy"
            value={strategy}
            onChange={(e) => update(setStrategy, e.target.value)}
          >
            {payload.strategy_order
              .filter((name) => name !== BENCHMARK)
              .map((name) => (
                <option key={name}>{name}</option>
              ))}
          </select>
        </label>
        {!current && (
          <label>
            Mois détenu
            <select
              id="portfolio-month"
              value={month}
              onChange={(e) => update(setMonth, e.target.value)}
            >
              {payload.calendar.available_months.map((value) => (
                <option key={value} value={value}>
                  {dateLabel(value, { day: undefined, month: "long" })}
                </option>
              ))}
            </select>
          </label>
        )}
        <label>
          Rechercher un titre
          <input
            id="ticker-search"
            type="search"
            placeholder="SATS, DELL…"
            value={query}
            onChange={(e) => update(setQuery, e.target.value)}
          />
        </label>
        <button
          className="secondary-button"
          id="export-holdings"
          onClick={() => downloadCsv(rows, "alpharank-positions.csv")}
        >
          Exporter CSV
        </button>
      </div>
      {current && (
        <div className="source-notice">
          <strong>
            Panier prévu pour{" "}
            {dateLabel(payload.current_portfolio.holding_month, {
              day: undefined,
              month: "long",
            })}
          </strong>
          <span>
            Décision : fin{" "}
            {dateLabel(payload.current_portfolio.decision_month, {
              day: undefined,
            })}
            . Les prix plus récents ne créent pas un nouveau signal.{" "}
            {payload.recent_valuation &&
              `Suivi prix jusqu’au ${dateLabel(payload.recent_valuation.latest_close)} ; maintien au-delà du mois prévu hypothétique.`}
          </span>
        </div>
      )}
      <div className="holdings-summary">
        {rows.length} position{rows.length > 1 ? "s" : ""} · {strategy}
      </div>
      <div className="table-scroll">
        <table id="holdings-table">
          <thead>
            <tr>
              <th>Titre</th>
              <th>Rang</th>
              <th>Poids cible</th>
              <th>Score</th>
              <th>Secteur</th>
              <th>Rendement réalisé</th>
              <th>Votes</th>
              <th>Décision</th>
              <th>Détention</th>
            </tr>
          </thead>
          <tbody id="holdings-body">
            {rows.slice(page * 20, (page + 1) * 20).map((row) => (
              <tr key={row.ticker}>
                <th>{row.ticker}</th>
                <td>{format(row.selection_rank, "int")}</td>
                <td>{format(row.target_weight, "pct")}</td>
                <td
                  title={
                    row.score == null ? "Score indisponible" : String(row.score)
                  }
                >
                  {format(row.score, "score")}
                </td>
                <td>{row.sector || "Non renseigné"}</td>
                <td>{format(row.realized_return, "pct")}</td>
                <td>{format(row.n_models, "int")}</td>
                <td>{dateLabel(row.decision_month, { day: undefined })}</td>
                <td>{dateLabel(row.holding_month, { day: undefined })}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!rows.length && (
        <div className="empty-state">
          Aucune position ne correspond à ces filtres.
        </div>
      )}
      <div className="pagination">
        <button disabled={!page} onClick={() => setPage(page - 1)}>
          Précédent
        </button>
        <span>
          Page {page + 1} / {Math.max(1, Math.ceil(rows.length / 20))}
        </span>
        <button
          disabled={(page + 1) * 20 >= rows.length}
          onClick={() => setPage(page + 1)}
        >
          Suivant
        </button>
      </div>
    </section>
  );
}

export function Methods({ payload }) {
  const estimated =
    payload.contracts?.daily_valuation?.policy === "last_observed_mark_v1";
  const methods = payload.recent_valuation?.methodology
    ? [...payload.methodologies, payload.recent_valuation.methodology]
    : payload.methodologies;
  return (
    <section className="audit-view">
      <header className="section-heading">
        <div>
          <h1>Méthodes et conventions</h1>
          <p>
            Règles, pseudo-codes et limites de chaque famille. Le statut R&amp;D
            reste distinct de la production.
          </p>
        </div>
      </header>
      <div className="methods-list" id="method-grid">
        {methods.map((method, index) => (
          <article key={index}>
            <header>
              <h2>{method.title}</h2>
              <span>{method.status}</span>
            </header>
            <p>{method.summary}</p>
            <ol>
              {method.pseudo_code?.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ol>
          </article>
        ))}
      </div>
      <article className="daily-method">
        <h2>Lecture journalière</h2>
        <p>
          Les positions mensuelles restent figées. Le moteur AlphaRank les
          valorise à chaque clôture ajustée disponible, laisse dériver les poids
          et conserve les coûts archivés. Une date libre sélectionne les
          rendements inclus, avec un repère à la clôture précédente. Les KPI
          journaliers sont annualisés sur 252 séances ; le CAGR utilise la durée
          calendaire réelle.
        </p>
        <p>
          {estimated ? (
            "Cette version active le scénario de dernier cours conservé décrit ci-dessous. Le mode strict reste une autre version du même run ; les positions, poids et frais n’ont pas changé."
          ) : (
            <>
              Un prix absent reste un trou. La volatilité et le drawdown ne sont
              pas présentés comme complets quand la fenêtre ne l’est pas. Les
              chiffres mensuels scellés sont conservés séparément dans les
              sources.
            </>
          )}
        </p>
      </article>
      {estimated && (
        <article className="daily-method" id="last-mark-method">
          <h2>Repli quotidien : dernier cours connu conservé</h2>
          <p>
            Scénario de valorisation estimé, pas une vente au dernier cours ni
            une correction des prix sources. Les risques peuvent être
            sous-estimés, notamment après une faillite. Cash de rachat, titre
            successeur et droits conditionnels non cotés restent à traiter avec
            leurs contreparties sourcées.
          </p>
          <ol>
            <li>
              Lire les titres, poids et frais mensuels scellés, sans
              réentraînement ni nouveau classement.
            </li>
            <li>
              Partir du cours ajusté observé à l’ancre du mois ; une ancre ou un
              benchmark absent reste une erreur.
            </li>
            <li>
              À chaque séance, utiliser le cours observé ; s’il manque,
              conserver le dernier cours déjà connu de ce seul titre dans le
              mois. Aucun cours futur n’est lu.
            </li>
            <li>
              Garder les quantités investies et les autres titres en mouvement.
              Un cours figé a un facteur brut de 1, soit un rendement simple nul
              ; une reprise de cotation conserve tout le rattrapage.
            </li>
            <li>
              Enregistrer chaque séance estimée avec la date et la valeur du
              cours conservé. Recalculer le capital et les risques à 252
              séances/an, puis composer les poches avec ce même capital.
            </li>
            <li>
              Rapprocher chaque rendement mensuel de l’archive ; ne jamais
              forcer une parité en rééchelonnant les courbes. Les chiffres
              mensuels archivés restent séparés dans Sources.
            </li>
          </ol>
          <p>
            Cette hypothèse ne donne pas la valeur récupérable d’une action
            SIVB, ni celle du CVR d’ABMD. Le sélecteur de version permet de
            revenir au journalier strict. Les années incomplètes ou non
            calculables ne deviennent pas artificiellement des années gagnantes.
          </p>
        </article>
      )}
      <article className="daily-method" id="monthly-fallback-method">
        <h2>Référence mensuelle scellée : repli volontaire</h2>
        <p>
          La référence mensuelle ne comble aucun cours quotidien. Le bouton de
          repli propose seulement les mois calendaires complets inclus dans la
          fenêtre ; les nouvelles bornes sont affichées avant application.
        </p>
        <ol>
          <li>
            Lire les rendements mensuels déjà archivés, sans nouveau signal ni
            modification des positions.
          </li>
          <li>
            Comparer, composer et tracer les frontières sur ces mêmes mois, avec
            les frais propres et le rééquilibrage mensuel de l’historique.
          </li>
          <li>
            Mesurer les risques sur 12 observations par an, et le drawdown aux
            fins de mois ; aucun creux intramensuel n’est reconstitué.
          </li>
          <li>
            Conserver le détail des prix quotidiens absents. Ce repli ne corrige
            pas les anciennes hypothèses de valorisation terminale des titres
            inactifs, ni la valeur d’un éventuel CVR.
          </li>
        </ol>
        <p>
          Le suivi récent reste distinct : panier figé, brut de frais, avec
          pondération à l’entrée et sans nouveau rééquilibrage. Une référence
          mensuelle de recherche n’est pas une réparation de la donnée
          quotidienne ni une promotion de stratégie.
        </p>
      </article>
    </section>
  );
}

export function MonthlyReference({ payload }) {
  const [kind, setKind] = useState("strategies"),
    [start, setStart] = useState(payload.calendar.start),
    [end, setEnd] = useState(payload.calendar.end),
    [field, setField] = useState("cagr"),
    [page, setPage] = useState(0);
  const key = `${start}|${end}`;
  const dataset =
    kind === "composer"
      ? payload.portfolio_composer
      : kind === "frontier"
        ? payload.strategy_frontiers
        : payload;
  const raw =
    kind === "frontier"
      ? dataset.windows?.[key]?.candidate_metrics
      : dataset.metric_windows?.[key];
  const fields = dataset.metric_fields || [];
  const labels =
    kind === "composer"
      ? dataset.combination_masks.map((mask) =>
          dataset.strategy_order
            .filter((_, i) => mask & (2 ** i))
            .map(shortName)
            .join(" + "),
        )
      : kind === "frontier"
        ? dataset.weights.map((weights) =>
            weights
              .map(
                (weight, i) =>
                  `${shortName(dataset.strategy_order[i])} ${format(weight, "pct")}`,
              )
              .join(" · "),
          )
        : payload.strategy_order;
  const shownFields =
    field === "all" ? fields : fields.filter((value) => value === field);
  const rows = (raw || []).map((values, index) => ({
    label: labels[index],
    values,
  }));
  const change = (setter, value) => {
    setter(value);
    setPage(0);
  };
  return (
    <details className="disclosure monthly-reference" id="monthly-reference">
      <summary>
        Référence mensuelle archivée{" "}
        <span>
          {payload.metric_fields?.length || 0} KPI,{" "}
          {payload.portfolio_composer?.combination_masks?.length || 0}{" "}
          compositions et {payload.strategy_frontiers?.weights?.length || 0}{" "}
          allocations conservés
        </span>
      </summary>
      <div className="disclosure-body">
        <p className="fine-print">
          Cette archive conserve les chiffres publiés et leur annualisation
          mensuelle. Ses bornes annuelles pré-calculées sont indépendantes de
          l’analyse journalière à dates libres.
        </p>
        <div className="audit-controls">
          <label>
            Contenu
            <select
              value={kind}
              onChange={(e) => change(setKind, e.target.value)}
            >
              <option value="strategies">Stratégies</option>
              <option value="composer">Compositions équipondérées</option>
              {payload.strategy_frontiers && (
                <option value="frontier">Allocations des frontières</option>
              )}
            </select>
          </label>
          <label>
            Début archivé
            <select
              value={start}
              onChange={(e) => change(setStart, e.target.value)}
            >
              {payload.calendar.available_start_months.map((value) => (
                <option key={value} value={value}>
                  {dateLabel(value, { day: undefined })}
                </option>
              ))}
            </select>
          </label>
          <label>
            Fin archivée
            <select
              value={end}
              onChange={(e) => change(setEnd, e.target.value)}
            >
              {payload.calendar.available_end_months.map((value) => (
                <option key={value} value={value}>
                  {dateLabel(value, { day: undefined })}
                </option>
              ))}
            </select>
          </label>
          <label>
            Indicateurs
            <select
              value={field}
              onChange={(e) => change(setField, e.target.value)}
            >
              <option value="all">Tous les indicateurs</option>
              {fields.map((value) => (
                <option key={value} value={value}>
                  {monthlyLabel(value)}
                </option>
              ))}
            </select>
          </label>
        </div>
        {!rows.length ? (
          <div className="empty-state">
            Aucune fenêtre archivée pour ces bornes.
          </div>
        ) : (
          <>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>
                      {kind === "strategies" ? "Stratégie" : "Composition"}
                    </th>
                    {shownFields.map((name) => (
                      <th key={name}>{monthlyLabel(name)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(page * 20, (page + 1) * 20).map((row, i) => (
                    <tr key={i}>
                      <th>{row.label}</th>
                      {shownFields.map((name) => (
                        <td key={name}>
                          {format(
                            row.values[fields.indexOf(name)],
                            METRICS[name]?.[1],
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <button disabled={!page} onClick={() => setPage(page - 1)}>
                Précédent
              </button>
              <span>
                {rows.length} séries ou compositions · page {page + 1}/
                {Math.ceil(rows.length / 20)}
              </span>
              <button
                disabled={(page + 1) * 20 >= rows.length}
                onClick={() => setPage(page + 1)}
              >
                Suivant
              </button>
            </div>
          </>
        )}
      </div>
    </details>
  );
}

export function Sources({ payload }) {
  const download = () => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
      url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = "alpharank-rapport-complet.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <section className="audit-view">
      <header className="section-heading">
        <div>
          <h1>Sources et contrôle des données</h1>
          <p>
            Le rapport lit des artefacts nommés et hashés ; il ne relance ni
            acquisition ni modèle.
          </p>
        </div>
        <button className="secondary-button" onClick={download}>
          Exporter les données du rapport
        </button>
      </header>
      <div className="source-notice">
        <strong>
          {payload.status?.publication_eligible
            ? "Publication éligible"
            : "Recherche, non promue"}
        </strong>
        <span>{payload.status?.message}</span>
      </div>
      <dl className="source-summary">
        <div>
          <dt>Historique mensuel</dt>
          <dd>
            {dateLabel(payload.calendar.start, { day: undefined })} –{" "}
            {dateLabel(payload.calendar.end, { day: undefined })}
          </dd>
        </div>
        <div>
          <dt>Stratégies</dt>
          <dd>{payload.strategy_order.length}, SPY inclus</dd>
        </div>
        <div>
          <dt>Positions historiques</dt>
          <dd>{format(payload.holdings.length, "int")}</dd>
        </div>
        <div>
          <dt>Composition du snapshot</dt>
          <dd>
            <code>{payload.lineage.composition_id}</code>
          </dd>
        </div>
      </dl>
      <MonthlyReference payload={payload} />
      <details className="disclosure">
        <summary>Conventions et couverture</summary>
        <div className="disclosure-body">
          <pre>
            {JSON.stringify(
              {
                contracts: payload.contracts,
                data_quality: payload.data_quality,
                daily_analysis:
                  payload.analysis?.provenance || payload.analysis?.coverage,
              },
              null,
              2,
            )}
          </pre>
        </div>
      </details>
      <div className="table-scroll">
        <table className="source-table">
          <thead>
            <tr>
              <th>Rôle</th>
              <th>Artefact</th>
              <th>SHA-256</th>
            </tr>
          </thead>
          <tbody>
            {payload.lineage.source_files.map((source, index) => (
              <tr key={index}>
                <th>{source.role}</th>
                <td>
                  <code>{source.path}</code>
                </td>
                <td>
                  <code>{source.sha256}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
