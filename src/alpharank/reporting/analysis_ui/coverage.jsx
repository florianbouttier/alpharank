import React from "react";
import { dateLabel, format, shortName } from "./format.js";

export function CoverageNotice({ data, names, onMonthlyFallback, busy }) {
  if (!data) return null;
  const monthly = data.frequency === "monthly";
  const selected = new Set(names);
  const gaps = (data.price_gaps || []).filter((row) =>
    selected.has(row.strategy),
  );
  const incomplete = names.filter(
    (name) => (data.series?.[name]?.coverage?.missing_closes || 0) > 0,
  );
  if (!monthly && !gaps.length && !incomplete.length) return null;
  const fallback = data.monthly_fallback;
  return (
    <section className="coverage-panel" aria-label="Diagnostic de couverture">
      <div className="coverage-heading">
        <div>
          <h2>
            {monthly
              ? "Référence mensuelle active"
              : "Pourquoi des mesures sont absentes"}
          </h2>
          <p>
            {monthly
              ? "Rendements mensuels scellés, 12 observations/an. Ce repli ne crée aucun prix quotidien et ne corrige pas les anciennes hypothèses de valorisation terminale."
              : "Une clôture ajustée absente laisse un trou : le risque du parcours quotidien n’est pas observable. Aucune valeur de sortie ou de CVR n’est supposée."}
          </p>
        </div>
        {!monthly && fallback?.available && fallback.start && fallback.end && (
          <button
            className="secondary-button fallback-button"
            id="apply-monthly-fallback"
            disabled={busy}
            onClick={() => onMonthlyFallback(fallback)}
          >
            Utiliser la référence mensuelle
            <span>
              {dateLabel(fallback.start)} – {dateLabel(fallback.end)}
              {fallback.period_changed
                ? " · bornes modifiées"
                : " · mêmes bornes"}
            </span>
          </button>
        )}
      </div>
      {!monthly && fallback?.available && fallback.period_changed && (
        <p className="fallback-period-note">
          Seuls les mois calendaires complets inclus sont proposés. Les dates
          affichées dans le bouton seront appliquées uniquement à votre clic.
        </p>
      )}
      {!monthly && fallback && !fallback.available && (
        <p className="fallback-period-note">
          Aucun mois complet dans cette fenêtre. Élargissez explicitement les
          dates pour disposer d’une référence mensuelle.
        </p>
      )}
      <details className="coverage-warning" id="price-gap-audit">
        <summary>
          {gaps.length
            ? `${gaps.length} ligne(s) titre / poche à examiner${monthly ? " dans le journalier strict" : " dans la sélection"}`
            : "Détail de la couverture"}
        </summary>
        {incomplete.length > 0 && (
          <ul>
            {incomplete.map((name) => (
              <li key={name}>
                {shortName(name)} : {data.series[name].coverage.missing_closes}{" "}
                clôture(s) manquante(s).
              </li>
            ))}
          </ul>
        )}
        {gaps.length > 0 ? (
          <div className="table-scroll gap-table-scroll">
            <table className="gap-table">
              <thead>
                <tr>
                  <th>Titre</th>
                  <th>Poche</th>
                  <th>Mois détenu</th>
                  <th>Séances sans prix</th>
                  <th>Poids cible</th>
                  <th>Dates exactes</th>
                </tr>
              </thead>
              <tbody>
                {gaps.map((row, index) => (
                  <tr
                    key={`${row.strategy}-${row.ticker}-${row.holding_month}-${index}`}
                  >
                    <th>{row.ticker}</th>
                    <td>{shortName(row.strategy)}</td>
                    <td>
                      {dateLabel(row.holding_month, {
                        day: undefined,
                        month: "long",
                      })}
                    </td>
                    <td>
                      {row.missing_sessions}
                      <small>
                        {dateLabel(row.first_missing_session)} –{" "}
                        {dateLabel(row.last_missing_session)}
                      </small>
                    </td>
                    <td>{format(row.target_weight, "pct")}</td>
                    <td>
                      <details>
                        <summary>Voir les séances</summary>
                        <p>
                          {row.dates
                            ?.map((date) => dateLabel(date))
                            .join(", ") || "Dates non détaillées"}
                        </p>
                      </details>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>
            {data.price_gap_audit_available
              ? "Aucune séance sans prix identifiée pour les poches affichées dans cette fenêtre."
              : "Le détail titre / séance n’est pas préparé dans ce rapport ; son absence n’établit pas une couverture complète."}
          </p>
        )}
        <p className="fine-print">
          Les trous concernent les clôtures ajustées de la source scellée, pas
          un changement de sélection. La référence mensuelle utilise les
          rendements déjà archivés ; elle ne résout pas à elle seule les
          hypothèses de sortie des titres inactifs.
        </p>
      </details>
    </section>
  );
}
