import React from "react";
import { dateLabel } from "./format.js";

export function PeriodControls({
  settings,
  source,
  onSource,
  draft,
  setDraft,
  period,
  onApply,
  resolution,
  setResolution,
  busy,
}) {
  const bounds = settings?.sources?.[source];
  const monthly = source === "monthly_reference";
  const pending = draft.start !== period.start || draft.end !== period.end;
  return (
    <form
      className="period-bar"
      onSubmit={(event) => {
        event.preventDefault();
        onApply(draft);
      }}
    >
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
        Toute la période
      </button>
      {monthly ? (
        <div className="monthly-resolution">
          <span>Mensuel · 12 observations/an</span>
          <button
            className="text-action"
            type="button"
            id="return-daily"
            onClick={() => onSource("history")}
          >
            Revenir au journalier
          </button>
        </div>
      ) : (
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
      )}
      {monthly && (
        <p className="period-requirement">
          Mois calendaires complets uniquement : début le premier jour, fin le
          dernier jour.
        </p>
      )}
      {pending && (
        <p className="draft-notice" id="unapplied-dates" role="status">
          Dates modifiées, non appliquées. Les résultats restent sur la fenêtre
          du {dateLabel(period.start)} au {dateLabel(period.end)}. Cliquez sur «
          Analyser ».
        </p>
      )}
    </form>
  );
}
