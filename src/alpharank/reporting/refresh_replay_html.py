"""Self-contained HTML rendering for refresh replay evidence."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Mapping, Sequence


def write_refresh_replay_html(report: Mapping[str, object], output_path: Path) -> None:
    """Render one offline audit report without recalculating model results."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_document(report), encoding="utf-8")


def _render_document(report: Mapping[str, object]) -> str:
    headline = _mapping(report, "headline")
    focus = _mapping(report, "focus")
    provenance = _mapping(report, "provenance")
    status = str(report["status"])
    gate = escape(str(report.get("gate_failure") or "Aucune gate bloquante"))
    verdict_class = "blocked" if not bool(report["promotion_allowed"]) else "passed"
    content = "".join(
        (
            _hero(headline, status, verdict_class),
            _answer_section(headline, focus),
            _causes_section(report),
            _focus_section(report, focus, gate),
            _legacy_section(report),
            _boosting_section(report),
            _data_section(report),
            _proof_section(report, provenance),
        )
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRank — rapport de drift data</title>
  <style>{_styles()}</style>
</head>
<body>
  {_sidebar(report)}
  <main>{content}</main>
  <script>{_script()}</script>
</body>
</html>
"""


def _sidebar(report: Mapping[str, object]) -> str:
    return f"""<aside>
  <div class="brand"><span>α</span> AlphaRank</div>
  <div class="eyebrow">PREUVE STATIQUE · REFRESH</div>
  <nav><a href="#verdict">Verdict</a><a href="#causes">Causes</a>
  <a href="#cvc">CVC.US</a><a href="#legacy">Legacy</a>
  <a href="#boosting">Boosting</a><a href="#data">Données</a>
  <a href="#proof">Preuves</a></nav>
  <div class="side-note">Cutoff historique<br>
  <strong>{escape(str(report["historical_cutoff"]))}</strong></div>
</aside>"""


def _hero(
    headline: Mapping[str, object],
    status: str,
    verdict_class: str,
) -> str:
    return f"""<header id="verdict"><div><div class="eyebrow">
DATA REFRESH · ANALYSE CAUSALE</div><h1>Ce qui a vraiment changé</h1>
<p class="lede">CVC.US déclenche l'arrêt final. Il n'est pas la cause des milliers
de différences : celles-ci viennent d'un réentraînement global alimenté par le
refresh SEC, avec un effet prix distinct et mesuré.</p></div>
<div class="status {verdict_class}"><span></span>{escape(status)}</div></header>
{_summary_cards(headline)}"""


def _answer_section(headline: Mapping[str, object], focus: Mapping[str, object]) -> str:
    return f"""<section class="answer"><div class="answer-mark">01</div><div>
<h2>Réponse directe</h2><p><strong>Le drift Legacy est quasi entièrement SEC :</strong>
SEC seuls reproduit {_integer(headline["legacy_sec_only_events"])} événements de position
et il ne reste que {_integer(headline["legacy_sec_to_full_events"])} événements entre SEC
seuls et le candidat complet. Prix seuls produit
{_integer(headline["legacy_price_only_events"])} petits changements, détaillés plus bas.</p>
<p><strong>Les {_integer(headline["boosting_full_changed_common"])} lignes Boosting sont des
scores ticker-mois, pas des positions.</strong> SEC seuls modifie
{_integer(headline["boosting_sec_score_changed"])} scores ; prix seuls en modifie
{_integer(headline["boosting_price_score_changed"])}. Après SEC, ajouter les prix en change
encore {_integer(headline["boosting_sec_to_full_score_changed"])}.</p>
<p><strong>CVC est spécifiquement SEC-driven :</strong> ses prix sont identiques, mais ses
fondamentaux passent de 0 à {_integer(_mapping(focus, "sec")["candidate_rows"])} observations
et son rang passe de 35 à 8.</p></div></section>"""


def _causes_section(report: Mapping[str, object]) -> str:
    return f"""<section id="causes"><div class="section-head"><div><div class="eyebrow">
ATTRIBUTION PAR ABLATION</div><h2>Quatre runs, une causalité visible</h2></div>
<p>Chaque scénario change une seule famille de données, avec le même code, la même
configuration et le même runtime.</p></div>{_causal_chain()}{_scenario_table(report)}</section>"""


def _focus_section(
    report: Mapping[str, object],
    focus: Mapping[str, object],
    gate: str,
) -> str:
    return f"""<section id="cvc"><div class="section-head"><div><div class="eyebrow">
CAS BLOQUANT</div><h2>CVC.US : SEC change le signal, pas le prix</h2></div>
<p>Décision juin 2016, détention juillet 2016.</p></div>
<div class="two-col"><div class="panel">{_focus_score_table(focus)}</div>
<div class="panel">{_focus_facts(focus)}</div></div>
<div class="panel fold-proof">{_feature_fold(report)}</div>
<div class="gate"><strong>Gate exacte</strong><code>{gate}</code></div></section>"""


def _legacy_section(report: Mapping[str, object]) -> str:
    return f"""<section id="legacy"><div class="section-head"><div><div class="eyebrow">
SIGNAL LEGACY</div><h2>Le drift Legacy vient quasi entièrement du SEC</h2></div>
<p>Ajouts, retraits et poids modifiés sont séparés.</p></div>
<div class="two-col"><div class="panel">{_legacy_comparison_table(report)}</div>
<div class="panel"><h3>Événements par année</h3>{_legacy_chart(report)}</div></div>
<div class="panel"><h3>Effet prix résiduel après SEC</h3>
{_table(_rows(report, "legacy_sec_to_full_events"))}</div>
<div class="panel"><h3>Tickers les plus affectés</h3>
{_searchable_table(_rows(report, "legacy_top_tickers"), "legacy-tickers")}</div></section>"""


def _boosting_section(report: Mapping[str, object]) -> str:
    return f"""<section id="boosting"><div class="section-head"><div><div class="eyebrow">
SIGNAL BOOSTING</div><h2>Beaucoup de scores, beaucoup moins de holdings</h2></div>
<p>Le Top-N est affiché avant la gate commune lorsque celle-ci refuse de publier.</p></div>
{_prediction_table(report)}<div class="two-col"><div class="panel">
<h3>Amplitude des écarts de score</h3>
{_bar_chart(_rows(report, "score_histogram"), "label", "rows")}</div>
<div class="panel"><h3>Holdings communs publiables</h3>{_common_table(report)}</div></div>
<div class="panel"><h3>Mois avec le plus de rotation du Top 10 signal</h3>
{_top10_chart(report)}</div><div class="panel"><h3>Plus grands déplacements de score</h3>
{_searchable_table(_rows(report, "top_score_movers"), "score-movers")}</div></section>"""


def _data_section(report: Mapping[str, object]) -> str:
    return f"""<section id="data"><div class="section-head"><div><div class="eyebrow">
SNAPSHOT</div><h2>Prix, SEC et univers ne sont plus confondus</h2></div>
<p>Comparaison au cutoff commun ; les nouvelles dates restent séparées du passé.</p></div>
{_snapshot_table(report)}</section>"""


def _proof_section(
    report: Mapping[str, object],
    provenance: Mapping[str, object],
) -> str:
    metadata = escape(json.dumps(provenance, indent=2, ensure_ascii=False))
    return f"""<section id="proof"><div class="section-head"><div><div class="eyebrow">
PROVENANCE</div><h2>Pourquoi cette conclusion est fiable</h2></div></div>
{_proof_cards(provenance)}<div class="panel"><h3>Empreintes des quatre scénarios</h3>
{_table(_rows(provenance, "scenario_artifacts"))}</div>
<details><summary>Empreinte et métadonnées</summary><pre>{metadata}</pre></details>
<p class="footnote">Rapport généré le {escape(str(report["generated_at_utc"]))}. Il est
autonome, sans ressource réseau, et ne déplace aucun pointeur de production.</p></section>"""


def _summary_cards(headline: Mapping[str, object]) -> str:
    cards = (
        ("Legacy", headline["legacy_full_changed_common"], "poids communs modifiés", "amber"),
        (
            "Boosting",
            headline["boosting_full_changed_common"],
            "scores ticker-mois modifiés",
            "blue",
        ),
        (
            "Prix seuls",
            headline["boosting_price_score_changed"],
            "scores Boosting modifiés",
            "slate",
        ),
        ("SEC seuls", headline["boosting_sec_score_changed"], "scores Boosting modifiés", "red"),
    )
    return (
        '<div class="cards">'
        + "".join(
            f'<article class="metric {color}"><span>{escape(label)}</span>'
            f"<strong>{_integer(value)}</strong><small>{escape(note)}</small></article>"
            for label, value, note, color in cards
        )
        + "</div>"
    )


def _causal_chain() -> str:
    nodes = (
        ("SEC", "révisions fondamentales"),
        ("Legacy", "gagnants et poids"),
        ("Features", "couples EMA retenus"),
        ("Boosting", "réentraînement global"),
        ("CVC", "rang 35 → 8"),
        ("Gate", "publication refusée"),
    )
    return (
        '<div class="chain">'
        + "<i>→</i>".join(
            f"<div><strong>{title}</strong><span>{detail}</span></div>" for title, detail in nodes
        )
        + "</div>"
    )


def _scenario_table(report: Mapping[str, object]) -> str:
    statuses = {str(row["scenario"]): row for row in _rows(report, "scenario_statuses")}
    legacy = {str(row["scenario"]): row for row in _rows(report, "legacy_comparisons")}
    boosting = {str(row["scenario"]): row for row in _rows(report, "prediction_comparisons")}
    focus = _mapping(report, "focus")
    scores = {str(row["scenario"]): row for row in _rows(focus, "scores")}
    rows = []
    for name in ("baseline", "price_only", "sec_only", "full"):
        row = statuses[name]
        if name == "baseline":
            legacy_events = "référence"
            score_changes = "référence"
        else:
            legacy_events = _integer(legacy[name]["total_position_events"])
            score_changes = _integer(_mapping(boosting[name], "score")["changed_rows"])
        rows.append(
            {
                "scénario": row["label"],
                "événements Legacy": legacy_events,
                "scores Boosting modifiés": score_changes,
                "rang CVC": scores[name]["rank"],
                "score CVC": _decimal(scores[name]["score"], 8),
                "replay commun": row["common_status"],
            }
        )
    return _table(rows)


def _focus_score_table(focus: Mapping[str, object]) -> str:
    rows = []
    for row in _rows(focus, "scores"):
        rows.append(
            {
                "scénario": row["scenario"],
                "score": _decimal(row["score"], 8),
                "rang": row["rank"],
                "commun": row["common_status"],
            }
        )
    return "<h3>Score et rang</h3>" + _table(rows)


def _focus_facts(focus: Mapping[str, object]) -> str:
    price = _mapping(focus, "price")
    sec = _mapping(focus, "sec")
    facts = (
        ("Prix communs modifiés", price["changed_common_rows"]),
        ("Lignes prix", price["baseline_rows"]),
        ("Période prix", f"{price['first_date']} → {price['last_date']}"),
        ("Fondamentaux baseline", sec["baseline_rows"]),
        ("Fondamentaux candidat", sec["candidate_rows"]),
        ("Dernier dépôt", sec["latest_filing_date"]),
    )
    return (
        "<h3>Preuve data CVC</h3><dl>"
        + "".join(
            f"<div><dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd></div>"
            for label, value in facts
        )
        + "</dl>"
        + _table(_rows(sec, "tables"))
    )


def _feature_fold(report: Mapping[str, object]) -> str:
    fold = _mapping(report, "feature_fold")
    baseline_rows = _mapping(fold, "baseline_rows")
    candidate_rows = _mapping(fold, "candidate_rows")
    rows = [
        {
            "fold": fold["fold"],
            "couples EMA baseline": fold["baseline_pair_count"],
            "couples EMA candidat": fold["candidate_pair_count"],
            "couples conservés": fold["retained_pair_count"],
            "train baseline / candidat": (
                f"{_integer(baseline_rows['train'])} / {_integer(candidate_rows['train'])}"
            ),
            "validation baseline / candidat": (
                f"{_integer(baseline_rows['validation'])} / "
                f"{_integer(candidate_rows['validation'])}"
            ),
            "test baseline / candidat": (
                f"{_integer(baseline_rows['test'])} / {_integer(candidate_rows['test'])}"
            ),
        }
    ]
    return "<h3>Pourquoi Boosting change sans lire directement le SEC</h3>" + _table(rows)


def _legacy_comparison_table(report: Mapping[str, object]) -> str:
    rows = []
    for row in _rows(report, "legacy_comparisons"):
        rows.append(
            {
                "scénario": row["scenario"],
                "lignes": row["candidate_rows"],
                "ajouts": row["added_rows"],
                "retraits": row["removed_rows"],
                "poids modifiés": row["changed_common_rows"],
                "événements": row["total_position_events"],
            }
        )
    return "<h3>Ablation des familles data</h3>" + _table(rows)


def _legacy_chart(report: Mapping[str, object]) -> str:
    rows = _rows(report, "legacy_timeline")
    totals: dict[str, int] = {}
    for row in rows:
        year = str(row["year"])
        totals[year] = totals.get(year, 0) + _number_as_int(row["rows"])
    chart_rows = [{"year": year, "events": value} for year, value in sorted(totals.items())]
    return _bar_chart(chart_rows, "year", "events")


def _prediction_table(report: Mapping[str, object]) -> str:
    rows = []
    for row in _rows(report, "prediction_comparisons"):
        score = _mapping(row, "score")
        top_10 = _mapping(row, "raw_signal_top_10")
        rows.append(
            {
                "scénario": row["scenario"],
                "communes": row["common_rows"],
                "toute colonne modifiée": row["any_changed_rows"],
                "scores modifiés": score["changed_rows"],
                "% scores": _percent(score["changed_share"]),
                "Top 10 entrées": top_10["entries"],
                "Top 10 sorties": top_10["exits"],
            }
        )
    return _table(rows)


def _common_table(report: Mapping[str, object]) -> str:
    rows = []
    for row in _rows(report, "common_portfolios"):
        values = {"scénario": row["scenario"], "statut": row["status"]}
        if "added_rows" in row:
            values.update(
                {
                    "entrées": row["added_rows"],
                    "sorties": row["removed_rows"],
                    "poids modifiés": row["changed_common_rows"],
                }
            )
        else:
            values.update({"entrées": "—", "sorties": "—", "poids modifiés": "—"})
        rows.append(values)
    return _table(rows)


def _top10_chart(report: Mapping[str, object]) -> str:
    totals: dict[str, int] = {}
    for row in _rows(report, "top10_timeline"):
        month = str(row["decision_month"])
        totals[month] = totals.get(month, 0) + _number_as_int(row["rows"])
    selected = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:18]
    rows = [{"month": month[:7], "events": events} for month, events in selected]
    return _bar_chart(rows, "month", "events")


def _snapshot_table(report: Mapping[str, object]) -> str:
    rows = []
    for row in _rows(report, "snapshot_tables"):
        rows.append(
            {
                "famille": row["family"],
                "table": row["table"],
                "baseline": row["baseline_rows"],
                "candidat": row["candidate_rows"],
                "ajouts": row["added_rows"],
                "retraits": row["removed_rows"],
                "communes modifiées": row["changed_common_rows"],
            }
        )
    return _table(rows)


def _proof_cards(provenance: Mapping[str, object]) -> str:
    checks = (
        ("Code", provenance["all_code_identical"]),
        ("Configuration", provenance["all_config_identical"]),
        ("Runtime", provenance["all_runtime_identical"]),
    )
    return (
        '<div class="proof-grid">'
        + "".join(
            f'<div><span class="check">✓</span><strong>{escape(name)}</strong>'
            f"<small>{'identique' if bool(value) else 'différent'}</small></div>"
            for name, value in checks
        )
        + "</div>"
    )


def _bar_chart(
    rows: Sequence[Mapping[str, object]],
    label_key: str,
    value_key: str,
) -> str:
    maximum = max((_number_as_int(row[value_key]) for row in rows), default=1) or 1
    parts = ['<div class="bars">']
    for row in rows:
        value = _number_as_int(row[value_key])
        width = max(1.5, value / maximum * 100) if value else 0
        parts.append(
            f'<div class="bar-row"><span>{escape(str(row[label_key]))}</span>'
            f'<i><b style="width:{width:.2f}%"></b></i><strong>{_integer(value)}</strong></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _searchable_table(rows: Sequence[Mapping[str, object]], table_id: str) -> str:
    return (
        f'<input class="search" type="search" placeholder="Filtrer…" '
        f"oninput=\"filterTable('{table_id}', this.value)\">" + _table(rows, table_id)
    )


def _table(rows: Sequence[Mapping[str, object]], table_id: str | None = None) -> str:
    if not rows:
        return '<p class="empty">Aucune ligne.</p>'
    columns = list(rows[0])
    identifier = f' id="{escape(table_id)}"' if table_id else ""
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{_display(row.get(column))}</td>" for column in columns)
            + "</tr>"
        )
    return f'<div class="table-wrap"><table{identifier}><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _display(value: object) -> str:
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, int):
        return _integer(value)
    if isinstance(value, float):
        return _decimal(value, 6)
    return escape(str(value if value is not None else "—"))


def _integer(value: object) -> str:
    return f"{_number_as_int(value):,}".replace(",", " ")


def _decimal(value: object, digits: int) -> str:
    return f"{_number_as_float(value):.{digits}f}".replace(".", ",")


def _percent(value: object) -> str:
    return f"{_number_as_float(value) * 100:.1f} %".replace(".", ",")


def _number_as_int(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected integer-compatible value, got {type(value).__name__}")
    return int(value)


def _number_as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected float-compatible value, got {type(value).__name__}")
    return float(value)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected mapping at {key}")
    return nested


def _rows(value: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    nested = value.get(key)
    if not isinstance(nested, list) or not all(isinstance(row, dict) for row in nested):
        raise ValueError(f"Expected table rows at {key}")
    return nested


def _styles() -> str:
    return """
:root{--ink:#17222d;--muted:#66727f;--line:#dce2e7;--paper:#f4f2ec;--card:#fff;
--red:#c8473a;--amber:#d8932f;--blue:#286fa3;--slate:#607888;--green:#398463}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);
font:14px/1.55 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:228px;background:#15232d;color:#fff;padding:28px 24px;z-index:3}
.brand{font-size:20px;font-weight:750;letter-spacing:-.03em}.brand span{display:inline-grid;place-items:center;
width:34px;height:34px;border:1px solid #55707f;border-radius:50%;margin-right:9px;color:#f0b85a}
.eyebrow{font-size:10px;letter-spacing:.18em;font-weight:800;color:#75828e;text-transform:uppercase}
aside .eyebrow{margin:34px 0 14px;color:#8197a5}nav{display:grid;gap:2px}nav a{padding:9px 10px;
border-radius:6px;color:#b9c6ce;text-decoration:none}nav a:hover{background:#213743;color:#fff}
.side-note{position:absolute;bottom:28px;color:#8396a2}.side-note strong{color:white;font-size:16px}
main{margin-left:228px;padding:54px clamp(30px,5vw,76px) 90px;max-width:1540px}header{display:flex;
justify-content:space-between;gap:40px;align-items:flex-start;border-bottom:1px solid #cfd5d9;padding-bottom:34px}
h1{font-family:Georgia,serif;font-size:clamp(42px,5vw,72px);line-height:.98;letter-spacing:-.045em;
margin:10px 0 18px;max-width:800px}h2{font:700 32px/1.1 Georgia,serif;letter-spacing:-.025em;margin:6px 0}
h3{font-size:14px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 16px}.lede{font-size:17px;
max-width:780px;color:#53616c}.status{display:flex;gap:9px;align-items:center;border:1px solid var(--line);background:#fff;
border-radius:999px;padding:9px 14px;font:700 11px/1 monospace;text-transform:uppercase;white-space:nowrap}
.status span{width:8px;height:8px;border-radius:50%;background:var(--green)}.status.blocked span{background:var(--red)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0 52px}.metric{background:var(--card);
border:1px solid var(--line);border-top:3px solid var(--slate);padding:20px;border-radius:4px;box-shadow:0 8px 25px #1c2c3610}
.metric.amber{border-top-color:var(--amber)}.metric.blue{border-top-color:var(--blue)}.metric.red{border-top-color:var(--red)}
.metric span,.metric small{display:block;color:var(--muted)}.metric strong{display:block;font:700 31px/1.1 Georgia,serif;margin:8px 0}
section{margin:66px 0}.answer{display:grid;grid-template-columns:74px 1fr;gap:22px;background:#fffdf7;border:1px solid #e3dac5;
padding:30px 34px;margin-top:0}.answer-mark{font:700 44px/1 Georgia,serif;color:#d4a34d}.answer p{font-size:16px;max-width:1000px}
.section-head{display:flex;justify-content:space-between;gap:30px;align-items:end;margin-bottom:24px}.section-head>p{color:var(--muted);max-width:440px}
.chain{display:flex;align-items:stretch;gap:8px;margin:20px 0 26px}.chain div{flex:1;background:#fff;border:1px solid var(--line);
padding:14px 12px;border-radius:4px;min-width:105px}.chain strong,.chain span{display:block}.chain span{font-size:11px;color:var(--muted);margin-top:4px}
.chain i{align-self:center;color:#9aa5ad}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#fff;border:1px solid var(--line);padding:22px;border-radius:4px;overflow:hidden}
section>.panel{margin-top:16px}
.table-wrap{overflow:auto;border:1px solid var(--line);background:#fff}table{width:100%;border-collapse:collapse;font-size:12px}
th{position:sticky;top:0;background:#edf0f2;color:#4e5b65;text-align:left;font-size:10px;letter-spacing:.07em;text-transform:uppercase}
th,td{padding:10px 12px;border-bottom:1px solid #e7eaed;white-space:nowrap}tbody tr:hover{background:#faf8f2}
dl{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);margin:0 0 18px}
dl div{background:white;padding:11px}dt{font-size:10px;color:var(--muted);text-transform:uppercase}dd{margin:3px 0 0;font-weight:700}
.gate{margin-top:16px;background:#2a2020;color:#f7e6e3;padding:18px 22px;border-left:4px solid var(--red)}.gate strong{display:block;margin-bottom:8px}.gate code{white-space:normal;font-size:11px;color:#e8c4be}
.bars{display:grid;gap:7px}.bar-row{display:grid;grid-template-columns:80px 1fr 65px;gap:9px;align-items:center;font-size:11px}
.bar-row i{height:9px;background:#e8ecef;border-radius:8px;overflow:hidden}.bar-row b{display:block;height:100%;background:linear-gradient(90deg,#286fa3,#69a8c7);border-radius:8px}.bar-row strong{text-align:right}
.search{width:260px;max-width:100%;border:1px solid var(--line);padding:9px 11px;margin-bottom:10px;background:#fafafa}
.proof-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.proof-grid>div{background:#fff;border:1px solid var(--line);padding:20px}.proof-grid strong,.proof-grid small{display:block}.proof-grid small{color:var(--muted)}.check{float:right;color:var(--green);font-size:20px}
details{margin-top:16px;border:1px solid var(--line);background:#fff;padding:14px}summary{cursor:pointer;font-weight:700}pre{white-space:pre-wrap;font-size:11px;color:#53616c}.footnote,.empty{color:var(--muted);font-size:12px}
@media(max-width:1000px){aside{position:static;width:auto}.side-note{display:none}nav{display:flex;overflow:auto}main{margin:0;padding:35px 22px}.cards{grid-template-columns:1fr 1fr}.chain{overflow:auto}.chain div{min-width:160px}}
@media(max-width:650px){header,.section-head{display:block}.status{margin-top:20px;width:max-content}.cards,.two-col,.proof-grid{grid-template-columns:1fr}.answer{grid-template-columns:1fr}.answer-mark{font-size:25px}}
@media print{aside{display:none}main{margin:0;padding:20px}.status,.metric,.panel,.answer{box-shadow:none}section{break-inside:avoid}}
"""


def _script() -> str:
    return """
function filterTable(id, query) {
  const needle = query.toLocaleLowerCase('fr');
  document.querySelectorAll(`#${id} tbody tr`).forEach((row) => {
    row.hidden = !row.textContent.toLocaleLowerCase('fr').includes(needle);
  });
}
"""
