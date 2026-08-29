"""Offline HTML rendering for ticker-transition replay evidence."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Mapping, Sequence


def write_ticker_transition_replay_html(
    report: Mapping[str, object],
    output_path: Path,
) -> None:
    """Render an autonomous report without recalculating portfolio metrics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_document(report), encoding="utf-8")


def _document(report: Mapping[str, object]) -> str:
    transition = _mapping(report, "transition")
    target = escape(str(transition["target_ticker"]))
    status = str(report["status"])
    status_class = "passed" if status == "passed" else "failed"
    sections = "".join(
        (
            _hero(report, target, status, status_class),
            _answer(report, target),
            _price_section(report, target),
            _signal_section(report, target),
            _portfolio_section(report, target),
            _proof_section(report),
        )
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRank — replay transition {target}</title>
  <style>{_styles()}</style>
</head>
<body>
  {_sidebar(report, target)}
  <main>{sections}</main>
  <script>{_script()}</script>
</body>
</html>
"""


def _sidebar(report: Mapping[str, object], target: str) -> str:
    snapshot = _mapping(report, "snapshot")
    composition = escape(str(snapshot["composition_id"]))
    return f"""<aside>
  <div class="brand"><span>α</span> AlphaRank</div>
  <div class="eyebrow">PREUVE STATIQUE · TICKER</div>
  <nav>
    <a href="#verdict">Verdict</a><a href="#prices">Prix</a>
    <a href="#signals">Signaux</a><a href="#portfolios">Portefeuilles</a>
    <a href="#proof">Preuves</a>
  </nav>
  <div class="side-note">Sécurité suivie<br><strong>{target}</strong></div>
  <div class="side-note">Snapshot<br><code>{composition[:16]}…</code></div>
</aside>"""


def _hero(
    report: Mapping[str, object],
    target: str,
    status: str,
    status_class: str,
) -> str:
    transition = _mapping(report, "transition")
    prices = _mapping(report, "prices")
    predictions = _mapping(report, "predictions")
    rank = _mapping(predictions, "candidate_causal_rank")
    focus = _mapping(predictions, "candidate_focus")
    cards = (
        ("Séances restaurées", _integer(prices["added_rows"]), "sans valeur manuelle"),
        ("Rang causal", f"#{rank['rank']}", "décision avril 2026"),
        ("Rendement mai", _percent(focus["future_return_1m"]), target),
        ("Scores modifiés", _integer(predictions["changed_common_scores"]), "lignes communes"),
    )
    card_html = "".join(
        f"<article><span>{escape(label)}</span><strong>{value}</strong><small>{escape(note)}</small></article>"
        for label, value, note in cards
    )
    return f"""<header id="verdict">
  <div><div class="eyebrow">REPLAY CAUSAL · {escape(str(transition["focus_holding_month"]))}</div>
  <h1>{target} avait bien un prix en mai — sous ECHO</h1>
  <p>La continuité économique est reconstruite par rendements du ticker fournisseur,
  puis Legacy, Boosting et la variante tendance sont rejoués sur le même snapshot.</p></div>
  <div class="status {status_class}"><i></i>{escape(status.upper())}</div>
</header>
<div class="cards">{card_html}</div>"""


def _answer(report: Mapping[str, object], target: str) -> str:
    predictions = _mapping(report, "predictions")
    baseline = _mapping(predictions, "baseline_focus")
    candidate = _mapping(predictions, "candidate_focus")
    rank = _mapping(predictions, "candidate_causal_rank")
    portfolios = _mapping(report, "portfolios")
    return f"""<section class="answer">
  <div class="answer-index">01</div><div><div class="eyebrow">RÉPONSE DIRECTE</div>
  <h2>Pourquoi Top‑15 et Top‑20 étaient invalides</h2>
  <p><strong>{target} était classé #{rank["rank"]} dans l’univers tendance d’avril.</strong>
  Il entrait donc dans Top‑15 et Top‑20 pour la détention de mai, mais son historique
  s’arrêtait au 24 avril. Le moteur voyait un rendement censuré à
  {_percent(baseline["future_return_1m"])} et refusait, à juste titre, de publier le replay.</p>
  <p>Après reconstruction, le rendement réalisé du 30 avril au 29 mai vaut
  <strong>{_percent(candidate["future_return_1m"])}</strong>, contre
  {_percent(candidate["benchmark_future_return_1m"])} pour SPY. Sa contribution brute est
  {_percent(portfolios["target_ticker_gross_contribution_top15"])} en Top‑15 et
  {_percent(portfolios["target_ticker_gross_contribution_top20"])} en Top‑20.</p>
  <p><strong>La correction ne change aucun score commun et aucun holding Legacy.</strong>
  Elle remplace uniquement une cible de rendement incomplète par le rendement observable ;
  les deux portefeuilles peuvent maintenant être calculés sans inventer un zéro.</p></div>
</section>"""


def _price_section(report: Mapping[str, object], target: str) -> str:
    prices = _mapping(report, "prices")
    rows = (
        {"contrôle": "Dernière séance avant", "résultat": prices["baseline_last_date"]},
        {"contrôle": "Dernière séance après", "résultat": prices["candidate_last_date"]},
        {
            "contrôle": "Plage ajoutée",
            "résultat": f"{prices['added_first_date']} → {prices['added_last_date']}",
        },
        {"contrôle": "Lignes ajoutées", "résultat": prices["added_rows"]},
        {"contrôle": "Lignes antérieures modifiées", "résultat": prices["changed_common_rows"]},
        {
            "contrôle": "Lignes sans rendement source",
            "résultat": prices["unlineaged_overlay_rows"],
        },
    )
    urls = "".join(
        f"<li><code>{escape(str(url))}</code></li>" for url in _sequence(prices, "evidence_urls")
    )
    return f"""<section id="prices">
  {
        _section_head(
            "DONNÉES PRIX",
            "Une continuité de sécurité, pas une fusion de tickers",
            f"{target} conserve son ancien historique ; seuls les rendements ECHO validés prolongent la série.",
        )
    }
  <div class="two-col"><div class="panel"><h3>Contrôles de non-réécriture</h3>{_table(rows)}</div>
  <div class="panel"><h3>Traçabilité</h3><dl>
    <div><dt>Source de rendement</dt><dd>{_joined(prices, "return_source_vintages")}</dd></div>
    <div><dt>Politique</dt><dd>{prices["policy_id"]}</dd></div>
    <div><dt>Formule</dt><dd><code>{prices["derivation_rule"]}</code></dd></div>
    <div><dt>Audit overlay</dt><dd>{_integer(prices["audit_rows"])} lignes</dd></div>
    <div><dt>Période auditée</dt><dd>{prices["audit_first_date"]} → {
        prices["audit_last_date"]
    }</dd></div>
  </dl><h4>Preuve émetteur</h4><ul class="sources">{urls}</ul></div></div>
</section>"""


def _signal_section(report: Mapping[str, object], target: str) -> str:
    predictions = _mapping(report, "predictions")
    baseline = _mapping(predictions, "baseline_focus")
    candidate = _mapping(predictions, "candidate_focus")
    rank = _mapping(predictions, "candidate_causal_rank")
    legacy = _mapping(report, "legacy")
    rows = (
        {
            "version": "avant",
            "score": _decimal(baseline["score"], 6),
            "rendement 1 mois": _percent(baseline["future_return_1m"]),
            "excès vs SPY": _percent(baseline["future_excess_return_1m"]),
            "statut": baseline["target_status_1m"],
        },
        {
            "version": "après",
            "score": _decimal(candidate["score"], 6),
            "rendement 1 mois": _percent(candidate["future_return_1m"]),
            "excès vs SPY": _percent(candidate["future_excess_return_1m"]),
            "statut": candidate["target_status_1m"],
        },
    )
    facts = (
        ("Rang causal", f"#{rank['rank']} sur {rank['eligible_rows']}"),
        ("Tendance positive", _percent(rank["trend_positive_pair_fraction"])),
        ("Nouvelles prédictions", predictions["only_candidate_rows"]),
        ("Scores communs modifiés", predictions["changed_common_scores"]),
        ("Holdings Legacy modifiés", legacy["only_candidate_holdings"]),
        (f"Holdings Legacy {target}", legacy["target_ticker_candidate_holdings"]),
    )
    return f"""<section id="signals">
  {
        _section_head(
            "SIGNAL ET CIBLE",
            "Le modèle ne change pas d’avis",
            "Le score et le rang restent identiques ; seule la vérité du rendement futur devient observable.",
        )
    }
  <div class="panel"><h3>{target} · décision avril 2026</h3>{_table(rows)}</div>
  <div class="fact-grid">{"".join(_fact(label, value) for label, value in facts)}</div>
</section>"""


def _portfolio_section(report: Mapping[str, object], target: str) -> str:
    portfolios = _mapping(report, "portfolios")
    transition = _mapping(report, "transition")
    focus_rows = _mapping_rows(portfolios, "focus_month")
    performance = _mapping_rows(portfolios, "causal_trend_performance")
    performance_rows = [
        {
            "stratégie": row["strategy"],
            "mois": row["months"],
            "CAGR": _percent(row["cagr"]),
            "Sharpe": _decimal(row["sharpe"], 3),
            "drawdown max": _percent(row["max_drawdown"]),
        }
        for row in performance
    ]
    may_rows = [
        {
            "stratégie": row["strategy"],
            "positions": row["n_positions"],
            "brut mai": _percent(row["gross_return"]),
            "coût": _percent(row["transaction_cost"]),
            "net mai": _percent(row["net_return"]),
            "actif vs SPY": _percent(row["active_return"]),
        }
        for row in focus_rows
    ]
    return f"""<section id="portfolios">
  {
        _section_head(
            "MOTEUR COMMUN",
            "Top‑15 et Top‑20 terminent désormais le replay",
            f"{target} est conservé au rang {transition['expected_causal_rank']} et reçoit son rendement réel ; aucun candidat moins bien classé ne le remplace.",
        )
    }
  <div class="panel"><h3>Détention mai 2026</h3>{_table(may_rows)}</div>
  <div class="panel"><div class="table-tools"><h3>Historique commun · 180 mois</h3>
  <input aria-label="Filtrer les stratégies" placeholder="Filtrer…" oninput="filterTable(this, 'perf-table')"></div>
  {_table(performance_rows, table_id="perf-table")}</div>
  <p class="note">Ces résultats sont une preuve de calcul et non une promotion : la variante
  tendance reste un diagnostic de recherche et le pointeur de production n’a pas été déplacé.</p>
</section>"""


def _proof_section(report: Mapping[str, object]) -> str:
    manifests = _mapping(report, "manifests")
    checks = _mapping(report, "checks")
    artifacts = _mapping_rows(report, "artifacts")
    check_rows = [
        {"contrat": key.replace("_", " "), "résultat": "PASS" if value else "FAIL"}
        for key, value in checks.items()
    ]
    artifact_rows = [
        {
            "artefact": str(row["path"]).split("/")[-1],
            "taille": _bytes(row["size_bytes"]),
            "SHA-256": str(row["sha256"]),
        }
        for row in artifacts
    ]
    metadata = escape(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    facts = (
        ("Commit du run", manifests["git_head"]),
        ("Worktree propre", "oui" if manifests["git_dirty"] is False else "non"),
        ("Lignée native", "PASS" if manifests["native_lineage_passed"] else "FAIL"),
        ("Lignée tendance", "PASS" if manifests["trend_lineage_passed"] else "FAIL"),
    )
    return f"""<section id="proof">
  {
        _section_head(
            "PROVENANCE",
            "Chaque conclusion se résout vers un artefact hashé",
            "Le rapport est autonome, sans asset réseau, et ne recalcule aucun KPI.",
        )
    }
  <div class="fact-grid">{"".join(_fact(label, value) for label, value in facts)}</div>
  <div class="two-col"><div class="panel"><h3>Gates</h3>{_table(check_rows)}</div>
  <div class="panel"><h3>Empreintes</h3>{_table(artifact_rows)}</div></div>
  <details><summary>Payload machine-lisible complet</summary><pre>{metadata}</pre></details>
  <p class="footnote">Généré le {escape(str(report["generated_at_utc"]))}.</p>
</section>"""


def _section_head(kicker: str, title: str, subtitle: str) -> str:
    return f"""<div class="section-head"><div><div class="eyebrow">{escape(kicker)}</div>
<h2>{escape(title)}</h2></div><p>{escape(subtitle)}</p></div>"""


def _fact(label: object, value: object) -> str:
    return f"<div class='fact'><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>"


def _table(rows: Sequence[Mapping[str, object]], *, table_id: str = "") -> str:
    if not rows:
        return "<p class='empty'>Aucune ligne.</p>"
    headers = list(rows[0])
    head = "".join(f"<th>{escape(str(column))}</th>" for column in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in headers)
        + "</tr>"
        for row in rows
    )
    identifier = f' id="{escape(table_id)}"' if table_id else ""
    return f"<div class='table-wrap'><table{identifier}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping[key]
    if not isinstance(value, dict):
        raise TypeError(f"Expected object at {key}")
    return value


def _mapping_rows(mapping: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = mapping[key]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Expected rows at {key}")
    return value


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Expected sequence at {key}")
    return value


def _joined(mapping: Mapping[str, object], key: str) -> str:
    return ", ".join(escape(str(value)) for value in _sequence(mapping, key))


def _integer(value: object) -> str:
    return f"{int(str(value)):,}".replace(",", " ")


def _decimal(value: object, digits: int) -> str:
    return f"{float(str(value)):.{digits}f}"


def _percent(value: object) -> str:
    return f"{float(str(value)) * 100:+.2f} %"


def _bytes(value: object) -> str:
    size = float(str(value))
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} Go"


def _script() -> str:
    return """function filterTable(input, tableId) {
  const query = input.value.toLowerCase();
  document.querySelectorAll(`#${tableId} tbody tr`).forEach((row) => {
    row.style.display = row.innerText.toLowerCase().includes(query) ? '' : 'none';
  });
}"""


def _styles() -> str:
    return """
:root{--bg:#F8FAFC;--panel:#FFF;--surface:#F1F5F9;--border:#D7E0EA;--text:#020617;
--muted:#475569;--navy:#111D55;--gold:#9B8816;--green:#265511;--red:#802331;
--blue:#0369A1;--radius:12px}*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:"IBM Plex Sans",Inter,system-ui,sans-serif;
font-size:14px;line-height:1.55}aside{position:fixed;inset:0 auto 0 0;width:248px;padding:28px 24px;
border-right:1px solid var(--border);background:var(--bg)}main{margin-left:248px;max-width:1440px;padding:28px 32px 64px}
.brand{font-weight:700;font-size:20px;margin-bottom:32px}.brand span{display:inline-grid;place-items:center;width:30px;
height:30px;margin-right:8px;background:var(--navy);color:white;border-radius:8px}.eyebrow,code,th,.status,
.fact span,.cards span,.cards small,dt{font-family:"IBM Plex Mono",monospace}.eyebrow{font-size:11px;letter-spacing:.1em;
color:var(--muted);font-weight:600}.side-note{margin-top:24px;padding-top:16px;border-top:1px solid var(--border);
color:var(--muted);font-size:12px}.side-note strong{color:var(--text);font-size:15px}.side-note code{font-size:11px}
nav{display:grid;gap:4px;margin-top:12px}nav a{padding:9px 10px;color:var(--muted);text-decoration:none;border-radius:8px}
nav a:hover{background:var(--surface);color:var(--text)}header{display:flex;justify-content:space-between;gap:28px;align-items:flex-start;
padding-bottom:24px;border-bottom:1px solid var(--border)}h1{font-size:32px;line-height:1.15;margin:8px 0 10px;letter-spacing:-.025em}
header p,.section-head p{max-width:700px;margin:0;color:var(--muted)}.status{display:flex;align-items:center;gap:8px;
padding:8px 12px;border:1px solid var(--border);border-radius:999px;font-size:11px;font-weight:600}.status i{width:8px;height:8px;
border-radius:50%}.status.passed i{background:var(--green)}.status.failed i{background:var(--red)}.cards{display:grid;
grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 40px}.cards article,.panel,.fact{background:var(--panel);
border:1px solid var(--border);border-radius:var(--radius);box-shadow:0 1px 2px rgba(2,6,23,.04)}.cards article{padding:16px}
.cards span,.cards small{display:block;color:var(--muted);font-size:10px}.cards strong{display:block;font-size:24px;margin:7px 0 4px}
section{margin:42px 0}.answer{display:grid;grid-template-columns:52px minmax(0,1fr);gap:20px;padding:24px;
background:var(--navy);color:white;border-radius:var(--radius)}.answer-index{font-family:"IBM Plex Mono",monospace;color:#CBD5E1}
.answer h2{margin:4px 0 12px}.answer p{color:#E2E8F0;max-width:970px}.answer .eyebrow{color:#94A3B8}.section-head{display:flex;
justify-content:space-between;gap:32px;align-items:end;margin-bottom:16px}.section-head h2{font-size:22px;margin:5px 0 0}
.section-head p{max-width:510px;text-align:right}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{padding:18px;
margin-bottom:14px;overflow:hidden}.panel h3{margin:0 0 14px;font-size:15px}.panel h4{margin:18px 0 8px}.table-wrap{overflow:auto}
table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:var(--muted);font-size:10px;letter-spacing:.04em;
padding:9px;border-bottom:1px solid var(--border);white-space:nowrap}td{padding:10px 9px;border-bottom:1px solid #E8EDF3;vertical-align:top}
td:last-child{font-family:"IBM Plex Mono",monospace}tbody tr:hover{background:var(--surface)}dl{margin:0}dl div{display:flex;
justify-content:space-between;gap:20px;padding:9px 0;border-bottom:1px solid var(--border)}dt{font-size:10px;color:var(--muted)}dd{margin:0;
text-align:right}.sources{padding-left:18px;overflow-wrap:anywhere}.sources code{font-size:10px}.fact-grid{display:grid;grid-template-columns:repeat(4,
minmax(0,1fr));gap:12px;margin:14px 0}.fact{padding:14px}.fact span{display:block;color:var(--muted);font-size:10px;
margin-bottom:6px}.fact strong{font-size:14px;overflow-wrap:anywhere}.table-tools{display:flex;justify-content:space-between;gap:16px;
align-items:center}.table-tools input{border:1px solid var(--border);border-radius:8px;padding:8px 10px;background:var(--surface);
color:var(--text)}.note,.footnote,.empty{color:var(--muted);font-size:12px}.note{border-left:3px solid var(--gold);padding-left:12px}
details{border:1px solid var(--border);border-radius:var(--radius);background:var(--panel);margin-top:14px}summary{cursor:pointer;
padding:14px;font-weight:600}pre{margin:0;padding:16px;max-height:520px;overflow:auto;background:#0B1220;color:#E2E8F0;
font:11px/1.5 "IBM Plex Mono",monospace;border-radius:0 0 var(--radius) var(--radius)}
@media(max-width:1000px){aside{position:static;width:auto;border-right:0;border-bottom:1px solid var(--border)}aside nav{display:flex;
flex-wrap:wrap}.side-note{display:none}main{margin-left:0;padding:22px}.cards,.fact-grid{grid-template-columns:repeat(2,1fr)}.two-col{grid-template-columns:1fr}}
@media(max-width:620px){header,.section-head{display:block}.status{margin-top:14px;width:max-content}.cards,.fact-grid{grid-template-columns:1fr}
.answer{grid-template-columns:1fr}.section-head p{text-align:left;margin-top:8px}h1{font-size:27px}}
"""
