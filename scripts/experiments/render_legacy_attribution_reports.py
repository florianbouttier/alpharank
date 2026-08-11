#!/usr/bin/env python3
"""Render standalone HTML reports for the audited Legacy attribution."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTRIBUTION_DIR = PROJECT_ROOT / "outputs/legacy_attribution_20260809"
DEFAULT_OUTPUT_DIR = DEFAULT_ATTRIBUTION_DIR / "html"
DEFAULT_PRE_FIX_RUN = PROJECT_ROOT / "outputs/2026-07-13/runs/20260713_201639"
DEFAULT_POST_FIX_RUN = PROJECT_ROOT / "outputs/2026-07-19/runs/20260719_194418"
DEFAULT_PRE_SELECTIONS = (
    PROJECT_ROOT
    / "outputs/checkpoints_open_source_20260713/polars_stocks_selections.parquet"
)
DEFAULT_POST_SELECTIONS = (
    PROJECT_ROOT
    / "outputs/checkpoints_open_source_20260719_fixed/polars_stocks_selections.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return round(value, 10) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _json_payload(value: Any) -> str:
    return json.dumps(_clean(value), ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _cagr(values: list[float]) -> float:
    wealth = float(np.prod(1.0 + np.asarray(values, dtype=float)))
    return wealth ** (12.0 / len(values)) - 1.0


def _monthly_returns(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("model") == "Combined_Frequency")
        .select("year_month", "monthly_return")
        .sort("year_month")
    )


def _preprocessing_payload(
    pre_fix_run: Path,
    post_fix_run: Path,
    pre_selections_path: Path,
    post_selections_path: Path,
) -> dict[str, Any]:
    pre = _monthly_returns(pre_fix_run / "legacy_monthly_returns_polars.parquet").rename(
        {"monthly_return": "pre_return"}
    )
    post = _monthly_returns(post_fix_run / "legacy_monthly_returns_polars.parquet").rename(
        {"monthly_return": "post_return"}
    )
    joined = (
        pre.join(post, on="year_month", how="inner")
        .with_columns(
            (pl.col("pre_return") - pl.col("post_return")).alias("raw_diff"),
            (
                pl.col("pre_return").log1p() - pl.col("post_return").log1p()
            ).alias("log_diff"),
        )
        .sort("year_month")
    )
    n_months = joined.height
    joined = joined.with_columns(
        pl.col("year_month").dt.strftime("%Y-%m").alias("month"),
        pl.col("year_month").dt.year().alias("year"),
        (pl.col("raw_diff") * 100.0).alias("raw_diff_pp"),
        (pl.col("log_diff") * 12.0 / n_months * 100.0).alias(
            "annualized_log_gap_pp"
        ),
    )
    yearly = (
        joined.group_by("year")
        .agg(
            pl.col("annualized_log_gap_pp").sum(),
            pl.col("raw_diff_pp").abs().mean().alias("mean_abs_monthly_diff_pp"),
            pl.col("raw_diff_pp").abs().max().alias("max_abs_monthly_diff_pp"),
        )
        .sort("year")
    )

    pre_selections = pl.read_parquet(pre_selections_path)
    post_selections = pl.read_parquet(post_selections_path)
    keys = ["ticker", "year_month"]
    common = pre_selections.join(post_selections, on=keys, how="inner", suffix="_post")
    common_keys = common.height
    only_pre = pre_selections.join(post_selections.select(keys), on=keys, how="anti").height
    only_post = post_selections.join(pre_selections.select(keys), on=keys, how="anti").height
    metric_stats: list[dict[str, Any]] = []
    for metric in ["pe", "ps_ratio", "pb_ratio", "market_cap"]:
        left = common[metric].to_numpy()
        right = common[f"{metric}_post"].to_numpy()
        changed = ~np.isclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        finite = np.isfinite(left) & np.isfinite(right)
        abs_diff = np.abs(left[finite] - right[finite])
        metric_stats.append(
            {
                "metric": metric,
                "changed_rows": int(changed.sum()),
                "changed_share": float(changed.mean()),
                "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else None,
                "max_abs_diff": float(abs_diff.max()) if abs_diff.size else None,
            }
        )

    rows = joined.select(
        "month",
        "year",
        "pre_return",
        "post_return",
        "raw_diff_pp",
        "annualized_log_gap_pp",
    ).to_dicts()
    pre_values = joined["pre_return"].to_list()
    post_values = joined["post_return"].to_list()
    return {
        "summary": {
            "months": n_months,
            "pre_cagr": _cagr(pre_values),
            "post_cagr": _cagr(post_values),
            "cagr_gap_pp": 100.0 * (_cagr(pre_values) - _cagr(post_values)),
            "annualized_log_gap_pp": float(joined["annualized_log_gap_pp"].sum()),
            "changed_months": int((joined["raw_diff"].abs() > 1e-12).sum()),
            "pre_selection_rows": pre_selections.height,
            "post_selection_rows": post_selections.height,
            "common_selection_rows": common_keys,
            "only_pre_selection_rows": only_pre,
            "only_post_selection_rows": only_post,
            "membership_key_differences": only_pre + only_post,
        },
        "metrics": metric_stats,
        "monthly": rows,
        "yearly": yearly.to_dicts(),
        "largest_months": sorted(
            rows, key=lambda row: abs(float(row["raw_diff_pp"])), reverse=True
        )[:20],
    }


COMMON_CSS = r"""
:root{--bg:#F8FAFC;--panel:#FFF;--surface:#F1F5F9;--line:#D7E0EA;--ink:#020617;--muted:#475569;--navy:#111D55;--gold:#9B8816;--green:#265511;--red:#802331;--cyan:#0369A1;--shadow:0 1px 2px rgba(2,6,23,.04)}
[data-theme="dark"]{--bg:#020617;--panel:#0B1220;--surface:#111C2F;--line:rgba(148,163,184,.16);--ink:#E2E8F0;--muted:#94A3B8;--navy:#94A3E8;--gold:#D6C45B;--green:#78B85C;--red:#EC7A83;--cyan:#38BDF8}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.48 "IBM Plex Sans","Helvetica Neue",Arial,sans-serif}button,input,select{font:inherit;color:inherit}.mono,.metric strong,.num,code{font-family:"IBM Plex Mono","SFMono-Regular",Consolas,monospace;font-variant-numeric:tabular-nums}.shell{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:100vh}aside{position:sticky;top:0;height:100vh;background:#111D55;color:#fff;padding:24px 16px;display:flex;flex-direction:column}.brand{font:600 18px "IBM Plex Mono","SFMono-Regular",monospace;padding:0 10px 22px;border-bottom:1px solid rgba(255,255,255,.18)}.brand small{display:block;font:500 10px "IBM Plex Sans",sans-serif;opacity:.65;letter-spacing:.12em;text-transform:uppercase;margin-top:5px}.nav{display:grid;gap:4px;margin-top:20px}.nav a{color:rgba(255,255,255,.72);text-decoration:none;padding:10px 12px;border-radius:4px;font-weight:600}.nav a:hover,.nav a.active{background:rgba(255,255,255,.13);color:#fff}.nav span{font:500 10px "IBM Plex Mono",monospace;opacity:.55;margin-right:8px}.aside-foot{margin-top:auto;padding:14px 10px 0;border-top:1px solid rgba(255,255,255,.18);font-size:11px;color:rgba(255,255,255,.65)}main{min-width:0}.topbar{height:64px;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:0 28px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:10}.crumb{color:var(--muted);min-width:0}.icon-btn{width:36px;height:36px;border:1px solid var(--line);background:var(--panel);border-radius:4px;cursor:pointer;font-size:18px}.page{padding:28px 28px 64px;max-width:1440px;margin:auto}.hero{margin-bottom:22px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.11em;color:var(--cyan);font-weight:700}.hero h1{font-size:34px;line-height:1.08;letter-spacing:0;margin:7px 0 10px}.hero p{color:var(--muted);max-width:980px;font-size:16px}.grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);padding:18px;border-radius:8px}.panel h2,.panel h3{margin:0 0 12px}.panel h2{font-size:19px}.panel h3{font-size:15px}.panel p{color:var(--muted)}.metric{min-height:112px}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{display:block;font-size:25px;letter-spacing:0;margin:12px 0 4px}.metric small{color:var(--muted)}.positive{color:var(--green)!important}.negative{color:var(--red)!important}.section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:30px 0 12px}.section-head h2{margin:0;font-size:21px}.section-head p{margin:4px 0 0;color:var(--muted)}.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.controls input,.controls select{min-height:36px;border:1px solid var(--line);background:var(--panel);border-radius:4px;padding:7px 10px}.controls input{min-width:250px}.callout{border-left:4px solid var(--cyan);background:var(--surface);padding:15px 17px;margin:16px 0}.callout.warn{border-color:#D97706}.callout.bad{border-color:var(--red)}.callout strong{display:block;margin-bottom:4px}.chart{height:330px;position:relative}.chart svg{display:block;width:100%;height:100%;overflow:visible}.chart text{fill:var(--muted);font:10px "IBM Plex Mono",monospace}.gridline{stroke:var(--line);stroke-dasharray:3 4}.zero{stroke:var(--muted);stroke-width:1}.tooltip{position:fixed;display:none;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:8px 10px;z-index:50;font:11px "IBM Plex Mono",monospace;max-width:240px}.bar-list{display:grid;gap:8px}.bar-row{display:grid;grid-template-columns:88px minmax(0,1fr) 90px;gap:10px;align-items:center}.bar-track{height:16px;background:var(--surface);position:relative}.bar-fill{height:100%;min-width:1px}.bar-row b{text-align:right}.table-wrap{overflow:auto;border:1px solid var(--line);background:var(--panel);border-radius:8px;max-height:720px}table{width:100%;border-collapse:collapse;min-width:940px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{position:sticky;top:0;background:var(--surface);z-index:2;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;cursor:pointer}td.num{text-align:right}tbody tr:hover{background:var(--surface)}.badge{display:inline-block;border:1px solid var(--line);padding:3px 7px;border-radius:999px;font:10px "IBM Plex Mono",monospace}.badge.good{color:var(--green)}.badge.bad{color:var(--red)}.fine{font-size:12px;color:var(--muted)}.report-link{display:block;color:inherit;text-decoration:none}.report-link:hover{border-color:var(--cyan)}.report-link .arrow{float:right;color:var(--cyan)}footer{margin-top:40px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:11px}
@media(max-width:1050px){.g4{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:1fr 1fr}}
@media(max-width:760px){.shell{display:block}aside{position:relative;height:auto;padding:12px}.brand{padding:2px 5px 10px}.nav{display:flex;overflow:auto;margin-top:8px}.nav a{white-space:nowrap;padding:8px}.aside-foot{display:none}.topbar{height:52px;padding:0 14px}.page{padding:18px 12px 48px}.hero h1{font-size:27px}.g4,.g3,.g2{grid-template-columns:1fr}.section-head{align-items:start;flex-direction:column}.chart{height:280px}.bar-row{grid-template-columns:72px minmax(0,1fr) 78px}.controls input{min-width:0;width:100%}}
"""


COMMON_JS = r"""
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const pct=(x,d=2)=>x==null?"—":new Intl.NumberFormat("fr-FR",{style:"percent",minimumFractionDigits:d,maximumFractionDigits:d}).format(x);
const num=(x,d=2)=>x==null?"—":new Intl.NumberFormat("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d}).format(x);
const pp=(x,d=3)=>x==null?"—":`${num(x,d)} pt`;
const color=x=>x>=0?"var(--green)":"var(--red)";
$("#theme").onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==="dark"?"":"dark";localStorage.setItem("ar-theme",document.documentElement.dataset.theme);window.dispatchEvent(new Event("resize"))};
document.documentElement.dataset.theme=localStorage.getItem("ar-theme")||"";
function showTip(event,text){const t=$("#tooltip");t.innerHTML=text;t.style.display="block";t.style.left=Math.min(innerWidth-260,event.clientX+12)+"px";t.style.top=Math.min(innerHeight-110,event.clientY+12)+"px"}
function hideTip(){$("#tooltip").style.display="none"}
"""


def _shell(title: str, active: str, content: str, script: str = "") -> str:
    nav = [
        ("index", "01", "Synthèse", "index.html"),
        ("tickers", "02", "Attribution tickers", "ticker_attribution.html"),
        ("months", "03", "Attribution mensuelle", "monthly_attribution.html"),
        ("preprocessing", "04", "Impact preprocessing", "preprocessing_impact.html"),
    ]
    links = "".join(
        f'<a class="{"active" if key == active else ""}" href="{href}"><span>{number}</span>{label}</a>'
        for key, number, label, href in nav
    )
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{COMMON_CSS}</style></head>
<body><div class="shell"><aside><div class="brand">ALPHARANK<small>Legacy attribution audit</small></div><nav class="nav">{links}</nav><div class="aside-foot">Run validé 20260727_221253<br>Févr. 2010 → juil. 2026</div></aside><main><header class="topbar"><div class="crumb">Legacy / Attribution / <b>{html.escape(title)}</b></div><button class="icon-btn" id="theme" title="Changer de thème" aria-label="Changer de thème">◐</button></header><div class="page">{content}<footer>Généré le 2026-08-09 · Combined_Frequency · décomposition additive en log-rendement · package de replay validé.</footer></div></main></div><div class="tooltip" id="tooltip"></div><script>{COMMON_JS}{script}</script></body></html>"""


def _metric(label: str, value: str, note: str, css: str = "") -> str:
    return f'<article class="panel metric"><span>{html.escape(label)}</span><strong class="{css}">{html.escape(value)}</strong><small>{html.escape(note)}</small></article>'


def _render_index(summary: dict[str, Any], output_path: Path) -> None:
    content = f"""
<div class="hero"><div class="eyebrow">Audit complet · restitution HTML</div><h1>Pourquoi le CAGR Legacy a changé</h1><p>Quatre pages reliées, conçues pour répondre directement aux trois questions : effet du preprocessing, dépendance aux tickers et contribution de chacun des 198 mois.</p></div>
<div class="grid g4">
{_metric('CAGR Legacy', f"{100*summary['portfolio_cagr']:.2f} %", '198 mois · février 2010 à juillet 2026')}
{_metric('SPY ajusté', f"{100*summary['spy_adjusted_cagr']:.2f} %", 'même calendrier et adjusted close')}
{_metric('Tickers détenus', str(summary['ticker_count']), f"{summary['positive_ticker_count']} positifs · {summary['negative_ticker_count']} négatifs")}
{_metric('Mois audités', str(summary['months']), 'chaque contribution recompose le CAGR')}
</div>
<div class="callout warn"><strong>Verdict de contrôle</strong>Le run propre n'a aucun rendement ticker/mois supérieur à 100 %. Il reste deux défauts identifiés : le CIK général de SNDK est celui de l'ancienne société et DFS est conservé un mois après sa disparition, ce qui ajoute 0,021 point de CAGR via la renormalisation Legacy.</div>
<div class="section-head"><div><h2>Rapports</h2><p>Chaque page contient ses propres données, graphiques et filtres.</p></div></div>
<div class="grid g3">
<a class="panel report-link" href="ticker_attribution.html"><span class="arrow">→</span><div class="eyebrow">374 tickers</div><h2>Attribution par ticker</h2><p>Contributions positives et négatives, impact cash, concentration et tableau complet filtrable.</p></a>
<a class="panel report-link" href="monthly_attribution.html"><span class="arrow">→</span><div class="eyebrow">198 mois</div><h2>Attribution mensuelle</h2><p>Richesse, benchmark, contribution au CAGR et lecture détaillée de 2026.</p></a>
<a class="panel report-link" href="preprocessing_impact.html"><span class="arrow">→</span><div class="eyebrow">Avant / après</div><h2>Impact du preprocessing</h2><p>Décomposition du gap de 2,09 points de CAGR par année, mois et métrique fondamentale.</p></a>
</div>"""
    output_path.write_text(_shell("Synthèse", "index", content), encoding="utf-8")


def _render_tickers(
    summary: dict[str, Any], tickers: list[dict[str, Any]], output_path: Path
) -> None:
    content = f"""
<div class="hero"><div class="eyebrow">Contribution au backtest courant</div><h1>374 tickers, aucun résidu caché</h1><p>La contribution additive alloue log(1 + rendement). La colonne impact cash répond à la question intuitive : quel serait le CAGR si ce ticker avait été remplacé par du cash, sans refaire les rangs ni l'entraînement ?</p></div>
<div class="grid g4">
{_metric('Log-rendement annualisé', f"{100*summary['annualized_log_return']:.2f} %", 'la somme des tickers est exacte')}
{_metric('Part du Top 5', f"{100*summary['top_5_ticker_share_of_net_log_return']:.2f} %", 'du log-rendement net')}
{_metric('Part du Top 20', f"{100*summary['top_20_ticker_share_of_net_log_return']:.2f} %", 'concentration élevée mais distribuée')}
{_metric('Rendements > 100 %', str(summary['ticker_month_returns_over_100pct']), 'contre CPWR +300 % dans l’ancien run')}
</div>
<div class="section-head"><div><h2>Plus fortes contributions</h2><p>Top 12 et bottom 12 en points de log-rendement annualisé.</p></div></div>
<div class="grid g2"><article class="panel"><h3>Contributeurs positifs</h3><div class="bar-list" id="positive-bars"></div></article><article class="panel"><h3>Contributeurs négatifs</h3><div class="bar-list" id="negative-bars"></div></article></div>
<div class="section-head"><div><h2>Contrôles d'intégrité</h2><p>Les rendements extrêmes et les conventions non neutres sont isolés.</p></div></div>
<div class="grid g3">
<article class="panel"><h3>SNDK · identité temporelle</h3><p>Trois mois détenus, tous après la nouvelle cotation. Impact cash : <b>+0,835 pt</b>. Le prix n'est pas splicé dans les holdings, mais US_General conserve le CIK historique.</p></article>
<article class="panel"><h3>DFS · rendement manquant</h3><p>Poids sélectionné 5,13 % en juin 2025 après disparition du titre. La renormalisation ajoute <b>+0,021 pt</b> au CAGR contre une poche cash.</p></article>
<article class="panel"><h3>Épisodes > 50 %</h3><p>Cinq tickers seulement : ANF, NFLX, SNDK, THC et WDC. Aucun épisode détenu ne dépasse 100 % en valeur absolue.</p></article>
</div>
<div class="section-head"><div><h2>Tous les tickers</h2><p id="ticker-count"></p></div><div class="controls"><input id="ticker-search" type="search" placeholder="Rechercher un ticker…"><select id="ticker-sign"><option value="all">Toutes contributions</option><option value="positive">Positives</option><option value="negative">Négatives</option><option value="extreme">Rendement |50 %|+</option></select></div></div>
<div class="table-wrap"><table><thead><tr><th data-key="ticker">Ticker</th><th data-key="months_held">Mois détenus</th><th data-key="annualized_log_contribution_pp">Contribution log ann.</th><th data-key="cash_cagr_impact_pp">Impact CAGR cash</th><th data-key="average_effective_weight_when_held_pct">Poids moyen</th><th data-key="min_stock_month_return_pct">Pire mois</th><th data-key="max_stock_month_return_pct">Meilleur mois</th><th data-key="max_abs_month_contribution_pp">Contribution mensuelle max</th></tr></thead><tbody id="ticker-rows"></tbody></table></div>"""
    data = _json_payload(tickers)
    script = r"""
const TICKERS=__DATA__;let tickerSort={key:"annualized_log_contribution_pp",asc:false};
function bars(id,rows){const mx=Math.max(...rows.map(x=>Math.abs(x.annualized_log_contribution_pp)));$(id).innerHTML=rows.map(x=>`<div class="bar-row"><b>${x.ticker.replace('.US','')}</b><span class="bar-track"><span class="bar-fill" style="display:block;width:${100*Math.abs(x.annualized_log_contribution_pp)/mx}%;background:${color(x.annualized_log_contribution_pp)}"></span></span><span class="num ${x.annualized_log_contribution_pp>=0?'positive':'negative'}">${pp(x.annualized_log_contribution_pp)}</span></div>`).join("")}
bars("#positive-bars",TICKERS.slice().sort((a,b)=>b.annualized_log_contribution_pp-a.annualized_log_contribution_pp).slice(0,12));bars("#negative-bars",TICKERS.slice().sort((a,b)=>a.annualized_log_contribution_pp-b.annualized_log_contribution_pp).slice(0,12));
function renderTickers(){const q=$("#ticker-search").value.trim().toLowerCase(),sign=$("#ticker-sign").value;let rows=TICKERS.filter(x=>(!q||x.ticker.toLowerCase().includes(q))&&(sign==="all"||(sign==="positive"&&x.annualized_log_contribution_pp>0)||(sign==="negative"&&x.annualized_log_contribution_pp<0)||(sign==="extreme"&&x.months_abs_return_over_50pct>0)));rows.sort((a,b)=>{const av=a[tickerSort.key],bv=b[tickerSort.key];return (typeof av==="string"?av.localeCompare(bv):(av??-Infinity)-(bv??-Infinity))*(tickerSort.asc?1:-1)});$("#ticker-count").textContent=`${rows.length} ticker(s) affiché(s)`;$("#ticker-rows").innerHTML=rows.map(x=>`<tr><td><b>${x.ticker}</b></td><td class="num">${x.months_held}</td><td class="num ${x.annualized_log_contribution_pp>=0?'positive':'negative'}">${pp(x.annualized_log_contribution_pp,4)}</td><td class="num ${x.cash_cagr_impact_pp>=0?'positive':'negative'}">${pp(x.cash_cagr_impact_pp,4)}</td><td class="num">${num(x.average_effective_weight_when_held_pct)} %</td><td class="num negative">${num(x.min_stock_month_return_pct)} %</td><td class="num positive">${num(x.max_stock_month_return_pct)} %</td><td class="num">${pp(x.max_abs_month_contribution_pp)}</td></tr>`).join("")}
$("#ticker-search").oninput=renderTickers;$("#ticker-sign").onchange=renderTickers;$$('th[data-key]').forEach(th=>th.onclick=()=>{tickerSort={key:th.dataset.key,asc:tickerSort.key===th.dataset.key?!tickerSort.asc:false};renderTickers()});renderTickers();
""".replace("__DATA__", data)
    output_path.write_text(_shell("Attribution tickers", "tickers", content, script), encoding="utf-8")


def _render_months(
    summary: dict[str, Any], months: list[dict[str, Any]], output_path: Path
) -> None:
    content = f"""
<div class="hero"><div class="eyebrow">Contribution temporelle</div><h1>Chaque mois qui construit le CAGR</h1><p>Le graphique supérieur montre la richesse cumulée Legacy et SPY. Le second attribue à chaque mois sa part additive du log-rendement annualisé ; la somme vaut exactement {100*summary['annualized_log_return']:.4f} %.</p></div>
<div class="grid g4">
{_metric('Meilleur mois', 'avr. 2026', '+28,58 % · +1,5235 pt log')}
{_metric('Pire mois', 'juil. 2026', '−20,90 % · −1,4207 pt log')}
{_metric('Richesse finale', f"× {summary['terminal_wealth']:.2f}", 'base 1 en février 2010')}
{_metric('Convention DFS', f"+{summary['missing_return_renormalization_cagr_lift_pp']:.3f} pt", 'renormalisation vs poche cash')}
</div>
<div class="section-head"><div><h2>Richesse cumulée</h2><p>Legacy et SPY, même calendrier.</p></div></div><article class="panel"><div class="chart" id="wealth-chart"></div></article>
<div class="section-head"><div><h2>Contribution mensuelle au CAGR</h2><p>Points de log-rendement annualisé, positifs en vert et négatifs en rouge.</p></div><div class="controls"><select id="chart-year"><option value="all">Toute la période</option></select></div></div><article class="panel"><div class="chart" id="month-chart"></div></article>
<div class="section-head"><div><h2>Détail des mois</h2><p id="month-count"></p></div><div class="controls"><select id="table-year"><option value="all">Toutes les années</option></select><select id="month-sort"><option value="date">Ordre chronologique</option><option value="best">Meilleures contributions</option><option value="worst">Pires contributions</option></select></div></div>
<div class="table-wrap"><table><thead><tr><th>Mois</th><th>Rendement Legacy</th><th>Rendement SPY</th><th>Excès</th><th>Contribution log ann.</th><th>Impact CAGR cash</th><th>Holdings</th><th>Poids max</th><th>Richesse fin</th></tr></thead><tbody id="month-rows"></tbody></table></div>"""
    data = _json_payload(months)
    script = r"""
const MONTHS=__DATA__,years=[...new Set(MONTHS.map(x=>x.year_month.slice(0,4)))];$("#chart-year").innerHTML+=years.map(y=>`<option>${y}</option>`).join("");$("#table-year").innerHTML+=years.map(y=>`<option>${y}</option>`).join("");
function svgLine(id,rows){const el=$(id),w=Math.max(700,el.clientWidth||900),h=310,p={l:58,r:18,t:16,b:34};let lw=1,sw=1;const data=rows.map(x=>{lw*=1+x.portfolio_return;sw*=1+x.spy_adjusted_return;return {m:x.year_month,l:lw,s:sw}}),vals=data.flatMap(x=>[x.l,x.s]),mn=Math.min(...vals),mx=Math.max(...vals),xx=i=>p.l+i*(w-p.l-p.r)/Math.max(1,data.length-1),yy=v=>p.t+(mx-v)*(h-p.t-p.b)/(mx-mn||1);let out=`<svg viewBox="0 0 ${w} ${h}">`;for(let j=0;j<5;j++){const v=mn+(mx-mn)*j/4,y=yy(v);out+=`<line class="gridline" x1="${p.l}" y1="${y}" x2="${w-p.r}" y2="${y}"/><text x="4" y="${y+3}">×${num(v,1)}</text>`}out+=`<polyline fill="none" stroke="var(--navy)" stroke-width="2.5" points="${data.map((x,i)=>`${xx(i)},${yy(x.l)}`).join(' ')}"/><polyline fill="none" stroke="var(--gold)" stroke-width="1.8" points="${data.map((x,i)=>`${xx(i)},${yy(x.s)}`).join(' ')}"/>`;[0,Math.floor(data.length/2),data.length-1].forEach(i=>out+=`<text x="${xx(i)}" y="${h-8}" text-anchor="${i===0?'start':i===data.length-1?'end':'middle'}">${data[i].m}</text>`);out+=`</svg><span class="badge" style="position:absolute;top:8px;right:126px;color:var(--navy)">Legacy</span><span class="badge" style="position:absolute;top:8px;right:18px;color:var(--gold)">SPY</span>`;el.innerHTML=out}
function svgBars(){const yr=$("#chart-year").value,rows=MONTHS.filter(x=>yr==="all"||x.year_month.startsWith(yr)),el=$("#month-chart"),w=Math.max(700,el.clientWidth||900),h=310,p={l:58,r:18,t:18,b:34},vals=rows.map(x=>x.annualized_log_contribution_pp),lim=Math.max(...vals.map(Math.abs),.01),xx=i=>p.l+i*(w-p.l-p.r)/rows.length,bw=Math.max(1,(w-p.l-p.r)/rows.length-1),yy=v=>p.t+(lim-v)*(h-p.t-p.b)/(2*lim),zero=yy(0);let out=`<svg viewBox="0 0 ${w} ${h}"><line class="zero" x1="${p.l}" y1="${zero}" x2="${w-p.r}" y2="${zero}"/>`;[-lim,-lim/2,0,lim/2,lim].forEach(v=>{const y=yy(v);out+=`<line class="gridline" x1="${p.l}" y1="${y}" x2="${w-p.r}" y2="${y}"/><text x="2" y="${y+3}">${pp(v,2)}</text>`});rows.forEach((x,i)=>{const y=yy(x.annualized_log_contribution_pp),height=Math.abs(zero-y);out+=`<rect x="${xx(i)}" y="${Math.min(y,zero)}" width="${bw}" height="${Math.max(1,height)}" fill="${x.annualized_log_contribution_pp>=0?'var(--green)':'var(--red)'}" data-i="${i}"/>`});[0,Math.floor(rows.length/2),rows.length-1].forEach(i=>out+=`<text x="${xx(i)}" y="${h-8}" text-anchor="${i===0?'start':i===rows.length-1?'end':'middle'}">${rows[i].year_month}</text>`);out+=`</svg>`;el.innerHTML=out;el.querySelectorAll('rect[data-i]').forEach(r=>{const x=rows[Number(r.dataset.i)];r.onmousemove=e=>showTip(e,`<b>${x.year_month}</b><br>Legacy ${pct(x.portfolio_return)}<br>SPY ${pct(x.spy_adjusted_return)}<br>Contribution ${pp(x.annualized_log_contribution_pp,4)}`);r.onmouseleave=hideTip})}
function renderMonths(){const yr=$("#table-year").value,sort=$("#month-sort").value;let rows=MONTHS.filter(x=>yr==="all"||x.year_month.startsWith(yr));if(sort==="best")rows.sort((a,b)=>b.annualized_log_contribution_pp-a.annualized_log_contribution_pp);else if(sort==="worst")rows.sort((a,b)=>a.annualized_log_contribution_pp-b.annualized_log_contribution_pp);else rows.sort((a,b)=>a.year_month.localeCompare(b.year_month));$("#month-count").textContent=`${rows.length} mois affiché(s)`;$("#month-rows").innerHTML=rows.map(x=>`<tr><td><b>${x.year_month}</b></td><td class="num ${x.portfolio_return>=0?'positive':'negative'}">${pct(x.portfolio_return)}</td><td class="num ${x.spy_adjusted_return>=0?'positive':'negative'}">${pct(x.spy_adjusted_return)}</td><td class="num ${x.active_return>=0?'positive':'negative'}">${pct(x.active_return)}</td><td class="num ${x.annualized_log_contribution_pp>=0?'positive':'negative'}">${pp(x.annualized_log_contribution_pp,4)}</td><td class="num ${x.cash_cagr_impact_pp>=0?'positive':'negative'}">${pp(x.cash_cagr_impact_pp,4)}</td><td class="num">${x.holdings}</td><td class="num">${pct(x.max_weight)}</td><td class="num">×${num(x.wealth_after,3)}</td></tr>`).join("")}
$("#chart-year").onchange=svgBars;$("#table-year").onchange=renderMonths;$("#month-sort").onchange=renderMonths;window.addEventListener('resize',()=>{svgLine('#wealth-chart',MONTHS);svgBars()});svgLine('#wealth-chart',MONTHS);svgBars();renderMonths();
""".replace("__DATA__", data)
    output_path.write_text(_shell("Attribution mensuelle", "months", content, script), encoding="utf-8")


def _render_preprocessing(payload: dict[str, Any], output_path: Path) -> None:
    summary = payload["summary"]
    content = f"""
<div class="hero"><div class="eyebrow">Run 13 juillet vs run 19 juillet</div><h1>Le preprocessing a modifié tout l'historique</h1><p>Ce rapport compare exactement les deux runs sur leurs {summary['months']} mois communs. Le gap n'est pas une correction de fin de série : l'univers fondamental, les rangs et les optimisations annuelles changent à travers toute la période.</p></div>
<div class="grid g4">
{_metric('CAGR avant', f"{100*summary['pre_cagr']:.2f} %", 'preprocessing ancien')}
{_metric('CAGR après', f"{100*summary['post_cagr']:.2f} %", 'preprocessing déterministe')}
{_metric('Gap de CAGR', f"−{summary['cagr_gap_pp']:.2f} pts", f"{summary['annualized_log_gap_pp']:.4f} pt de log annualisé", 'negative')}
{_metric('Mois modifiés', f"{summary['changed_months']} / {summary['months']}", 'aucun segment historique épargné')}
</div>
<div class="callout bad"><strong>Cause racine</strong>Des lignes fondamentales sans date as-of exploitable et des doublons ticker/date étaient forward-fillés, puis plusieurs group_by().last() dépendaient de l'ordre implicite des lignes. Le correctif impose un tri stable, une règle de tie-break et exclut les lignes sans date causale.</div>
<div class="section-head"><div><h2>Contribution annuelle au gap</h2><p>Somme des écarts mensuels de log-rendement annualisé. Les signes opposés montrent les fortes compensations.</p></div></div><article class="panel"><div class="chart" id="year-chart"></div></article>
<div class="section-head"><div><h2>Écart mois par mois</h2><p>Rendement avant moins rendement après, en points de pourcentage.</p></div></div><article class="panel"><div class="chart" id="diff-chart"></div></article>
<div class="section-head"><div><h2>Propagation dans les fondamentaux</h2><p>{summary['membership_key_differences']:,} clés ticker/mois de membership diffèrent.</p></div></div><div class="grid g4" id="metric-cards"></div>
<div class="section-head"><div><h2>Plus gros écarts mensuels</h2><p>Août 2011 inclut l'exposition CPWR de l'ancien run.</p></div></div><div class="table-wrap" style="max-height:none"><table><thead><tr><th>Mois</th><th>Avant</th><th>Après</th><th>Écart</th><th>Contribution log ann.</th></tr></thead><tbody id="largest-rows"></tbody></table></div>
<div class="section-head"><div><h2>Tous les mois</h2><p>Lecture complète de la réécriture des sélections dérivées.</p></div><div class="controls"><select id="diff-year"><option value="all">Toutes les années</option></select></div></div><div class="table-wrap"><table><thead><tr><th>Mois</th><th>Rendement avant</th><th>Rendement après</th><th>Écart brut</th><th>Contribution log ann.</th></tr></thead><tbody id="diff-rows"></tbody></table></div>""".replace(
        f"{summary['membership_key_differences']:,}",
        f"{summary['membership_key_differences']:,}".replace(",", " "),
    )
    data = _json_payload(payload)
    script = r"""
const P=__DATA__,years=P.yearly.map(x=>String(x.year));$("#diff-year").innerHTML+=years.map(y=>`<option>${y}</option>`).join("");
function verticalBars(id,rows,key,label){const el=$(id),w=Math.max(700,el.clientWidth||900),h=310,p={l:58,r:18,t:18,b:34},vals=rows.map(x=>x[key]),lim=Math.max(...vals.map(Math.abs),.01),xx=i=>p.l+i*(w-p.l-p.r)/rows.length,bw=Math.max(2,(w-p.l-p.r)/rows.length-2),yy=v=>p.t+(lim-v)*(h-p.t-p.b)/(2*lim),zero=yy(0);let out=`<svg viewBox="0 0 ${w} ${h}"><line class="zero" x1="${p.l}" y1="${zero}" x2="${w-p.r}" y2="${zero}"/>`;[-lim,-lim/2,0,lim/2,lim].forEach(v=>{const y=yy(v);out+=`<line class="gridline" x1="${p.l}" y1="${y}" x2="${w-p.r}" y2="${y}"/><text x="2" y="${y+3}">${pp(v,2)}</text>`});rows.forEach((x,i)=>{const y=yy(x[key]);out+=`<rect x="${xx(i)}" y="${Math.min(y,zero)}" width="${bw}" height="${Math.max(1,Math.abs(zero-y))}" fill="${x[key]>=0?'var(--green)':'var(--red)'}" data-i="${i}"/>`});[0,Math.floor(rows.length/2),rows.length-1].forEach(i=>out+=`<text x="${xx(i)}" y="${h-8}" text-anchor="${i===0?'start':i===rows.length-1?'end':'middle'}">${rows[i][label]}</text>`);out+='</svg>';el.innerHTML=out;el.querySelectorAll('rect').forEach(r=>{const x=rows[Number(r.dataset.i)];r.onmousemove=e=>showTip(e,`<b>${x[label]}</b><br>${pp(x[key],4)}`);r.onmouseleave=hideTip})}
function renderDiffRows(){const yr=$("#diff-year").value,rows=P.monthly.filter(x=>yr==="all"||String(x.year)===yr);$("#diff-rows").innerHTML=rows.map(x=>`<tr><td><b>${x.month}</b></td><td class="num">${pct(x.pre_return)}</td><td class="num">${pct(x.post_return)}</td><td class="num ${x.raw_diff_pp>=0?'positive':'negative'}">${pp(x.raw_diff_pp)}</td><td class="num ${x.annualized_log_gap_pp>=0?'positive':'negative'}">${pp(x.annualized_log_gap_pp,4)}</td></tr>`).join("")}
$("#metric-cards").innerHTML=P.metrics.map(x=>`<article class="panel metric"><span>${x.metric}</span><strong>${x.changed_rows.toLocaleString('fr-FR')}</strong><small>${pct(x.changed_share,1)} des clés communes · écart max ${num(x.max_abs_diff,2)}</small></article>`).join("");$("#largest-rows").innerHTML=P.largest_months.map(x=>`<tr><td><b>${x.month}</b></td><td class="num">${pct(x.pre_return)}</td><td class="num">${pct(x.post_return)}</td><td class="num ${x.raw_diff_pp>=0?'positive':'negative'}">${pp(x.raw_diff_pp)}</td><td class="num ${x.annualized_log_gap_pp>=0?'positive':'negative'}">${pp(x.annualized_log_gap_pp,4)}</td></tr>`).join("");$("#diff-year").onchange=renderDiffRows;window.addEventListener('resize',()=>{verticalBars('#year-chart',P.yearly,'annualized_log_gap_pp','year');verticalBars('#diff-chart',P.monthly,'raw_diff_pp','month')});verticalBars('#year-chart',P.yearly,'annualized_log_gap_pp','year');verticalBars('#diff-chart',P.monthly,'raw_diff_pp','month');renderDiffRows();
""".replace("__DATA__", data)
    output_path.write_text(
        _shell("Impact preprocessing", "preprocessing", content, script),
        encoding="utf-8",
    )


def render_reports(
    *,
    attribution_dir: Path,
    output_dir: Path,
    pre_fix_run: Path,
    post_fix_run: Path,
    pre_selections: Path,
    post_selections: Path,
) -> list[Path]:
    inputs = [
        attribution_dir / "summary.json",
        attribution_dir / "ticker_contributions.csv",
        attribution_dir / "monthly_contributions.csv",
        attribution_dir / "ticker_month_contributions.csv",
        pre_fix_run / "legacy_monthly_returns_polars.parquet",
        post_fix_run / "legacy_monthly_returns_polars.parquet",
        pre_selections,
        post_selections,
    ]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report inputs:\n" + "\n".join(map(str, missing)))

    summary = json.loads((attribution_dir / "summary.json").read_text(encoding="utf-8"))
    tickers = pd.read_csv(attribution_dir / "ticker_contributions.csv").replace(
        {np.nan: None}
    ).to_dict(orient="records")
    months = pd.read_csv(attribution_dir / "monthly_contributions.csv").replace(
        {np.nan: None}
    ).to_dict(orient="records")
    preprocessing = _preprocessing_payload(
        pre_fix_run, post_fix_run, pre_selections, post_selections
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        output_dir / "index.html",
        output_dir / "ticker_attribution.html",
        output_dir / "monthly_attribution.html",
        output_dir / "preprocessing_impact.html",
    ]
    _render_index(summary, reports[0])
    _render_tickers(summary, tickers, reports[1])
    _render_months(summary, months, reports[2])
    _render_preprocessing(preprocessing, reports[3])

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "legacy_attribution_html_reports",
        "reports": [path.name for path in reports],
        "sources": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        ],
        "preprocessing_summary": preprocessing["summary"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_clean(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, default=DEFAULT_ATTRIBUTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pre-fix-run", type=Path, default=DEFAULT_PRE_FIX_RUN)
    parser.add_argument("--post-fix-run", type=Path, default=DEFAULT_POST_FIX_RUN)
    parser.add_argument("--pre-selections", type=Path, default=DEFAULT_PRE_SELECTIONS)
    parser.add_argument("--post-selections", type=Path, default=DEFAULT_POST_SELECTIONS)
    args = parser.parse_args()
    for report in render_reports(
        attribution_dir=args.attribution_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        pre_fix_run=args.pre_fix_run.resolve(),
        post_fix_run=args.post_fix_run.resolve(),
        pre_selections=args.pre_selections.resolve(),
        post_selections=args.post_selections.resolve(),
    ):
        print(report)


if __name__ == "__main__":
    main()
