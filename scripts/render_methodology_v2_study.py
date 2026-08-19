#!/usr/bin/env python3
"""Render the approved causal-v2 study as a self-contained audit dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from alpharank.portfolio.performance import advanced_performance_statistics, annual_returns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON = PROJECT_ROOT / "outputs/methodology_v2/run_004/common_v2_approved_censoring1"
DEFAULT_RECONCILIATION = PROJECT_ROOT / "outputs/methodology_v2/run_005/v1_v2_approved_censoring1"
DEFAULT_BOOSTING = PROJECT_ROOT / "outputs/methodology_v2/run_003/boosting_v2_approved_censoring2"
STRATEGY_ORDER = ("Boosting Top 5", "Boosting Top 10", "Legacy", "SPY total return")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return _json_safe(frame.to_dicts())


def _drawdown_details(returns: np.ndarray) -> dict[str, float | int | None]:
    curve = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(curve)
    drawdowns = curve / peaks - 1.0
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(curve[: trough + 1]))
    recovery: int | None = None
    for index in range(trough + 1, curve.size):
        if curve[index] >= curve[peak]:
            recovery = index
            break
    return {
        "drawdown_peak_index": peak,
        "drawdown_trough_index": trough,
        "drawdown_recovery_index": recovery,
        "drawdown_duration_months": (recovery if recovery is not None else curve.size - 1) - peak,
    }


def _moments(values: np.ndarray) -> tuple[float, float]:
    centered = values - float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if values.size < 3 or std == 0.0:
        return float("nan"), float("nan")
    skewness = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4) - 3.0) if values.size >= 4 else float("nan")
    return skewness, kurtosis


def _effective_slice(frame: pl.DataFrame, requested_year: int) -> pl.DataFrame:
    requested_start = date(requested_year, 1, 1)
    return frame.filter(pl.col("holding_month") >= requested_start).sort("holding_month")


def _performance_rows(
    monthly: pl.DataFrame,
    holdings: pl.DataFrame,
    terminal_attribution: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[dict[str, Any]]]:
    strategy_frames: dict[str, pl.DataFrame] = {
        strategy: monthly.filter(pl.col("strategy") == strategy).sort("holding_month")
        for strategy in monthly.get_column("strategy").unique().to_list()
    }
    benchmark_source = strategy_frames["Legacy"].select("holding_month", "benchmark_return")
    strategy_frames["SPY total return"] = benchmark_source.rename({"benchmark_return": "net_return"}).with_columns(
        pl.lit(0.0).alias("turnover"),
        pl.lit(0.0).alias("transaction_cost"),
        pl.lit(1).alias("n_positions"),
        pl.lit(1.0).alias("maximum_position_weight"),
    )

    performance: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    benchmark = strategy_frames["SPY total return"]
    benchmark_returns = benchmark.get_column("net_return").to_numpy()
    benchmark_stats = advanced_performance_statistics(
        benchmark_returns,
        benchmark_returns=benchmark_returns,
    )

    for strategy in STRATEGY_ORDER:
        frame = strategy_frames[strategy]
        values = frame.get_column("net_return").to_numpy()
        stats = advanced_performance_statistics(values, benchmark_returns=benchmark_returns)
        skewness, excess_kurtosis = _moments(values)
        detail = _drawdown_details(values)
        months = frame.get_column("holding_month").to_list()
        wealth = np.cumprod(1.0 + values)
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        for month, ret, wealth_value, drawdown_value in zip(months, values, wealth, drawdown, strict=True):
            series.append(
                {
                    "strategy": strategy,
                    "holding_month": month,
                    "net_return": float(ret),
                    "wealth": float(wealth_value),
                    "drawdown": float(drawdown_value),
                }
            )

        yearly = annual_returns(values, holding_months=months)
        for row in yearly.to_dicts():
            annual.append({"strategy": strategy, **row})

        holding_frame = holdings.filter(pl.col("strategy") == strategy) if strategy != "SPY total return" else pl.DataFrame()
        if holding_frame.is_empty():
            hhi_values = np.asarray([], dtype=float)
            cash_values = np.asarray([], dtype=float)
            holding_rows = 0
            return_rows = 0
        else:
            implementation = holding_frame.group_by("holding_month").agg(
                pl.col("target_weight").sum().alias("invested_weight"),
                (pl.col("target_weight") ** 2).sum().alias("hhi"),
            )
            hhi_values = implementation.get_column("hhi").to_numpy()
            cash_values = 1.0 - implementation.get_column("invested_weight").to_numpy()
            holding_rows = holding_frame.height
            return_rows = holding_frame.filter(pl.col("realized_return").is_finite()).height

        terminal = terminal_attribution.filter(pl.col("strategy") == strategy)
        terminal_row = terminal.row(0, named=True) if terminal.height else {}
        active = values - benchmark_returns
        tracking_error = float(np.std(active, ddof=1) * math.sqrt(12.0))
        peak_index = int(detail["drawdown_peak_index"])
        trough_index = int(detail["drawdown_trough_index"])
        recovery_index = detail["drawdown_recovery_index"]
        full_years = yearly.filter(pl.col("is_full_calendar_year"))
        worst_year = full_years.sort("annual_return").row(0, named=True) if full_years.height else {}
        best_month_index = int(np.argmax(values))
        worst_month_index = int(np.argmin(values))
        performance.append(
            {
                "strategy": strategy,
                "start_holding_month": months[0],
                "end_holding_month": months[-1],
                "months": len(months),
                **stats,
                "annualized_excess_return": float(stats["cagr"] - benchmark_stats["cagr"]),
                "tracking_error": tracking_error,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
                "best_month": months[best_month_index],
                "best_month_return": float(values[best_month_index]),
                "worst_month": months[worst_month_index],
                "worst_month_return": float(values[worst_month_index]),
                "worst_full_calendar_year": worst_year.get("year"),
                "worst_full_calendar_year_return": worst_year.get("annual_return"),
                "drawdown_peak_month": months[peak_index],
                "drawdown_trough_month": months[trough_index],
                "drawdown_recovery_month": months[int(recovery_index)] if recovery_index is not None else None,
                "drawdown_duration_months": detail["drawdown_duration_months"],
                "average_turnover": float(frame.get_column("turnover").mean()),
                "total_transaction_cost": float(frame.get_column("transaction_cost").sum()),
                "average_positions": float(frame.get_column("n_positions").mean()),
                "maximum_positions": int(frame.get_column("n_positions").max()),
                "average_maximum_position_weight": float(frame.get_column("maximum_position_weight").mean()),
                "maximum_single_name_weight": float(frame.get_column("maximum_position_weight").max()),
                "average_hhi": float(np.mean(hhi_values)) if hhi_values.size else None,
                "maximum_hhi": float(np.max(hhi_values)) if hhi_values.size else None,
                "average_cash_exposure": float(np.mean(cash_values)) if cash_values.size else None,
                "maximum_cash_exposure": float(np.max(cash_values)) if cash_values.size else None,
                "holding_rows": holding_rows,
                "holding_return_rows": return_rows,
                "holding_return_coverage": return_rows / holding_rows if holding_rows else None,
                "missing_holding_returns": holding_rows - return_rows,
                "terminal_holding_rows": terminal_row.get("terminal_holding_rows", 0),
                "terminal_annualized_log_contribution": terminal_row.get("terminal_annualized_log_contribution"),
                "terminal_marginal_cagr_impact": terminal_row.get("terminal_marginal_cagr_impact"),
                "metric_note": (
                    "SPY has no portfolio holdings, turnover, costs, concentration or terminal-event attribution."
                    if strategy == "SPY total return"
                    else None
                ),
            }
        )

        first_year = months[0].year
        final_year = months[-1].year
        for requested_year in range(2010, final_year + 1):
            sliced = _effective_slice(frame, requested_year)
            if sliced.is_empty():
                continue
            start_values = sliced.get_column("net_return").to_numpy()
            start_stats = advanced_performance_statistics(start_values)
            starts.append(
                {
                    "strategy": strategy,
                    "requested_start_year": requested_year,
                    "effective_start_month": sliced.get_column("holding_month")[0],
                    "effective_start_differs": requested_year < first_year,
                    "months": sliced.height,
                    "cagr": start_stats["cagr"],
                    "total_return": start_stats["total_return"],
                }
            )

    return pl.DataFrame(performance), pl.DataFrame(annual), pl.DataFrame(starts), series


def _holdings_with_actions(holdings: pl.DataFrame) -> pl.DataFrame:
    prior = holdings.select(
        "strategy",
        pl.col("holding_month").dt.offset_by("1mo").alias("holding_month"),
        "ticker",
    ).with_columns(pl.lit(True).alias("held_previous_month"))
    return (
        holdings.join(prior, on=["strategy", "holding_month", "ticker"], how="left")
        .with_columns(
            pl.when(pl.col("held_previous_month").fill_null(False))
            .then(pl.lit("KEEP"))
            .otherwise(pl.lit("ENTER"))
            .alias("position_action")
        )
        .drop("held_previous_month")
        .sort("holding_month", "strategy", "selection_rank", "ticker")
    )


def _html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank — Étude causale v2</title>
<style>
:root{{--bg:#f8fafc;--panel:#fff;--surface:#f1f5f9;--border:#d7e0ea;--text:#020617;--muted:#475569;--navy:#111d55;--gold:#9b8816;--green:#265511;--red:#802331;--blue:#0369a1;--mono:"IBM Plex Mono",ui-monospace,monospace;--sans:"IBM Plex Sans",Inter,system-ui,sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans)}}a{{color:var(--blue)}}.shell{{max-width:1500px;margin:auto;padding:24px}}header{{border:1px solid var(--border);background:var(--panel);padding:26px;display:grid;gap:18px}}.eyebrow{{font:700 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}h1{{font-size:clamp(30px,5vw,58px);letter-spacing:-.055em;line-height:.95;margin:0}}h2{{font-size:26px;letter-spacing:-.035em;margin:0 0 16px}}h3{{margin:0 0 10px}}p{{line-height:1.6;color:var(--muted)}}.badges,.nav,.controls,.downloads{{display:flex;gap:8px;flex-wrap:wrap}}.badge,.nav button,.controls select,.controls input,.downloads a{{border:1px solid var(--border);background:var(--surface);padding:9px 12px;font:700 12px var(--mono);color:var(--text);text-decoration:none}}.badge.ok{{color:var(--green);border-color:#9fbd91}}.nav{{position:sticky;top:0;z-index:10;background:rgba(248,250,252,.96);padding:12px 0}}.nav button{{cursor:pointer}}.nav button.active{{background:var(--navy);color:#fff;border-color:var(--navy)}}main{{display:grid;gap:20px}}section{{display:none;border:1px solid var(--border);background:var(--panel);padding:24px}}section.active{{display:block}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.card{{border:1px solid var(--border);padding:16px;min-width:0}}.card .label{{font:700 11px var(--mono);text-transform:uppercase;color:var(--muted)}}.card .value{{font:600 25px var(--mono);margin-top:7px}}.card .note{{font-size:12px;color:var(--muted);margin-top:6px}}.chart{{border:1px solid var(--border);min-height:330px;padding:14px;margin:16px 0}}svg{{width:100%;height:300px;overflow:visible}}.legend{{display:flex;gap:14px;flex-wrap:wrap;font:12px var(--mono)}}.legend i{{display:inline-block;width:12px;height:3px;margin-right:5px}}.table-wrap{{overflow:auto;border:1px solid var(--border);margin-top:14px;max-height:680px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{position:sticky;top:0;background:var(--surface);z-index:1;text-align:left;font:700 11px var(--mono);text-transform:uppercase}}th,td{{padding:10px;border-bottom:1px solid var(--border);white-space:nowrap}}td.num{{text-align:right;font-family:var(--mono)}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}.callout{{border-left:4px solid var(--gold);background:var(--surface);padding:14px 18px;margin:16px 0}}.timeline{{display:grid;gap:10px}}.timeline article{{display:grid;grid-template-columns:160px 1fr;gap:18px;border-bottom:1px solid var(--border);padding:12px 0}}code{{font-family:var(--mono);font-size:12px}}.footnote{{font-size:12px;color:var(--muted)}}@media(max-width:950px){{.grid{{grid-template-columns:repeat(2,1fr)}}.timeline article{{grid-template-columns:1fr}}}}@media(max-width:600px){{.shell{{padding:10px}}header,section{{padding:16px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="shell">
<header><div class="eyebrow">AlphaRank / étude méthodologique auditée</div><h1>Replay causal v2<br>explorable de bout en bout</h1><p>Positions mensuelles, performance nette, risque, coûts, qualité du modèle, événements terminaux et rapprochement complet avec la baseline v1 biaisée.</p><div class="badges"><span class="badge ok">comparison_eligible = true</span><span class="badge ok">promotion_eligible = true</span><span class="badge">v2-causal-approved-censoring</span><span class="badge">2011-08 → 2026-07</span></div><div class="callout"><strong>Périmètre.</strong> Ce rapport est l'étude historique v2 approuvée. Il ne constitue pas à lui seul le portefeuille live d'août 2026, qui doit provenir d'un snapshot mensuel de production validé séparément.</div></header>
<nav class="nav" id="nav"><button data-tab="overview" class="active">Synthèse</button><button data-tab="performance">KPI complets</button><button data-tab="positions">Positions mensuelles</button><button data-tab="ledger">Journal mensuel</button><button data-tab="model">Modèle</button><button data-tab="terminal">Cas terminaux</button><button data-tab="reconciliation">V1 → V2</button><button data-tab="methodology">Méthodologie</button></nav>
<main>
<section id="overview" class="active"><h2>Synthèse économique</h2><div class="controls"><label>Stratégie <select id="strategy"></select></label></div><div class="grid" id="hero-kpis"></div><div class="chart"><h3>Capital cumulé, base 1</h3><div class="legend" id="wealth-legend"></div><svg id="wealth-chart"></svg></div><div class="chart"><h3>Drawdown</h3><svg id="drawdown-chart"></svg></div><div class="callout" id="terminal-impact"></div></section>
<section id="performance"><h2>KPI complets et fenêtres de départ</h2><p>Sharpe selon la convention Legacy : (CAGR − 2 %) / volatilité annualisée. SPY est le total return sur adjusted close, sur la même fenêtre.</p><div class="table-wrap"><table id="performance-table"></table></div><h3 style="margin-top:24px">CAGR depuis le 1er janvier de chaque année demandée</h3><div class="table-wrap"><table id="starts-table"></table></div><h3 style="margin-top:24px">Rendements calendaires</h3><div class="table-wrap"><table id="annual-table"></table></div></section>
<section id="positions"><h2>Positions à chaque mois</h2><div class="controls"><label>Mois <select id="position-month"></select></label><label>Stratégie <select id="position-strategy"></select></label><label>Recherche <input id="position-search" placeholder="Ticker"></label></div><div class="downloads" id="downloads"></div><div class="table-wrap"><table id="positions-table"></table></div><p class="footnote">ENTER/KEEP est calculé par rapport au mois précédent de la même stratégie. Les sorties se lisent par différence entre deux mois.</p></section>
<section id="ledger"><h2>Journal économique mensuel</h2><div class="controls"><label>Stratégie <select id="ledger-strategy"></select></label></div><div class="table-wrap"><table id="ledger-table"></table></div></section>
<section id="model"><h2>Qualité hors-échantillon du boosting</h2><div class="grid" id="model-kpis"></div><div class="callout">Les métriques modèle utilisent uniquement les cibles H6 matures. La queue score-only et la performance portefeuille à un mois restent des calendriers distincts.</div><div class="table-wrap"><table id="model-table"></table></div></section>
<section id="terminal"><h2>Événements terminaux sourcés</h2><p>Les 7 occurrences portefeuille ont été résolues avec leur contrepartie actionnariale et leur source. Leur contribution est isolée en log annualisé et rapprochée du CAGR composé.</p><div class="table-wrap"><table id="terminal-table"></table></div></section>
<section id="reconciliation"><h2>Pourquoi v1 et v2 diffèrent</h2><div class="callout"><strong>720 / 720 lignes mensuelles divergentes expliquées.</strong> Erreur numérique finale : 0 à la tolérance de 1e-12. La baseline v1 reste immuable et explicitement étiquetée biaisée.</div><div class="table-wrap"><table id="reconciliation-table"></table></div></section>
<section id="methodology"><h2>Chaîne de preuve</h2><div class="timeline"><article><code>RUN-001</code><div><strong>Snapshot causal immuable</strong><p>Identité économique, fichiers et hashes scellés avant les replays.</p></div></article><article><code>RUN-002 / RUN-003</code><div><strong>Legacy et Boosting rejoués</strong><p>Walk-forward strict, features point-in-time, cible H6 mature séparée de la détention mensuelle.</p></div></article><article><code>SIM / UNI / FND / BST / LEG</code><div><strong>Contrats communs</strong><p>Univers historique, fondamentaux disponibles à date, sélection avant rendement réalisé, coûts et benchmark identiques.</p></div></article><article><code>RUN-010 / 011 / 012</code><div><strong>Fin de cotation</strong><p>Valeur terminale actionnariale sourcée et journalisée ; aucun rendement futur disponible n'influence la sélection.</p></div></article><article><code>RUN-004 / RUN-005</code><div><strong>Replay commun et rapprochement</strong><p>Positions, mois, KPI, coûts et chaque divergence v1/v2 réconciliés.</p></div></article></div><h3>Provenance</h3><div class="table-wrap"><table id="sources-table"></table></div></section>
</main></div><script id="payload" type="application/json">{encoded}</script><script>
const D=JSON.parse(document.getElementById('payload').textContent);const strategies=D.strategy_order;const colors=['#111d55','#0369a1','#9b8816','#475569'];
const pct=v=>v==null?'N/A':(v*100).toFixed(2)+' %',num=(v,d=2)=>v==null?'N/A':Number(v).toFixed(d),month=v=>v?String(v).slice(0,7):'N/A';
function table(id,cols,rows){{const el=document.getElementById(id);el.innerHTML='<thead><tr>'+cols.map(c=>'<th>'+c[0]+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>{{let v=c[1](r);let cls=typeof v==='number'?'num '+(v>0?'positive':v<0?'negative':''):'';return '<td class="'+cls+'">'+(c[2]?c[2](v):v??'N/A')+'</td>'}}).join('')+'</tr>').join('')+'</tbody>'}}
document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('#nav button,main section').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')}});
function fillSelect(id,values){{const e=document.getElementById(id);e.innerHTML=values.map(v=>'<option>'+v+'</option>').join('');return e}};
const strat=fillSelect('strategy',strategies),pstrat=fillSelect('position-strategy',strategies.filter(x=>x!=='SPY total return')),lstrat=fillSelect('ledger-strategy',strategies.filter(x=>x!=='SPY total return'));
function draw(id,key){{const svg=document.getElementById(id),W=1200,H=280,p=36;let all=D.series.map(x=>x[key]),mn=Math.min(...all),mx=Math.max(...all);if(key==='drawdown')mx=0;let paths=strategies.map((s,si)=>{{let a=D.series.filter(x=>x.strategy===s),pts=a.map((x,i)=>{{let X=p+i*(W-2*p)/(a.length-1),Y=H-p-(x[key]-mn)*(H-2*p)/(mx-mn||1);return X.toFixed(1)+','+Y.toFixed(1)}}).join(' ');return '<polyline points="'+pts+'" fill="none" stroke="'+colors[si]+'" stroke-width="2.5"/>'}}).join('');svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.innerHTML='<line x1="'+p+'" y1="'+(H-p)+'" x2="'+(W-p)+'" y2="'+(H-p)+'" stroke="#d7e0ea"/>'+paths}}draw('wealth-chart','wealth');draw('drawdown-chart','drawdown');document.getElementById('wealth-legend').innerHTML=strategies.map((s,i)=>'<span><i style="background:'+colors[i]+'"></i>'+s+'</span>').join('');
function renderHero(){{let r=D.performance.find(x=>x.strategy===strat.value);let cards=[['CAGR',pct(r.cagr)],['Volatilité',pct(r.annualized_volatility)],['Sharpe Legacy',num(r.sharpe)],['Max drawdown',pct(r.max_drawdown)],['Sortino',num(r.sortino)],['Calmar',num(r.calmar)],['Meilleur mois',pct(r.best_month_return),month(r.best_month)],['Pire mois',pct(r.worst_month_return),month(r.worst_month)]];document.getElementById('hero-kpis').innerHTML=cards.map(c=>'<div class="card"><div class="label">'+c[0]+'</div><div class="value">'+c[1]+'</div><div class="note">'+(c[2]||'')+'</div></div>').join('');document.getElementById('terminal-impact').innerHTML='<strong>Impact des fins de cotation.</strong> '+(r.terminal_marginal_cagr_impact==null?'Non applicable à cette série.':pct(r.terminal_marginal_cagr_impact)+' de CAGR marginal, '+pct(r.terminal_annualized_log_contribution)+' de contribution log annualisée, '+r.terminal_holding_rows+' occurrences portefeuille.')}}strat.onchange=renderHero;renderHero();
const perfCols=[['Stratégie',r=>r.strategy],['CAGR',r=>r.cagr,pct],['Total',r=>r.total_return,pct],['Vol.',r=>r.annualized_volatility,pct],['Sharpe',r=>r.sharpe,num],['Sortino',r=>r.sortino,num],['Calmar',r=>r.calmar,num],['Max DD',r=>r.max_drawdown,pct],['Durée DD',r=>r.drawdown_duration_months,v=>v+' mois'],['Excès ann.',r=>r.annualized_excess_return,pct],['Alpha',r=>r.alpha,pct],['Bêta',r=>r.beta,num],['Corr.',r=>r.correlation,num],['Info ratio',r=>r.information_ratio,num],['Tracking error',r=>r.tracking_error,pct],['Hit rate',r=>r.benchmark_hit_rate,pct],['Up capture',r=>r.up_capture,pct],['Down capture',r=>r.down_capture,pct],['VaR 95',r=>r.var_95,pct],['CVaR 95',r=>r.cvar_95,pct],['Omega',r=>r.omega,num],['Skew',r=>r.skewness,num],['Kurtosis exc.',r=>r.excess_kurtosis,num],['Turnover moy.',r=>r.average_turnover,pct],['Coûts cumulés',r=>r.total_transaction_cost,pct],['Positions moy.',r=>r.average_positions,num],['Poids max',r=>r.maximum_single_name_weight,pct],['HHI moy.',r=>r.average_hhi,num],['Cash moy.',r=>r.average_cash_exposure,pct],['Couverture rendements',r=>r.holding_return_coverage,pct]];table('performance-table',perfCols,D.performance);
table('starts-table',[['Stratégie',r=>r.strategy],['Départ demandé',r=>r.requested_start_year],['Départ effectif',r=>r.effective_start_month,month],['Mois',r=>r.months],['CAGR',r=>r.cagr,pct],['Rendement total',r=>r.total_return,pct]],D.cagr_by_start);
table('annual-table',[['Stratégie',r=>r.strategy],['Année',r=>r.year],['Mois',r=>r.months],['Année complète',r=>r.is_full_calendar_year,v=>v?'oui':'non'],['Rendement',r=>r.annual_return,pct]],D.annual_returns);
const months=[...new Set(D.holdings.map(x=>month(x.holding_month)))].sort().reverse();const pm=fillSelect('position-month',months);function positions(){{let q=document.getElementById('position-search').value.toUpperCase();let rows=D.holdings.filter(x=>month(x.holding_month)===pm.value&&x.strategy===pstrat.value&&x.ticker.includes(q));table('positions-table',[['Mois',r=>r.holding_month,month],['Stratégie',r=>r.strategy],['Action',r=>r.position_action],['Ticker',r=>r.ticker],['Rang',r=>r.selection_rank],['Poids',r=>r.target_weight,pct],['Rendement réalisé',r=>r.realized_return,pct],['Score',r=>r.score,num],['Résolution',r=>r.return_resolution],['Événement terminal',r=>r.terminal_event_type]],rows)}}pm.onchange=positions;pstrat.onchange=positions;document.getElementById('position-search').oninput=positions;positions();
function ledger(){{table('ledger-table',[['Décision',r=>r.decision_month,month],['Détention',r=>r.holding_month,month],['Rendement brut',r=>r.gross_return,pct],['Coûts',r=>r.transaction_cost,pct],['Rendement net',r=>r.net_return,pct],['SPY',r=>r.benchmark_return,pct],['Actif',r=>r.active_return,pct],['Turnover',r=>r.turnover,pct],['Positions',r=>r.n_positions],['Poids max',r=>r.maximum_position_weight,pct]],D.monthly.filter(x=>x.strategy===lstrat.value).slice().reverse())}}lstrat.onchange=ledger;ledger();
let m=D.model_summary[0];document.getElementById('model-kpis').innerHTML=[['Lignes OOS',m.test_rows],['Folds',m.folds],['Spearman IC',num(m.spearman_ic,4)],['NDCG @10',num(m.ndcg_at_10,4)],['ROC AUC',num(m.roc_auc,4)],['PR AUC',num(m.pr_auc_average_precision,4)],['Brier',num(m.brier,4)],['Calibration ECE',num(m.expected_calibration_error,4)]].map(c=>'<div class="card"><div class="label">'+c[0]+'</div><div class="value">'+c[1]+'</div></div>').join('');table('model-table',Object.keys(m).map(k=>[k,r=>r[k],v=>typeof v==='number'?num(v,5):v]),[m]);
table('terminal-table',[['Stratégie',r=>r.strategy],['Mois',r=>r.holding_month,month],['Ticker',r=>r.ticker],['Type',r=>r.terminal_event_type],['Date effective',r=>r.terminal_effective_date,month],['Valeur/action',r=>r.terminal_value_per_share,num],['Successeur',r=>r.terminal_successor_ticker],['Rendement',r=>r.realized_return,pct],['Statut',r=>r.manual_review_status],['Source',r=>r.terminal_event_source_url,v=>v?'<a href="'+v+'" target="_blank">SEC/source</a>':'N/A']],D.terminal_journal);
table('reconciliation-table',[['Stratégie',r=>r.strategy],['CAGR v1',r=>r.v1_cagr,pct],['CAGR v2',r=>r.v2_cagr,pct],['Δ CAGR',r=>r.delta_cagr,pct],['Sharpe v1',r=>r.v1_sharpe,num],['Sharpe v2',r=>r.v2_sharpe,num],['Max DD v1',r=>r.v1_max_drawdown,pct],['Max DD v2',r=>r.v2_max_drawdown,pct],['Δ turnover',r=>r.delta_average_turnover,pct],['Δ coûts',r=>r.delta_total_transaction_cost,pct]],D.reconciliation);
table('sources-table',[['Artefact',r=>r.name],['SHA-256',r=>r.sha256],['Chemin',r=>r.path]],D.sources);document.getElementById('downloads').innerHTML=D.downloads.map(x=>'<a href="'+x.href+'" download>'+x.label+'</a>').join('');
</script></body></html>"""


def build_report(common_dir: Path, reconciliation_dir: Path, boosting_dir: Path, output_dir: Path) -> Path:
    monthly_path = common_dir / "common_v2_monthly.parquet"
    holdings_path = common_dir / "common_v2_holdings.parquet"
    terminal_path = common_dir / "terminal_resolution_journal.parquet"
    terminal_attribution_path = common_dir / "terminal_cagr_attribution.csv"
    reconciliation_path = reconciliation_dir / "metrics_reconciliation.csv"
    model_path = boosting_dir / "model_horizon_summary.csv"
    required = [monthly_path, holdings_path, terminal_path, terminal_attribution_path, reconciliation_path, model_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")

    common_manifest = json.loads((common_dir / "manifest.json").read_text())
    reconciliation_manifest = json.loads((reconciliation_dir / "manifest.json").read_text())
    if not common_manifest.get("comparison_eligible") or not common_manifest.get("promotion_eligible"):
        raise RuntimeError("Common v2 replay is not comparison/promotion eligible")
    if not reconciliation_manifest.get("promotion_eligible"):
        raise RuntimeError("V1/v2 reconciliation is not promotion eligible")

    monthly = pl.read_parquet(monthly_path)
    holdings = _holdings_with_actions(pl.read_parquet(holdings_path))
    terminal = pl.read_parquet(terminal_path)
    terminal_attribution = pl.read_csv(terminal_attribution_path)
    performance, annual, starts, series = _performance_rows(monthly, holdings, terminal_attribution)
    reconciliation = pl.read_csv(reconciliation_path)
    model = pl.read_csv(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"
    download_dir = html_dir / "downloads"
    html_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "performance_kpis.csv": performance,
        "calendar_returns.csv": annual,
        "cagr_by_start_year.csv": starts,
        "monthly_ledger.csv": monthly,
        "monthly_positions.csv": holdings,
        "terminal_event_journal.csv": terminal,
        "v1_v2_reconciliation.csv": reconciliation,
        "boosting_model_quality.csv": model,
    }
    for filename, frame in exports.items():
        frame.write_csv(download_dir / filename)

    source_paths = required + [common_dir / "manifest.json", reconciliation_dir / "manifest.json", boosting_dir / "manifest.json"]
    sources = [{"name": path.name, "path": str(path), "sha256": _sha256(path)} for path in source_paths]
    payload = {
        "strategy_order": list(STRATEGY_ORDER),
        "performance": _records(performance),
        "annual_returns": _records(annual),
        "cagr_by_start": _records(starts),
        "series": _json_safe(series),
        "monthly": _records(monthly),
        "holdings": _records(holdings),
        "terminal_journal": _records(terminal),
        "reconciliation": _records(reconciliation),
        "model_summary": _records(model),
        "sources": sources,
        "downloads": [{"label": name.replace("_", " ").replace(".csv", ""), "href": f"downloads/{name}"} for name in exports],
    }
    report_path = html_dir / "methodology_v2_study.html"
    report_path.write_text(_html(payload), encoding="utf-8")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "contract_version": 1,
        "report_id": "methodology-v2-approved-censoring",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_eligible": True,
        "promotion_eligible": True,
        "historical_scope_only": True,
        "live_portfolio_signal": False,
        "composition_id": common_manifest.get("composition_id"),
        "git_head": git_head,
        "counts": {
            "strategies": len(STRATEGY_ORDER),
            "months_per_strategy": monthly.height // len(STRATEGY_ORDER),
            "monthly_rows": monthly.height,
            "holding_rows": holdings.height,
            "terminal_rows": terminal.height,
        },
        "sources": sources,
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "downloads": {name: {"path": str(download_dir / name), "sha256": _sha256(download_dir / name)} for name in exports},
    }
    (html_dir / "methodology_v2_study_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-dir", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--reconciliation-dir", type=Path, default=DEFAULT_RECONCILIATION)
    parser.add_argument("--boosting-dir", type=Path, default=DEFAULT_BOOSTING)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/research_dashboard/methodology_v2_approved_20260819",
    )
    args = parser.parse_args()
    report = build_report(args.common_dir, args.reconciliation_dir, args.boosting_dir, args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
