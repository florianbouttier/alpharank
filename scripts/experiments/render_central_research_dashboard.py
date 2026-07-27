#!/usr/bin/env python3
"""Build the self-contained AlphaRank research and production dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPN_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/"
    "legacy_ema_top5_vs_top10_quarantine_v7_20260726"
)
CHAMPION_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/"
    "legacy_ema_long_history_ticker_quarantine_v6_20260726"
)
RISK_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/"
    "legacy_ema_risk_overlay_ticker_quarantine_v6_20260726"
)
SCREENING_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/screening_clean_20260725"
)
EMA_SCREENING_DIR = PROJECT_ROOT / (
    "outputs/multihorizon_boosting/legacy_winners_pit_ema_only_20260725"
)
LIVE_DIR = PROJECT_ROOT / (
    "outputs/live_alpha/"
    "ema_classification_h6_202606_20260727_production_candidate_v3"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/research_dashboard/"
    "legacy_ema_alpha_central_20260727"
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    return _clean(pl.read_csv(path, try_parse_dates=True).to_dicts())


def _parquet_records(path: Path) -> list[dict[str, Any]]:
    return _clean(pl.read_parquet(path).to_dicts())


def _monthly_shap_payload(
    samples_path: Path,
    lexicon_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    lexicon = pl.read_csv(lexicon_path).sort("importance_rank")
    features = lexicon["feature"].to_list()
    samples = pl.read_parquet(samples_path).sort(
        ["decision_month", "ticker"]
    )
    payload: list[dict[str, Any]] = []
    for row in samples.iter_rows(named=True):
        payload.append(
            {
                "m": _clean(row["decision_month"]),
                "t": row["ticker"],
                "f": int(row["fold"]),
                "v": [_clean(row.get(f"value__{feature}")) for feature in features],
                "s": [_clean(row.get(f"shap__{feature}")) for feature in features],
            }
        )
    return payload, _clean(lexicon.to_dicts()), features


def _drawdowns(monthly: pl.DataFrame) -> list[dict[str, Any]]:
    columns = {
        "alpha_top5_return": "Top 5 égal",
        "alpha_top10_return": "Top 10 égal",
        "legacy_return": "Legacy",
        "spy_return": "SPY",
    }
    wealth = {label: 1.0 for label in columns.values()}
    peak = wealth.copy()
    rows: list[dict[str, Any]] = []
    for row in monthly.sort("holding_month").iter_rows(named=True):
        item: dict[str, Any] = {"month": _clean(row["holding_month"])}
        for column, label in columns.items():
            wealth[label] *= 1.0 + float(row[column])
            peak[label] = max(peak[label], wealth[label])
            item[label] = {
                "wealth": round(wealth[label], 8),
                "drawdown": round(wealth[label] / peak[label] - 1.0, 8),
            }
        rows.append(item)
    return rows


def _regimes(monthly: pl.DataFrame) -> list[dict[str, Any]]:
    periods = [
        ("2011–2015", date(2011, 1, 1), date(2015, 12, 31)),
        ("2016–2020", date(2016, 1, 1), date(2020, 12, 31)),
        ("2021–2025", date(2021, 1, 1), date(2025, 12, 31)),
    ]
    columns = {
        "alpha_top5_return": "Top 5 égal",
        "alpha_top10_return": "Top 10 égal",
        "legacy_return": "Legacy",
        "spy_return": "SPY",
    }
    output: list[dict[str, Any]] = []
    for label, start, end in periods:
        frame = monthly.filter(
            pl.col("holding_month").is_between(start, end)
        )
        for column, series in columns.items():
            values = frame[column].to_list()
            wealth = 1.0
            peak = 1.0
            max_drawdown = 0.0
            for value in values:
                wealth *= 1.0 + float(value)
                peak = max(peak, wealth)
                max_drawdown = min(max_drawdown, wealth / peak - 1.0)
            years = len(values) / 12
            cagr = wealth ** (1 / years) - 1 if years else None
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values)
                / max(1, len(values) - 1)
            )
            volatility = math.sqrt(variance * 12)
            output.append(
                {
                    "period": label,
                    "series": series,
                    "months": len(values),
                    "cagr": cagr,
                    "volatility": volatility,
                    "sharpe": (cagr - 0.02) / volatility
                    if volatility
                    else None,
                    "max_drawdown": max_drawdown,
                }
            )
    return _clean(output)


def _live_payload() -> dict[str, Any]:
    manifest = json.loads((LIVE_DIR / "manifest.json").read_text())
    return {
        "manifest": _clean(manifest),
        "top5": _records(LIVE_DIR / "portfolio_top5.csv"),
        "top10": _records(LIVE_DIR / "portfolio_top10.csv"),
        "legacy": _records(
            LIVE_DIR / "legacy_portfolio_same_holding_month.csv"
        ),
    }


def build_payload() -> tuple[dict[str, Any], list[Path]]:
    source_files = [
        TOPN_DIR / "monthly_portfolios.parquet",
        TOPN_DIR / "monthly_portfolio_returns.csv",
        TOPN_DIR / "performance_legacy_convention.csv",
        TOPN_DIR / "annual_returns_wide.csv",
        TOPN_DIR / "cost_sensitivity.csv",
        TOPN_DIR / "paired_block_bootstrap.csv",
        TOPN_DIR / "promotion_gates.csv",
        TOPN_DIR / "rank_bucket_diagnostics.csv",
        TOPN_DIR / "alpha_shap_feature_lexicon.csv",
        CHAMPION_DIR / "classification_h06/shap_samples.parquet",
        CHAMPION_DIR / "model_horizon_summary.csv",
        SCREENING_DIR / "model_horizon_summary.csv",
        EMA_SCREENING_DIR / "model_horizon_summary.csv",
        RISK_DIR / "risk_model_metrics.csv",
        RISK_DIR / "allocation_performance_legacy_convention.csv",
        RISK_DIR / "allocation_acceptance_gates.csv",
        LIVE_DIR / "manifest.json",
        LIVE_DIR / "portfolio_top5.csv",
        LIVE_DIR / "portfolio_top10.csv",
        LIVE_DIR / "legacy_portfolio_same_holding_month.csv",
    ]
    missing = [path for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dashboard sources:\n" + "\n".join(map(str, missing))
        )
    shap, lexicon, features = _monthly_shap_payload(
        CHAMPION_DIR / "classification_h06/shap_samples.parquet",
        TOPN_DIR / "alpha_shap_feature_lexicon.csv",
    )
    monthly = pl.read_csv(
        TOPN_DIR / "monthly_portfolio_returns.csv",
        try_parse_dates=True,
    )
    payload = {
        "meta": {
            "created": datetime.now().astimezone().isoformat(),
            "test_start": str(monthly["holding_month"].min()),
            "test_end": str(monthly["holding_month"].max()),
            "test_months": monthly.height,
            "shap_rows": len(shap),
            "shap_features": len(features),
            "shap_months": len({row["m"] for row in shap}),
        },
        "performance": _records(
            TOPN_DIR / "performance_legacy_convention.csv"
        ),
        "monthly": _records(TOPN_DIR / "monthly_portfolio_returns.csv"),
        "wealth": _drawdowns(monthly),
        "regimes": _regimes(monthly),
        "annual": _records(TOPN_DIR / "annual_returns_wide.csv"),
        "holdings": _parquet_records(TOPN_DIR / "monthly_portfolios.parquet"),
        "costs": _records(TOPN_DIR / "cost_sensitivity.csv"),
        "bootstrap": _records(TOPN_DIR / "paired_block_bootstrap.csv"),
        "promotion": _records(TOPN_DIR / "promotion_gates.csv"),
        "buckets": _records(TOPN_DIR / "rank_bucket_diagnostics.csv"),
        "screening": _records(SCREENING_DIR / "model_horizon_summary.csv"),
        "ema_screening": _records(
            EMA_SCREENING_DIR / "model_horizon_summary.csv"
        ),
        "champion": _records(CHAMPION_DIR / "model_horizon_summary.csv"),
        "risk_models": _records(RISK_DIR / "risk_model_metrics.csv"),
        "risk_performance": _records(
            RISK_DIR / "allocation_performance_legacy_convention.csv"
        ),
        "risk_gates": _records(
            RISK_DIR / "allocation_acceptance_gates.csv"
        ),
        "shap": shap,
        "lexicon": lexicon,
        "features": features,
        "live": _live_payload(),
    }
    return payload, source_files


HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank — Centre de recherche Legacy & Boosting</title>
<style>
:root{--bg:#F5F6FA;--panel:#FFF;--ink:#141927;--muted:#687083;--line:#DDE1EA;--navy:#111D55;--navy2:#26387E;--gold:#9B8816;--green:#16794B;--red:#B03A45;--bluewash:#EEF1FB;--goldwash:#F7F3DD;--shadow:0 1px 2px rgba(17,29,85,.04)}
[data-theme="dark"]{--bg:#11141D;--panel:#1A1E29;--ink:#F2F3F7;--muted:#9EA6B7;--line:#313746;--navy:#8998E2;--navy2:#A7B2ED;--gold:#D6C45B;--green:#55B888;--red:#EC7A83;--bluewash:#20283E;--goldwash:#2B291B}
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.48 "IBM Plex Sans",sans-serif}
button,select,input{font:inherit;color:inherit}.mono,.metric strong,td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.shell{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;background:var(--navy);color:#fff;padding:24px 16px;display:flex;flex-direction:column}
.brand{font:600 18px "IBM Plex Mono";letter-spacing:-.03em;padding:0 10px 24px;border-bottom:1px solid rgba(255,255,255,.18)}.brand small{display:block;font:500 10px "IBM Plex Sans";opacity:.65;letter-spacing:.12em;text-transform:uppercase;margin-top:5px}
.tabs{display:grid;gap:4px;margin-top:20px}.tab{border:0;background:transparent;color:rgba(255,255,255,.72);text-align:left;padding:10px 12px;border-radius:4px;cursor:pointer;font-weight:600}.tab:hover,.tab.active{background:rgba(255,255,255,.13);color:#fff}.tab span{font-family:"IBM Plex Mono";font-size:10px;opacity:.55;margin-right:8px}
.aside-foot{margin-top:auto;padding:14px 10px 0;border-top:1px solid rgba(255,255,255,.18);font-size:11px;color:rgba(255,255,255,.65)}
main{min-width:0}.topbar{height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:10}.topbar .crumb{color:var(--muted)}.topbar button{border:1px solid var(--line);background:var(--panel);padding:7px 10px;border-radius:4px;cursor:pointer}
.page{display:none;padding:28px;max-width:1500px;margin:auto}.page.active{display:block}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.11em;color:var(--navy);font-weight:700}.hero h1{font-size:34px;line-height:1.08;letter-spacing:-.035em;margin:7px 0 10px}.hero p{color:var(--muted);max-width:1000px;font-size:16px}.hero{margin-bottom:22px}
.grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.panel{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);padding:18px}.panel h2,.panel h3{margin:0 0 12px;letter-spacing:-.02em}.panel h2{font-size:19px}.panel h3{font-size:15px}.panel p{color:var(--muted)}.metric{min-height:116px}.metric span{font-size:12px;color:var(--muted)}.metric strong{font-size:27px;display:block;margin:7px 0 3px;letter-spacing:-.04em}.metric small{color:var(--muted)}
.callout{border-left:4px solid var(--navy);background:var(--bluewash);padding:14px 16px;margin:14px 0}.callout.warn{border-color:var(--gold);background:var(--goldwash)}.callout.bad{border-color:var(--red)}.callout strong{display:block;margin-bottom:3px}
.section-head{display:flex;justify-content:space-between;align-items:end;gap:16px;margin:30px 0 12px}.section-head h2{font-size:22px;margin:0}.section-head p{margin:4px 0 0;color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.controls label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}select,input{border:1px solid var(--line);background:var(--panel);padding:8px 10px;border-radius:3px}
.table-wrap{overflow:auto;border:1px solid var(--line);background:var(--panel)}table{width:100%;border-collapse:collapse;min-width:780px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);z-index:1;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}td.num{text-align:right}tr:last-child td{border-bottom:0}.badge{display:inline-block;font:600 10px "IBM Plex Mono";padding:3px 6px;border:1px solid currentColor;border-radius:2px}.pass{color:var(--green)}.fail{color:var(--red)}.gold{color:var(--gold)}.navy{color:var(--navy)}
.chart{height:330px;position:relative}.chart svg{width:100%;height:100%;display:block}.chart text{fill:var(--muted);font:10px "IBM Plex Mono"}.chart .gridline{stroke:var(--line);stroke-width:1}.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:12px}.legend i{width:10px;height:10px;display:inline-block;margin-right:5px}
.portfolio-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.portfolio-card{border-top:4px solid var(--navy)}.portfolio-card.legacy{border-color:var(--gold)}.ticker-row{display:grid;grid-template-columns:54px 1fr auto;gap:8px;padding:9px 0;border-bottom:1px solid var(--line);align-items:center}.ticker-row:last-child{border-bottom:0}.ticker{font:600 12px "IBM Plex Mono"}.action{font:600 9px "IBM Plex Mono";padding:2px 4px;border:1px solid currentColor}.enter{color:var(--green)}.exit{color:var(--red)}.keep{color:var(--muted)}
.shap-layout{display:grid;grid-template-columns:minmax(340px,.9fr) minmax(500px,1.4fr);gap:14px}.shap-bar{display:grid;grid-template-columns:minmax(150px,1fr) 4fr 72px;gap:8px;align-items:center;margin:7px 0;font:11px "IBM Plex Mono"}.bar-track{height:10px;background:var(--bluewash)}.bar-fill{height:100%;background:var(--navy)}canvas{width:100%;height:400px;border:1px solid var(--line);background:var(--panel)}
.two-col-doc{columns:2 340px;column-gap:32px}.doc-block{break-inside:avoid;margin-bottom:18px}.doc-block h3{margin:0 0 5px}.doc-block p,.doc-block li{color:var(--muted)}code{font-family:"IBM Plex Mono";font-size:11px;color:var(--navy)}
.source-list{font:11px "IBM Plex Mono";word-break:break-all}.fine{font-size:11px;color:var(--muted)}.spacer{height:8px}
@media(max-width:1100px){.g4{grid-template-columns:repeat(2,1fr)}.g3,.portfolio-cards{grid-template-columns:1fr 1fr}.shap-layout{grid-template-columns:1fr}}
@media(max-width:760px){.shell{display:block}aside{position:sticky;height:auto;padding:12px;z-index:30}.brand{padding:2px 5px 10px}.tabs{display:flex;overflow:auto;margin-top:8px}.tab{white-space:nowrap;padding:8px}.aside-foot{display:none}.topbar{top:104px;height:52px;padding:0 14px}.page{padding:18px 12px}.hero h1{font-size:28px}.g4,.g3,.g2,.portfolio-cards{grid-template-columns:1fr}.chart{height:270px}.two-col-doc{columns:1}.section-head{align-items:start;flex-direction:column}}
</style>
</head>
<body>
<div class="shell">
<aside>
  <div class="brand">ALPHARANK<small>Research control center</small></div>
  <nav class="tabs">
    <button class="tab active" data-page="overview"><span>01</span>Synthèse</button>
    <button class="tab" data-page="models"><span>02</span>Modèles & horizons</button>
    <button class="tab" data-page="backtest"><span>03</span>Backtest approfondi</button>
    <button class="tab" data-page="portfolios"><span>04</span>Actions par mois</button>
    <button class="tab" data-page="shap"><span>05</span>SHAP mensuel</button>
    <button class="tab" data-page="risk"><span>06</span>Risque</button>
    <button class="tab" data-page="live"><span>07</span>Production 2026</button>
    <button class="tab" data-page="docs"><span>08</span>Documentation & audit</button>
  </nav>
  <div class="aside-foot">Une source unique de lecture.<br>Résultats hors-échantillon, lineage et limites.</div>
</aside>
<main>
<header class="topbar"><div class="crumb">Legacy EMA / Boosting / <b id="crumb">Synthèse</b></div><button id="theme">◐ Thème</button></header>

<section class="page active" id="overview">
 <div class="hero"><div class="eyebrow">Décision actuelle · recherche non promue automatiquement</div><h1>Le boosting produit un alpha fort, mais la preuve reste conditionnelle.</h1><p>Ce centre réunit le screening, le champion classification à 6 mois, les portefeuilles mensuels, les explications SHAP, le risque et le candidat live. Tous les chiffres comparatifs utilisent le même intervalle août 2011–novembre 2025.</p></div>
 <div class="grid g4" id="headline-metrics"></div>
 <div class="callout warn"><strong>Conclusion de gouvernance</strong>Top 5 égal reste la meilleure allocation testée, mais aucun overlay risque ni Top 10 ne passe tous ses critères de promotion. Le live de juillet 2026 est un candidat opérationnel, pas une nouvelle preuve hors-échantillon.</div>
 <div class="section-head"><div><h2>Ce qui est démontré</h2><p>Lecture synthétique, sans confondre performance et validité causale.</p></div></div>
 <div class="grid g3">
  <article class="panel"><h3>Signal</h3><p>La classification de surperformance à 6 mois, basée uniquement sur les EMA gagnantes de Legacy et leurs transformations mensuelles, est le champion retenu.</p></article>
  <article class="panel"><h3>Allocation</h3><p>Les cinq meilleurs scores, équipondérés à 20 %, dominent Top 10, Legacy et SPY sur ce backtest. Les rangs 6–10 diluent nettement le signal.</p></article>
  <article class="panel"><h3>Risque</h3><p>Les têtes volatilité/downside sont informatives, mais leur utilisation en pondération ne satisfait pas les garde-fous complets.</p></article>
 </div>
 <div class="section-head"><div><h2>Chaîne de décision</h2><p>Du modèle à l'action, puis au contrôle.</p></div></div>
 <div class="grid g4">
  <article class="panel"><div class="eyebrow">1 · Features</div><h3>EMA relatives Legacy</h3><p>35 paires gagnantes, ratio action/SPY, puis ratio brut, rang, z-score et quartiles.</p></article>
  <article class="panel"><div class="eyebrow">2 · Modèle</div><h3>XGBoost classification H6</h3><p>Probabilité qu'une action soit dans le décile supérieur de surperformance future à six mois.</p></article>
  <article class="panel"><div class="eyebrow">3 · Portefeuille</div><h3>Top 5 égal</h3><p>Classement par score alpha brut, cinq titres, 20 % chacun, rebalancement mensuel.</p></article>
  <article class="panel"><div class="eyebrow">4 · Contrôle</div><h3>Legacy + SPY</h3><p>CAGR, Sharpe Legacy, drawdown, pire année, coûts et bootstrap apparié.</p></article>
 </div>
</section>

<section class="page" id="models">
 <div class="hero"><div class="eyebrow">Cadrage expérimental</div><h1>Quel objectif et quel horizon reproduisent le mieux Legacy ?</h1><p>Classification, régression, ranking et teacher ont été comparés sur 1, 3, 6, 12, 24 et 36 mois. Le tableau permet de basculer entre le screening large et la version EMA-only demandée.</p></div>
 <div class="controls"><label>Jeu d'entrée</label><select id="model-dataset"><option value="ema">EMA gagnantes Legacy</option><option value="broad">Screening large initial</option></select><label>Méthode</label><select id="model-method"><option value="all">Toutes</option></select></div>
 <div class="callout"><strong>Champion retenu : classification, horizon 6 mois</strong>Le choix final ne repose pas sur le seul ROC AUC : PR AUC, lift sur la prévalence, rendement Top 5 à un mois, overlap Legacy, calibration et stabilité temporelle sont lus ensemble.</div>
 <div class="table-wrap"><table><thead><tr><th>Méthode</th><th>Horizon</th><th>Folds</th><th>Rows test</th><th>ROC AUC</th><th>PR AUC</th><th>Lift PR</th><th>Spearman</th><th>NDCG@10</th><th>RMSE norm.</th><th>Top5 excès H</th><th>Top5 excès 1m</th><th>Overlap Legacy</th></tr></thead><tbody id="model-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Lecture des tâches</h2></div></div>
 <div class="grid g4">
  <article class="panel"><h3>Classification</h3><p>Classe positive = décile supérieur de surperformance future contre SPY. Mesures : ROC AUC, PR AUC, lift, Brier et log-loss.</p></article>
  <article class="panel"><h3>Régression</h3><p>Prévoit directement la surperformance future. Mesures : RMSE, RMSE normalisée, MAE, R² et IC mensuel.</p></article>
  <article class="panel"><h3>Ranking</h3><p>Optimise l'ordre relatif par mois. Mesures : NDCG@5/10/20 et lift contre l'absence de signal.</p></article>
  <article class="panel"><h3>Teacher</h3><p>Essaie de reproduire le choix Legacy. Utile comme diagnostic d'imitation, insuffisant pour prouver l'alpha futur.</p></article>
 </div>
</section>

<section class="page" id="backtest">
 <div class="hero"><div class="eyebrow">Même calendrier · mêmes conventions</div><h1>Backtest fin contre Legacy et SPY</h1><p>172 mois de détention, août 2011–novembre 2025. Les stratégies ML incluent 10 pb × turnover. Sharpe Legacy = (CAGR − 2 %) / volatilité annualisée.</p></div>
 <div class="grid g4" id="backtest-metrics"></div>
 <div class="section-head"><div><h2>Richesse cumulée</h2><p>Base 1 au début du test.</p></div></div>
 <article class="panel"><div class="legend" id="wealth-legend"></div><div class="chart" id="wealth-chart"></div></article>
 <div class="section-head"><div><h2>Drawdowns</h2><p>Baisse depuis le plus haut historique de chaque méthode.</p></div></div>
 <article class="panel"><div class="chart" id="dd-chart"></div></article>
 <div class="section-head"><div><h2>Performance complète</h2><p>Toutes les méthodes d'allocation, puis décomposition par régime.</p></div></div>
 <div class="table-wrap"><table><thead><tr><th>Méthode</th><th>CAGR</th><th>Sharpe</th><th>Vol.</th><th>Max DD</th><th>Pire année</th><th>Turnover</th><th>Poids max</th><th>Secteur max</th></tr></thead><tbody id="perf-rows"></tbody></table></div>
 <div class="spacer"></div>
 <div class="table-wrap"><table><thead><tr><th>Période</th><th>Méthode</th><th>Mois</th><th>CAGR</th><th>Vol.</th><th>Sharpe</th><th>Max DD</th></tr></thead><tbody id="regime-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Rendements par année</h2><p>* année partielle.</p></div></div>
 <div class="table-wrap"><table id="annual-table"></table></div>
 <div class="section-head"><div><h2>Robustesse et dilution</h2></div></div>
 <div class="grid g2">
  <article class="panel"><h3>Bootstrap apparié, blocs de 12 mois</h3><div id="bootstrap-summary"></div><p class="fine">2 000 réplications. L'intervalle préserve mieux l'autocorrélation qu'un bootstrap mensuel naïf.</p></article>
  <article class="panel"><h3>Rangs 1–5 contre 6–10</h3><div id="bucket-summary"></div></article>
  <article class="panel"><h3>Sensibilité aux coûts</h3><div id="cost-summary"></div></article>
  <article class="panel"><h3>Garde-fous Top 10</h3><div id="promotion-summary"></div></article>
 </div>
</section>

<section class="page" id="portfolios">
 <div class="hero"><div class="eyebrow">Audit mensuel des décisions</div><h1>Quelles actions sont détenues chaque mois ?</h1><p>Choisissez un mois de détention. Les badges ENTER / KEEP sont calculés par rapport au mois précédent ; les sorties sont listées séparément. Les rendements affichés sont réalisés sur le mois de détention.</p></div>
 <div class="controls"><label>Mois de détention</label><select id="holding-month"></select><span id="month-returns" class="mono"></span></div>
 <div class="portfolio-cards" id="portfolio-cards"></div>
</section>

<section class="page" id="shap">
 <div class="hero"><div class="eyebrow">Explication hors-échantillon</div><h1>SHAP filtré par mois de test</h1><p>Chaque filtre mensuel ne conserve que les observations SHAP appartenant au mois de décision sélectionné. L'ordre des variables et les graphiques sont recalculés sur ce sous-échantillon.</p></div>
 <div class="callout warn"><strong>Point de méthode essentiel</strong>Le backtest ne réentraîne pas tous les mois : un modèle est ajusté une fois par fold, puis utilisé sur son bloc test (généralement 12 mois). Le filtre mensuel explique donc un mois hors-échantillon sous le modèle de son fold. Le live, lui, est réentraîné à chaque exécution mensuelle.</div>
 <div class="controls"><label>Mois de décision</label><select id="shap-month"><option value="all">Tous les mois</option></select><label>Variable individuelle</label><select id="shap-feature"></select></div>
 <div class="grid g4" id="shap-metrics"></div>
 <div class="shap-layout">
  <article class="panel"><h2>Importance |SHAP| décroissante</h2><p class="fine">Recalculée pour le mois sélectionné. Unité : marge brute XGBoost / log-odds, avant calibration isotone.</p><div id="shap-bars"></div></article>
  <article class="panel"><h2>Beeswarm — Top 15</h2><p class="fine">Axe horizontal = contribution SHAP ; couleur = valeur de la variable, faible → élevée.</p><canvas id="beeswarm" width="1000" height="620"></canvas></article>
 </div>
 <div class="section-head"><div><h2 id="individual-title">Analyse individuelle</h2><p>Valeur de la variable contre contribution SHAP.</p></div></div>
 <canvas id="individual" width="1200" height="500"></canvas>
 <div class="section-head"><div><h2>Détail des observations</h2><p id="shap-detail-note"></p></div></div>
 <div class="table-wrap"><table><thead><tr><th>Mois décision</th><th>Ticker</th><th>Fold</th><th>Valeur</th><th>SHAP</th></tr></thead><tbody id="shap-detail"></tbody></table></div>
 <div class="section-head"><div><h2>Lexique exact des 185 variables</h2></div><div class="controls"><input id="lexicon-search" placeholder="Rechercher une variable…"></div></div>
 <div class="table-wrap"><table><thead><tr><th>Rang global</th><th>Variable</th><th>|SHAP| global</th><th>Transformation</th><th>Spans EMA</th><th>Unité</th><th>Définition exacte</th><th>Interprétation</th></tr></thead><tbody id="lexicon-rows"></tbody></table></div>
</section>

<section class="page" id="risk">
 <div class="hero"><div class="eyebrow">Deuxième tête · contrôle du risque</div><h1>Volatilité, downside et allocation</h1><p>Les têtes de risque sont entraînées uniquement sur des rendements quotidiens strictement futurs. Elles servent à tester des poids inverse-volatilité ou inverse-downside sur la même sélection alpha.</p></div>
 <div class="callout bad"><strong>Aucun overlay risque n'est promu</strong>Certaines variantes améliorent le Sharpe, mais aucune ne passe simultanément le drawdown, le coût, la perte de CAGR et, le cas échéant, la concentration sectorielle.</div>
 <div class="section-head"><div><h2>Qualité prédictive des têtes risque</h2></div></div>
 <div class="table-wrap"><table><thead><tr><th>Tête</th><th>Horizon</th><th>Tâche</th><th>Target</th><th>Rows test</th><th>Spearman</th><th>RMSE</th><th>RMSE norm.</th><th>MAE</th><th>ROC AUC</th><th>PR AUC</th><th>Lift PR</th></tr></thead><tbody id="risk-model-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Allocations risque vs références</h2></div></div>
 <div class="table-wrap"><table><thead><tr><th>Méthode</th><th>CAGR</th><th>Vol.</th><th>Sharpe</th><th>Max DD</th><th>Pire année</th></tr></thead><tbody id="risk-perf-rows"></tbody></table></div>
 <div class="section-head"><div><h2>Garde-fous d'acceptation</h2></div></div>
 <div id="risk-gates" class="grid g2"></div>
</section>

<section class="page" id="live">
 <div class="hero"><div class="eyebrow">Candidat de production · juillet 2026</div><h1>Portefeuille calculé sur les prix disponibles fin juin</h1><p>Le modèle est réentraîné sur l'historique causal disponible, calibré sur juillet–décembre 2025, puis score l'univers connu du mois de détention juillet 2026.</p></div>
 <div class="grid g4" id="live-metrics"></div>
 <div class="callout warn"><strong>Ce n'est pas un test scellé</strong>Les métriques affichées sont celles de la fenêtre de validation 2025-07 à 2025-12. Le portefeuille juillet 2026 est une décision live, sans rendement futur encore utilisé.</div>
 <div class="portfolio-cards" id="live-portfolios"></div>
 <div class="section-head"><div><h2>Lineage de production</h2></div></div>
 <div class="table-wrap"><table><tbody id="live-lineage"></tbody></table></div>
</section>

<section class="page" id="docs">
 <div class="hero"><div class="eyebrow">Référence centrale</div><h1>Méthode, biais, conventions et actions</h1><p>Cette page documente ce qu'il faut savoir pour interpréter, reproduire et faire évoluer les résultats sans fuite de données.</p></div>
 <div class="two-col-doc">
  <div class="doc-block"><h3>Legacy</h3><p>Méthode déterministe fondée sur des paires d'EMA relatives action/SPY sélectionnées historiquement, puis sur un portefeuille concentré. Elle est toujours conservée comme référence côte à côte.</p></div>
  <div class="doc-block"><h3>Features exactes</h3><p>Pour chaque paire (a,b) : <code>EMA_a(P_action/P_SPY) / EMA_b(P_action/P_SPY)</code>, puis ratio brut, rang centile mensuel, z-score cross-sectionnel et indicateurs top/bottom quartile. Aucune fondamentale dans le champion présenté.</p></div>
  <div class="doc-block"><h3>Target alpha H6</h3><p>Classe positive si la surperformance cumulée future contre SPY à six mois appartient au décile supérieur du mois. Le modèle classe les actions ; la probabilité calibrée sert à l'interprétation, le score brut au ranking.</p></div>
  <div class="doc-block"><h3>Géométrie temporelle</h3><p>En historique : entraînement ancien, validation/calibration suivante, puis bloc test chronologique. Le modèle est réentraîné entre les folds, pas entre les mois d'un même bloc. Le dernier fold peut être plus court.</p></div>
  <div class="doc-block"><h3>Anti-fuite</h3><ul><li>Features datées au mois de décision.</li><li>Labels entièrement mûrs avant la fin du train.</li><li>Calibration apprise avant le test.</li><li>Univers et exclusions explicités.</li><li>Comparaisons sur mois communs.</li><li>Prix incohérents mis en quarantaine sur toute leur trajectoire.</li></ul></div>
  <div class="doc-block"><h3>SHAP</h3><p>Les valeurs sont calculées sur un échantillon sauvegardé de 80 lignes par fold, soit 1 200 lignes. Un mois ne contient donc que 1 à 22 observations : l'analyse mensuelle est fidèle à l'échantillon OOS, mais ne représente pas la cross-section complète.</p></div>
  <div class="doc-block"><h3>Risque</h3><p>Volatilité réalisée = écart-type annualisé des rendements quotidiens futurs. Downside = racine annualisée de la moyenne des carrés des rendements négatifs futurs. Classe high-vol = top 20 % cross-sectionnel de volatilité future.</p></div>
  <div class="doc-block"><h3>Coûts et Sharpe</h3><p>Le backtest principal applique 10 pb multipliés par le turnover. La sensibilité couvre plusieurs coûts. Le Sharpe dit « Legacy » emploie un taux sans risque annuel fixe de 2 % et le CAGR au numérateur.</p></div>
  <div class="doc-block"><h3>Ce qu'il ne faut pas conclure</h3><p>Une forte performance ne prouve pas que le signal survivra hors de la période étudiée. Multiple testing, révisions d'univers, survivorship historique imparfait et faible nombre de régimes restent des risques.</p></div>
  <div class="doc-block"><h3>Décision recommandée</h3><p>Conserver Top 5 égal comme challenger live surveillé, Legacy comme contrôle et SPY comme benchmark. Ne promouvoir Top 10 ou un overlay risque qu'après passage des gates sur une nouvelle période réellement scellée.</p></div>
 </div>
 <div class="section-head"><div><h2>Sources et intégrité</h2><p>Le manifeste adjacent contient les empreintes SHA-256 de chaque source.</p></div></div>
 <div class="source-list" id="sources"></div>
</section>
</main></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("payload").textContent);
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const pct=(v,d=1)=>v==null?"—":(100*v).toFixed(d)+" %";
const num=(v,d=2)=>v==null?"—":Number(v).toFixed(d);
const money=(v)=>v==null?"—":"$"+Number(v).toFixed(2);
const labels={alpha_top5_equal:"Top 5 égal",alpha_top10_equal:"Top 10 égal","SPY total return":"SPY",Legacy:"Legacy"};
const color={"Top 5 égal":"#111D55","Top 10 égal":"#6071B5","Legacy":"#9B8816","SPY":"#657083"};
function metric(label,value,note){return `<article class="panel metric"><span>${label}</span><strong>${value}</strong><small>${note||""}</small></article>`}
function badge(ok){return `<span class="badge ${ok?"pass":"fail"}">${ok?"PASS":"FAIL"}</span>`}
function td(v,cls="num"){return `<td class="${cls}">${v}</td>`}

$$(".tab").forEach(b=>b.onclick=()=>{$$(".tab,.page").forEach(x=>x.classList.remove("active"));b.classList.add("active");$("#"+b.dataset.page).classList.add("active");$("#crumb").textContent=b.textContent.trim().replace(/^\\d+/,"");window.scrollTo(0,0);if(b.dataset.page==="backtest"){drawLine("wealth-chart","wealth");drawLine("dd-chart","drawdown")}if(b.dataset.page==="shap") renderShap()});
$("#theme").onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==="dark"?"":"dark";if($("#shap").classList.contains("active"))renderShap();if($("#backtest").classList.contains("active")){drawLine("wealth-chart","wealth");drawLine("dd-chart","drawdown")}};

const perfBy=s=>D.performance.find(x=>x.series===s);
const p5=perfBy("alpha_top5_equal"),p10=perfBy("alpha_top10_equal"),leg=perfBy("Legacy"),spy=perfBy("SPY total return");
$("#headline-metrics").innerHTML=metric("Top 5 · CAGR",pct(p5.cagr),"net 10 pb × turnover")+metric("Legacy · CAGR",pct(leg.cagr),"même intervalle")+metric("SPY · CAGR",pct(spy.cagr),"total return")+metric("Top 5 · Sharpe",num(p5.sharpe),"Legacy convention");
$("#backtest-metrics").innerHTML=metric("Top 5 · CAGR",pct(p5.cagr),`Δ Legacy ${pct(p5.cagr-leg.cagr)}`)+metric("Top 5 · Sharpe",num(p5.sharpe),`Legacy ${num(leg.sharpe)}`)+metric("Top 5 · Max DD",pct(p5.max_drawdown),`Legacy ${pct(leg.max_drawdown)}`)+metric("Top 10 · CAGR",pct(p10.cagr),`Δ Top 5 ${pct(p10.cagr-p5.cagr)}`);

function modelRows(){
 const rows=$("#model-dataset").value==="ema"?D.ema_screening:D.screening, method=$("#model-method").value;
 $("#model-rows").innerHTML=rows.filter(r=>method==="all"||r.method===method).map(r=>`<tr><td>${r.method}${r.method==="classification"&&r.horizon===6?' <span class="badge pass">CHAMPION</span>':""}</td>${td(r.horizon+"m")}${td(r.folds)}${td(r.test_rows)}${td(num(r.roc_auc,3))}${td(num(r.pr_auc_average_precision,3))}${td(num(r.pr_auc_lift_vs_prevalence,2))}${td(num(r.spearman_ic,3))}${td(num(r.ndcg_at_10,3))}${td(num(r.normalized_rmse,3))}${td(pct(r.top5_horizon_excess))}${td(pct(r.top5_one_month_excess))}${td(pct(r.top5_legacy_overlap))}</tr>`).join("");
}
const methods=[...new Set(D.ema_screening.map(x=>x.method))];$("#model-method").innerHTML+=[...methods].map(x=>`<option>${x}</option>`).join("");$("#model-dataset").onchange=modelRows;$("#model-method").onchange=modelRows;modelRows();

const perfOrder=["alpha_top5_equal","alpha_top10_equal",...D.performance.map(x=>x.series).filter(x=>!["alpha_top5_equal","alpha_top10_equal","Legacy","SPY total return"].includes(x)),"Legacy","SPY total return"];
$("#perf-rows").innerHTML=perfOrder.map(s=>perfBy(s)).filter(Boolean).map(r=>`<tr><td>${labels[r.series]||r.series.replaceAll("_"," ")}</td>${td(pct(r.cagr))}${td(num(r.sharpe))}${td(pct(r.annualized_volatility))}${td(pct(r.max_drawdown))}${td(`${r.worst_full_calendar_year} · ${pct(r.worst_full_calendar_year_return)}`)}${td(pct(r.average_turnover))}${td(pct(r.average_maximum_position_weight))}${td(pct(r.maximum_sector_weight))}</tr>`).join("");
$("#regime-rows").innerHTML=D.regimes.map(r=>`<tr><td>${r.period}</td><td>${r.series}</td>${td(r.months)}${td(pct(r.cagr))}${td(pct(r.volatility))}${td(num(r.sharpe))}${td(pct(r.max_drawdown))}</tr>`).join("");
const annualCols=["year","months","alpha_top5_equal","alpha_top10_equal","Legacy","SPY total return"];
$("#annual-table").innerHTML=`<thead><tr>${annualCols.map(x=>`<th>${labels[x]||x}</th>`).join("")}</tr></thead><tbody>`+D.annual.map(r=>`<tr>${annualCols.map((x,i)=>i<2?`<td>${r[x]}${x==="year"&&r.months<12?"*":""}</td>`:td(pct(r[x]))).join("")}</tr>`).join("")+"</tbody>";

function drawLine(id,field){
 const el=$("#"+id),w=Math.max(700,el.clientWidth||900),h=310,p={l:52,r:18,t:16,b:34},series=["Top 5 égal","Top 10 égal","Legacy","SPY"];
 let vals=D.wealth.flatMap(r=>series.map(s=>r[s][field])),min=Math.min(...vals),max=Math.max(...vals);if(field==="wealth")min=0;
 const x=i=>p.l+i*(w-p.l-p.r)/(D.wealth.length-1),y=v=>p.t+(max-v)*(h-p.t-p.b)/(max-min||1);
 let svg=`<svg viewBox="0 0 ${w} ${h}">`;
 for(let j=0;j<5;j++){const v=min+(max-min)*j/4,yy=y(v);svg+=`<line class="gridline" x1="${p.l}" y1="${yy}" x2="${w-p.r}" y2="${yy}"/><text x="4" y="${yy+3}">${field==="wealth"?v.toFixed(1):pct(v,0)}</text>`}
 series.forEach(s=>{const pts=D.wealth.map((r,i)=>`${x(i)},${y(r[s][field])}`).join(" ");svg+=`<polyline points="${pts}" fill="none" stroke="${color[s]}" stroke-width="${s==="Top 5 égal"?2.5:1.7}"/>`});
 [0,Math.floor(D.wealth.length/2),D.wealth.length-1].forEach(i=>svg+=`<text x="${x(i)}" y="${h-8}" text-anchor="${i===0?"start":i===D.wealth.length-1?"end":"middle"}">${D.wealth[i].month.slice(0,7)}</text>`);el.innerHTML=svg+"</svg>";
}
$("#wealth-legend").innerHTML=["Top 5 égal","Top 10 égal","Legacy","SPY"].map(s=>`<span><i style="background:${color[s]}"></i>${s}</span>`).join("");
const boot=D.bootstrap.filter(x=>x.strategy==="alpha_top5_equal"&&["Legacy","SPY total return"].includes(x.comparator));
$("#bootstrap-summary").innerHTML=boot.map(x=>`<p><b>vs ${labels[x.comparator]||x.comparator}</b><br>Δ Sharpe observé <span class="mono">${num(x.observed_sharpe_difference,3)}</span><br>IC 95 % <span class="mono">[${num(x.sharpe_difference_ci_low,3)} ; ${num(x.sharpe_difference_ci_high,3)}]</span></p>`).join("");
$("#bucket-summary").innerHTML=D.buckets.map(x=>`<p><b>${x.bucket==="ranks_1_5"?"Rangs 1–5":"Rangs 6–10"}</b> · CAGR <span class="mono">${pct(x.cagr)}</span> · Sharpe <span class="mono">${num(x.sharpe)}</span> · DD <span class="mono">${pct(x.max_drawdown)}</span></p>`).join("");
$("#cost-summary").innerHTML=[10,25,50,100].map(c=>{const x=D.costs.find(r=>r.strategy==="alpha_top5_equal"&&r.cost_bps===c);return `<p><b>${c} pb × turnover</b> · CAGR <span class="mono">${pct(x?.cagr)}</span> · Sharpe <span class="mono">${num(x?.sharpe)}</span></p>`}).join("");
$("#promotion-summary").innerHTML=D.promotion.map(x=>`<p>${badge(x.pass)} ${x.gate.replaceAll("_"," ")}</p>`).join("");

const months=[...new Set(D.holdings.map(x=>x.holding_month))].sort();$("#holding-month").innerHTML=months.map(x=>`<option value="${x}">${x.slice(0,7)}</option>`).join("");$("#holding-month").value=months.at(-1);
function portfolioCard(title,portfolio,month,cls=""){
 const current=D.holdings.filter(x=>x.holding_month===month&&x.portfolio===portfolio).sort((a,b)=>(a.rank??99)-(b.rank??99)),idx=months.indexOf(month),prev=idx>0?new Set(D.holdings.filter(x=>x.holding_month===months[idx-1]&&x.portfolio===portfolio).map(x=>x.ticker)):new Set(),now=new Set(current.map(x=>x.ticker)),exits=idx>0?D.holdings.filter(x=>x.holding_month===months[idx-1]&&x.portfolio===portfolio&&!now.has(x.ticker)).map(x=>x.ticker):[];
 return `<article class="panel portfolio-card ${cls}"><h2>${title}</h2>${current.map(x=>`<div class="ticker-row"><span class="ticker">${x.ticker.replace(".US","")}</span><span>${x.sector||"—"}<br><span class="fine">${x.rank?`rang ${x.rank} · `:""}${x.calibrated_probability==null?"":`p ${pct(x.calibrated_probability,0)} · `}réalisé ${pct(x.realized_return_1m)}</span></span><span><b>${pct(x.weight,0)}</b><br><span class="action ${prev.has(x.ticker)?"keep":"enter"}">${prev.has(x.ticker)?"KEEP":"ENTER"}</span></span></div>`).join("")}<p class="fine">Sorties : ${exits.length?exits.map(x=>x.replace(".US","")).join(", "):"aucune"}</p></article>`;
}
function renderPortfolios(){const m=$("#holding-month").value,r=D.monthly.find(x=>x.holding_month===m);$("#month-returns").textContent=`Top 5 ${pct(r?.alpha_top5_return)} · Top 10 ${pct(r?.alpha_top10_return)} · Legacy ${pct(r?.legacy_return)} · SPY ${pct(r?.spy_return)}`;$("#portfolio-cards").innerHTML=portfolioCard("Legacy","Legacy publié",m,"legacy")+portfolioCard("Alpha Top 5","Alpha Top 5 égal",m)+portfolioCard("Alpha Top 10","Alpha Top 10 égal",m)}
$("#holding-month").onchange=renderPortfolios;renderPortfolios();

const shapMonths=[...new Set(D.shap.map(x=>x.m))].sort();$("#shap-month").innerHTML+=shapMonths.map(x=>`<option value="${x}">${x.slice(0,7)}</option>`).join("");
$("#shap-feature").innerHTML=D.features.map((f,i)=>`<option value="${i}">${i+1}. ${f}</option>`).join("");
function shapSubset(){const m=$("#shap-month").value;return m==="all"?D.shap:D.shap.filter(x=>x.m===m)}
function importance(rows){return D.features.map((f,i)=>({f,i,v:rows.reduce((a,r)=>a+(r.s[i]==null?0:Math.abs(r.s[i])),0)/Math.max(1,rows.filter(r=>r.s[i]!=null).length)})).sort((a,b)=>b.v-a.v)}
function canvasSetup(c){const rect=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=Math.max(600,rect.width*dpr);c.height=Math.max(300,rect.height*dpr);const ctx=c.getContext("2d");ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height}}
function renderBeeswarm(rows,imp){
 const c=$("#beeswarm"),{ctx,w,h}=canvasSetup(c),left=185,right=20,top=20,bottom=28,topf=imp.slice(0,15),all=topf.flatMap(x=>rows.map(r=>r.s[x.i]).filter(Number.isFinite)),lim=Math.max(.001,...all.map(Math.abs));
 const styles=getComputedStyle(document.documentElement),ink=styles.getPropertyValue("--muted"),line=styles.getPropertyValue("--line");ctx.clearRect(0,0,w,h);ctx.font='10px "IBM Plex Mono"';ctx.fillStyle=ink;ctx.strokeStyle=line;
 const x=v=>left+(v+lim)*(w-left-right)/(2*lim),y=j=>top+j*(h-top-bottom)/14;ctx.beginPath();ctx.moveTo(x(0),top-8);ctx.lineTo(x(0),h-bottom+8);ctx.stroke();
 topf.forEach((q,j)=>{const vals=rows.map(r=>r.v[q.i]).filter(Number.isFinite),mn=Math.min(...vals),mx=Math.max(...vals);ctx.fillStyle=ink;ctx.textAlign="right";ctx.fillText(q.f.slice(0,27),left-8,y(j)+3);rows.forEach((r,k)=>{const sv=r.s[q.i],fv=r.v[q.i];if(!Number.isFinite(sv))return;const z=Number.isFinite(fv)?(fv-mn)/(mx-mn||1):.5;ctx.fillStyle=`rgb(${Math.round(55+180*z)},${Math.round(98-30*z)},${Math.round(190-90*z)})`;ctx.globalAlpha=.66;ctx.beginPath();ctx.arc(x(sv),y(j)+((k*37)%13-6)*.65,2.5,0,Math.PI*2);ctx.fill()})});ctx.globalAlpha=1;ctx.fillStyle=ink;ctx.textAlign="center";ctx.fillText("- contribution",left+50,h-7);ctx.fillText("+ contribution",w-right-50,h-7);
}
function renderIndividual(rows,idx){
 const c=$("#individual"),{ctx,w,h}=canvasSetup(c),p={l:62,r:22,t:24,b:42},pts=rows.map(r=>({x:r.v[idx],y:r.s[idx],t:r.t,m:r.m})).filter(x=>Number.isFinite(x.x)&&Number.isFinite(x.y));if(!pts.length)return;
 let xmin=Math.min(...pts.map(x=>x.x)),xmax=Math.max(...pts.map(x=>x.x)),ymin=Math.min(...pts.map(x=>x.y)),ymax=Math.max(...pts.map(x=>x.y));const xx=x=>p.l+(x-xmin)*(w-p.l-p.r)/(xmax-xmin||1),yy=y=>p.t+(ymax-y)*(h-p.t-p.b)/(ymax-ymin||1),styles=getComputedStyle(document.documentElement),ink=styles.getPropertyValue("--muted"),line=styles.getPropertyValue("--line");ctx.clearRect(0,0,w,h);ctx.strokeStyle=line;ctx.fillStyle=ink;ctx.font='10px "IBM Plex Mono"';
 for(let j=0;j<5;j++){const y=p.t+j*(h-p.t-p.b)/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();const v=ymax-j*(ymax-ymin)/4;ctx.fillText(v.toFixed(3),5,y+3)}
 pts.forEach(q=>{ctx.fillStyle="#26387E";ctx.globalAlpha=.58;ctx.beginPath();ctx.arc(xx(q.x),yy(q.y),3,0,Math.PI*2);ctx.fill()});ctx.globalAlpha=1;ctx.fillStyle=ink;ctx.textAlign="center";ctx.fillText(`Valeur · ${xmin.toFixed(3)} → ${xmax.toFixed(3)}`,w/2,h-10);$("#individual-title").textContent=`Analyse individuelle · ${D.features[idx]}`;
}
function renderShap(){
 const rows=shapSubset(),imp=importance(rows),folds=[...new Set(rows.map(x=>x.f))],m=$("#shap-month").value,idx=Number($("#shap-feature").value||imp[0].i);if(m!=="all"&&!$("#shap-feature").dataset.manual){$("#shap-feature").value=imp[0].i}
 $("#shap-metrics").innerHTML=metric("Observations",rows.length,m==="all"?"échantillon OOS complet":"échantillon du mois")+metric("Folds",folds.join(", "),m==="all"?"15 modèles":"modèle(s) associé(s)")+metric("Variables",D.features.length,"EMA uniquement")+metric("|SHAP| n°1",num(imp[0].v,4),imp[0].f);
 const mx=imp[0].v||1;$("#shap-bars").innerHTML=imp.slice(0,20).map((x,j)=>`<div class="shap-bar"><span>${j+1}. ${x.f.slice(0,28)}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${100*x.v/mx}%"></span></span><b>${num(x.v,4)}</b></div>`).join("");
 renderBeeswarm(rows,imp);renderIndividual(rows,Number($("#shap-feature").value));const fi=Number($("#shap-feature").value);$("#shap-detail-note").textContent=`${rows.length} observations ; variable ${D.features[fi]}`;$("#shap-detail").innerHTML=rows.slice().sort((a,b)=>Math.abs(b.s[fi]||0)-Math.abs(a.s[fi]||0)).map(r=>`<tr><td>${r.m.slice(0,7)}</td><td>${r.t}</td>${td(r.f)}${td(num(r.v[fi],5))}${td(num(r.s[fi],5))}</tr>`).join("");
}
$("#shap-month").onchange=()=>{$("#shap-feature").dataset.manual="";renderShap()};$("#shap-feature").onchange=()=>{$("#shap-feature").dataset.manual="1";renderShap()};
function renderLexicon(){const q=$("#lexicon-search").value.toLowerCase();$("#lexicon-rows").innerHTML=D.lexicon.filter(x=>!q||Object.values(x).join(" ").toLowerCase().includes(q)).map(x=>`<tr>${td(x.importance_rank)}<td><code>${x.feature}</code></td>${td(num(x.mean_abs_shap,5))}<td>${x.transformation}</td><td>${x.ema_numerator_span_days} / ${x.ema_denominator_span_days}</td><td>${x.unit}</td><td>${x.exact_definition}</td><td>${x.economic_interpretation}</td></tr>`).join("")}$("#lexicon-search").oninput=renderLexicon;renderLexicon();

$("#risk-model-rows").innerHTML=D.risk_models.map(r=>`<tr><td>${r.head}</td>${td(r.horizon+"m")}<td>${r.task_type}</td><td>${r.target}</td>${td(r.test_rows)}${td(num(r.monthly_spearman,3))}${td(num(r.rmse,3))}${td(num(r.normalized_rmse,3))}${td(num(r.mae,3))}${td(num(r.roc_auc,3))}${td(num(r.pr_auc_average_precision,3))}${td(num(r.pr_auc_lift_vs_prevalence,2))}</tr>`).join("");
$("#risk-perf-rows").innerHTML=D.risk_performance.map(r=>`<tr><td>${labels[r.series]||r.series.replaceAll("_"," ")}</td>${td(pct(r.cagr))}${td(pct(r.annualized_volatility))}${td(num(r.sharpe))}${td(pct(r.max_drawdown))}${td(`${r.worst_full_calendar_year} · ${pct(r.worst_full_calendar_year_return)}`)}</tr>`).join("");
$("#risk-gates").innerHTML=D.risk_gates.map(r=>`<article class="panel"><h3>${r.strategy.replaceAll("_"," ")}</h3>${Object.entries(r).filter(([k])=>k!=="strategy").map(([k,v])=>`<p>${badge(v)} ${k.replaceAll("_"," ")}</p>`).join("")}</article>`).join("");

const live=D.live.manifest,vm=live.validation_metrics;
$("#live-metrics").innerHTML=metric("ROC AUC validation",num(vm.roc_auc,3),"2025-07 → 2025-12")+metric("PR AUC",num(vm.pr_auc_average_precision,3),`prévalence ${pct(vm.positive_rate)}`)+metric("Lift PR",num(vm.pr_auc_lift_vs_prevalence,2),"contre prévalence")+metric("Univers live",live.live_universe_rows,`${live.holding_month_universe_filter.rows_before-live.holding_month_universe_filter.rows_after} retraits`);
function liveCard(title,rows,cls=""){return `<article class="panel portfolio-card ${cls}"><h2>${title}</h2>${rows.map(x=>{const price=x.last_close??x.price,weight=x.equal_weight??x.weight_normalized??x.weight;return `<div class="ticker-row"><span class="ticker">${String(x.ticker).replace(".US","")}</span><span>${x.company_name||x.sector||""}<br><span class="fine">${price==null?"":money(price)} ${x.calibrated_probability==null?"":`· p ${pct(x.calibrated_probability,0)}`}</span></span><b>${pct(weight,0)}</b></div>`}).join("")}</article>`}
$("#live-portfolios").innerHTML=liveCard("Legacy juillet",D.live.legacy,"legacy")+liveCard("Alpha Top 5",D.live.top5)+liveCard("Alpha Top 10",D.live.top10);
const lineage=[["Statut",live.status],["Décision",live.decision_month],["Détention",live.holding_month],["Train",`${live.train_start} → ${live.train_end} · ${live.train_rows} lignes`],["Validation",`${live.validation_start} → ${live.validation_end} · ${live.validation_rows} lignes`],["Maturité labels",live.label_maturity_cutoff],["Features",`${live.feature_count} · ${live.winner_pair_count} paires EMA`],["Tickers retirés",live.holding_month_universe_filter.removed_tickers.join(", ")]];
$("#live-lineage").innerHTML=lineage.map(x=>`<tr><th>${x[0]}</th><td class="mono">${x[1]}</td></tr>`).join("");
$("#sources").innerHTML=__SOURCES__.map(x=>`<p>${x.path}<br><span class="fine">${x.sha256}</span></p>`).join("");
</script>
</body></html>"""


def render(output_dir: Path) -> tuple[Path, Path]:
    payload, sources = build_payload()
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    output_path = html_dir / "alpharank_research_center.html"
    manifest_path = output_dir / "manifest.json"
    source_manifest = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _hash(path),
            "bytes": path.stat().st_size,
        }
        for path in sources
    ]
    encoded = json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    page = HTML.replace("__PAYLOAD__", encoded).replace(
        "__SOURCES__", json.dumps(source_manifest, ensure_ascii=False)
    )
    output_path.write_text(page, encoding="utf-8")
    try:
        report_reference = str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        report_reference = str(output_path)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "report": report_reference,
                "status": "research_dashboard",
                "semantics": {
                    "historical_retraining": "once per outer fold",
                    "monthly_shap_filter": (
                        "test-month observations explained by that fold model"
                    ),
                    "live_retraining": "once per monthly execution",
                    "shap_unit": "raw XGBoost margin / log-odds",
                    "shap_sampling": "80 rows per fold; 1200 total",
                },
                "counts": payload["meta"],
                "sources": source_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report, manifest = render(args.output_dir.resolve())
    print(report)
    print(manifest)


if __name__ == "__main__":
    main()
