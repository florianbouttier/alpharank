#!/usr/bin/env python3
"""Render the latest same-snapshot Legacy/boosting/SPY research dashboard."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.portfolio.attribution import (  # noqa: E402
    portfolio_return_attribution,
    reference_return_attribution,
)
from alpharank.portfolio.lineage import load_manifest  # noqa: E402
from alpharank.portfolio.performance import (  # noqa: E402
    advanced_performance_statistics,
)


PERIOD_METRIC_FIELDS = (
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "information_ratio",
    "beta",
    "alpha",
    "correlation",
    "benchmark_hit_rate",
    "var_95",
    "cvar_95",
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
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return _clean(frame.to_dicts())


def _wide_monthly(monthly: pl.DataFrame) -> pl.DataFrame:
    return (
        monthly.pivot(on="strategy", index="holding_month", values="net_return")
        .sort("holding_month")
    )


def _period_metrics(monthly: pl.DataFrame) -> dict[str, list[list[float | None]]]:
    strategies = monthly["strategy"].unique().sort().to_list()
    wide = _wide_monthly(monthly)
    benchmark = wide["SPY total return"].to_numpy()
    months = wide["holding_month"].to_list()
    output: dict[str, list[list[float | None]]] = {}
    for start in range(len(months)):
        for end in range(start, len(months)):
            rows: list[list[float | None]] = []
            for strategy in strategies:
                stats = advanced_performance_statistics(
                    wide[strategy].to_numpy()[start : end + 1],
                    benchmark_returns=benchmark[start : end + 1],
                )
                rows.append([_clean(stats[field]) for field in PERIOD_METRIC_FIELDS])
            output[f"{months[start].isoformat()}|{months[end].isoformat()}"] = rows
    return output


def _attribution(
    holdings: pl.DataFrame,
    monthly: pl.DataFrame,
) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for strategy in monthly["strategy"].unique().sort().to_list():
        strategy_monthly = monthly.filter(pl.col("strategy") == strategy)
        strategy_holdings = holdings.filter(pl.col("strategy") == strategy)
        if strategy_holdings.is_empty():
            parts.append(
                reference_return_attribution(
                    strategy_monthly,
                    component="SPY",
                )
            )
        else:
            parts.append(
                portfolio_return_attribution(strategy_holdings, strategy_monthly)
            )
    return pl.concat(parts, how="diagonal_relaxed").sort(
        ["strategy", "holding_month", "component"]
    )


def _fold_shap_payload(samples: pl.DataFrame) -> list[dict[str, Any]]:
    shap_columns = [column for column in samples.columns if column.startswith("shap__")]
    rows: list[dict[str, Any]] = []
    for frame in samples.partition_by("fold", maintain_order=True):
        fold = int(frame["fold"][0])
        importance = []
        for column in shap_columns:
            mean_abs_shap = frame[column].abs().mean()
            if mean_abs_shap is None:
                continue
            importance.append(
                {
                    "feature": column.removeprefix("shap__"),
                    "mean_abs_shap": float(mean_abs_shap),
                }
            )
        importance.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
        rows.append(
            {
                "fold": fold,
                "rows": frame.height,
                "start": str(frame["decision_month"].min()),
                "end": str(frame["decision_month"].max()),
                "importance": importance,
            }
        )
    return rows


def _write_shap_sidecars(
    *,
    samples: pl.DataFrame,
    predictions: pl.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [
        column.removeprefix("shap__")
        for column in samples.columns
        if column.startswith("shap__")
    ]
    prediction_counts = predictions.group_by("decision_month").len(name="predictions")
    files: list[dict[str, Any]] = []
    for month_rows in samples.sort(["decision_month", "ticker"]).partition_by(
        "decision_month", maintain_order=True
    ):
        month = month_rows["decision_month"][0]
        expected = prediction_counts.filter(pl.col("decision_month") == month)
        if expected.height != 1 or int(expected["predictions"][0]) != month_rows.height:
            raise ValueError(
                f"Monthly SHAP coverage mismatch for {month}: "
                f"shap={month_rows.height}, predictions={expected.to_dicts()}"
            )
        compact = []
        for row in month_rows.iter_rows(named=True):
            compact.append(
                {
                    "ticker": row["ticker"],
                    "fold": int(row["fold"]),
                    "values": [row.get(f"value__{name}") for name in feature_names],
                    "shap": [row.get(f"shap__{name}") for name in feature_names],
                }
            )
        path = output_dir / f"{month.isoformat()[:7]}.json.gz"
        encoded = json.dumps(
            _clean({"month": month.isoformat(), "features": feature_names, "rows": compact}),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
        files.append(
            {
                "month": month.isoformat(),
                "fold": int(month_rows["fold"][0]),
                "rows": month_rows.height,
                "file": path.name,
                "sha256": _hash(path),
            }
        )
    manifest = {
        "sampling": "exhaustive",
        "rows": samples.height,
        "prediction_rows": predictions.height,
        "features": len(feature_names),
        "months": len(files),
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_payload(
    *,
    comparison_dir: Path,
    boosting_dir: Path,
    legacy_run_dir: Path,
) -> tuple[dict[str, Any], list[Path], pl.DataFrame, pl.DataFrame]:
    sources = [
        comparison_dir / "manifest.json",
        comparison_dir / "comparison_common_holdings.parquet",
        comparison_dir / "comparison_common_monthly.parquet",
        comparison_dir / "comparison_common_performance.csv",
        boosting_dir / "manifest.json",
        boosting_dir / "model_horizon_summary.csv",
        boosting_dir / "classification_h06/predictions.parquet",
        boosting_dir / "classification_h06/fold_metrics.csv",
        boosting_dir / "classification_h06/fold_feature_manifest.csv",
        boosting_dir / "classification_h06/shap_samples.parquet",
        boosting_dir / "classification_h06/shap_importance.csv",
        legacy_run_dir / "data_input_manifest.json",
    ]
    revision_audit_path = comparison_dir / "data_revision_audit.json"
    if revision_audit_path.exists():
        sources.append(revision_audit_path)
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dashboard sources:\n" + "\n".join(map(str, missing)))

    comparison_manifest = load_manifest(sources[0])
    boosting_manifest = load_manifest(boosting_dir / "manifest.json")
    legacy_manifest = load_manifest(legacy_run_dir / "data_input_manifest.json")
    revision_audit = (
        load_manifest(revision_audit_path) if revision_audit_path.exists() else None
    )
    if not comparison_manifest["lineage_check"]["passed"]:
        raise ValueError("The comparison manifest is not same-snapshot eligible.")
    if not comparison_manifest.get("ticker_exclusion_check", {}).get("passed"):
        raise ValueError(
            "The comparison manifest does not prove matching ticker exclusions."
        )

    holdings = pl.read_parquet(comparison_dir / "comparison_common_holdings.parquet")
    monthly = pl.read_parquet(comparison_dir / "comparison_common_monthly.parquet")
    predictions = pl.read_parquet(boosting_dir / "classification_h06/predictions.parquet")
    shap_samples = pl.read_parquet(boosting_dir / "classification_h06/shap_samples.parquet")
    attribution = _attribution(holdings, monthly)
    folds = (
        pl.read_csv(
            boosting_dir / "classification_h06/fold_feature_manifest.csv",
            try_parse_dates=True,
        )
        .join(
            pl.read_csv(
                boosting_dir / "classification_h06/fold_metrics.csv",
                try_parse_dates=True,
            ),
            on="fold",
        )
        .sort("fold")
    )
    prediction_months = (
        predictions.group_by("decision_month", "fold", "target_status")
        .len(name="rows")
        .sort(["decision_month", "target_status"])
    )
    dataset_summary = {
        name: {
            "rows": dataset.get("summary", {}).get("rows"),
            "sha256": dataset.get("sha256"),
            "max_temporal_values": dataset.get("summary", {}).get("max_temporal_values", {}),
        }
        for name, dataset in legacy_manifest.get("datasets", {}).items()
    }
    strategies = monthly["strategy"].unique().sort().to_list()
    payload = {
        "meta": {
            "created": datetime.now().astimezone().isoformat(),
            "snapshot_id": legacy_manifest.get("open_source_output_run_id"),
            "legacy_run_id": legacy_run_dir.name,
            "start": str(monthly["holding_month"].min()),
            "end": str(monthly["holding_month"].max()),
            "months": monthly.filter(pl.col("strategy") == strategies[0]).height,
            "benchmark": "SPY total return from adjusted_close",
            "transaction_cost_bps": comparison_manifest["transaction_cost_bps_times_turnover"],
            "comparison_profile": comparison_manifest.get("comparison_profile"),
            "stock_price_max": legacy_manifest["datasets"]["final_price"]["summary"][
                "max_temporal_values"
            ]["date"],
            "benchmark_price_max": legacy_manifest["datasets"]["sp500_price"]["summary"][
                "max_temporal_values"
            ]["date"],
        },
        "calendar": comparison_manifest["calendar"],
        "strategies": strategies,
        "metric_fields": PERIOD_METRIC_FIELDS,
        "period_metrics": _period_metrics(monthly),
        "monthly": _records(monthly),
        "holdings": _records(holdings),
        "performance": _records(
            pl.read_csv(comparison_dir / "comparison_common_performance.csv", try_parse_dates=True)
        ),
        "attribution": _records(
            attribution.select(
                "strategy",
                "holding_month",
                "component",
                "component_type",
                "simple_return_contribution",
                "log_return_contribution",
            )
        ),
        "boosting": {
            "manifest": _clean(boosting_manifest),
            "summary": _records(
                pl.read_csv(boosting_dir / "model_horizon_summary.csv", try_parse_dates=True)
            ),
            "folds": _records(folds),
            "prediction_months": _records(prediction_months),
            "fold_shap": _fold_shap_payload(shap_samples),
        },
        "lineage": {
            "comparison": _clean(comparison_manifest["lineage_check"]),
            "ticker_exclusions": _clean(
                comparison_manifest["ticker_exclusion_check"]
            ),
            "revision": _clean(revision_audit),
            "legacy": {
                "snapshot_id": legacy_manifest.get("snapshot_id"),
                "input_snapshot_dir": legacy_manifest.get("input_snapshot_dir"),
                "open_source_output_run_id": legacy_manifest.get("open_source_output_run_id"),
                "open_source_ingestion_run_id": legacy_manifest.get("open_source_ingestion_run_id"),
                "published_snapshot_match": legacy_manifest.get(
                    "open_source_output_matches_published_snapshot"
                ),
                "datasets": dataset_summary,
            },
        },
    }
    return payload, sources, shap_samples, predictions


HTML = r'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank - Backtest commun</title>
<style>.bar-row>span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}</style>
<style>
:root{--bg:#f8fafc;--panel:#fff;--surface:#f1f5f9;--line:#d7e0ea;--ink:#020617;--muted:#475569;--navy:#111d55;--gold:#9b8816;--green:#265511;--red:#802331;--blue:#0369a1;--r:6px}*{box-sizing:border-box;letter-spacing:0}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Sans",Arial,sans-serif}.shell{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:100vh}aside{position:sticky;top:0;height:100vh;background:var(--navy);color:#fff;padding:22px 16px;display:flex;flex-direction:column}.brand{font:700 18px "IBM Plex Mono",monospace;padding:0 10px 20px;border-bottom:1px solid #ffffff35}.brand small{display:block;font:10px "IBM Plex Sans";opacity:.65;margin-top:5px;text-transform:uppercase}.nav{display:grid;gap:4px;margin-top:18px}.nav button{border:0;border-radius:4px;background:transparent;color:#ffffffb8;text-align:left;padding:10px 12px;font-weight:650;cursor:pointer}.nav button.active,.nav button:hover{background:#ffffff1f;color:#fff}.aside-meta{margin-top:auto;border-top:1px solid #ffffff35;padding:14px 10px 0;font:11px "IBM Plex Mono",monospace;color:#ffffffb3}main{min-width:0}.top{height:60px;position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:0 26px;display:flex;align-items:center;justify-content:space-between}.page{display:none;padding:26px;max-width:1480px;margin:auto}.page.active{display:block}h1{font-size:30px;line-height:1.1;margin:5px 0 8px}h2{font-size:20px;margin:0 0 12px}h3{font-size:15px;margin:0 0 10px}.lede{color:var(--muted);max-width:1050px;font-size:15px}.eyebrow{font-size:10px;text-transform:uppercase;color:var(--blue);font-weight:750}.grid{display:grid;gap:12px}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:16px;min-width:0}.metric span{font-size:11px;color:var(--muted)}.metric strong{display:block;font:700 25px "IBM Plex Mono",monospace;margin:6px 0}.metric small,.fine{color:var(--muted);font-size:11px}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:end;margin:16px 0 12px}.field{display:grid;gap:4px}.field span{font-size:10px;text-transform:uppercase;color:var(--muted);font-weight:700}select,button.control{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:8px 10px;color:var(--ink)}button.control{cursor:pointer}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:var(--r);background:var(--panel)}table{width:100%;border-collapse:collapse;min-width:760px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}th{font-size:10px;text-transform:uppercase;color:var(--muted);background:var(--surface)}td.num{text-align:right;font-family:"IBM Plex Mono",monospace}.chart{height:340px;border:1px solid var(--line);background:var(--panel);border-radius:var(--r);position:relative}.chart svg,.chart canvas{display:block;width:100%;height:100%}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0;font-size:11px}.legend i{display:inline-block;width:10px;height:10px;margin-right:5px}.callout{border-left:4px solid var(--blue);background:#eef6fb;padding:13px 15px;margin:14px 0}.callout.warn{border-color:var(--gold);background:#faf7e8}.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}.step{border-top:3px solid var(--navy);padding:12px;background:var(--panel);border-left:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.step b{display:block;font:700 10px "IBM Plex Mono",monospace;color:var(--navy);margin-bottom:6px}.badge{display:inline-block;padding:3px 6px;border:1px solid currentColor;border-radius:3px;font:700 10px "IBM Plex Mono",monospace}.ok{color:var(--green)}.pending{color:var(--gold)}.bad{color:var(--red)}.ticker-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:6px}.ticker-row{display:grid;grid-template-columns:1fr auto;gap:8px;border-bottom:1px solid var(--line);padding:8px 2px}.ticker{font:700 12px "IBM Plex Mono",monospace}.split{display:grid;grid-template-columns:1fr 1.25fr;gap:12px}.bars{display:grid;gap:7px}.bar-row{display:grid;grid-template-columns:minmax(140px,1fr) 3fr 72px;gap:8px;align-items:center;font:11px "IBM Plex Mono",monospace}.track{height:10px;background:var(--surface)}.fill{height:100%;background:var(--navy)}.formula{font:12px "IBM Plex Mono",monospace;background:var(--surface);padding:12px;border-left:3px solid var(--navy)}@media(max-width:1050px){.shell{grid-template-columns:1fr}aside{position:relative;height:auto}.nav{grid-template-columns:repeat(4,1fr)}.aside-meta{display:none}.g4,.g3,.g2,.split,.flow{grid-template-columns:1fr 1fr}.top{position:relative}}@media(max-width:680px){.page{padding:16px}.g4,.g3,.g2,.split,.flow,.nav{grid-template-columns:1fr}.chart{height:280px}}
</style></head><body><div class="shell"><aside><div class="brand">ALPHARANK<small>Backtest commun auditable</small></div><div class="nav"><button class="active" data-page="overview">Performance</button><button data-page="portfolios">Portefeuilles</button><button data-page="methods">Methodes</button><button data-page="boosting">Boosting & SHAP</button></div><div class="aside-meta" id="aside-meta"></div></aside><main><header class="top"><b>Legacy / XGBoost / SPY</b><span class="badge ok" id="lineage-badge">MEME SNAPSHOT</span></header>
<section class="page active" id="overview"><div class="eyebrow">Backtest comparable</div><h1>Trois series, un snapshot, un calendrier</h1><p class="lede" id="overview-lede"></p><div class="controls"><label class="field"><span>Debut</span><select id="start"></select></label><label class="field"><span>Fin</span><select id="end"></select></label><button class="control" data-preset="all">Tout</button><button class="control" data-preset="120">10 ans</button><button class="control" data-preset="60">5 ans</button><span class="fine" id="period-label"></span></div><div class="grid g4" id="kpis"></div><div class="panel" style="margin-top:12px"><h2>Indicateurs avances</h2><div class="table-wrap"><table><thead><tr><th>Strategie</th><th>Rendement total</th><th>CAGR</th><th>Volatilite</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Information ratio</th><th>Beta SPY</th><th>Alpha ann.</th></tr></thead><tbody id="metrics"></tbody></table></div></div><div class="panel" style="margin-top:12px"><h2>Croissance de 1 EUR</h2><div class="legend" id="legend"></div><div class="chart" id="wealth"></div></div></section>
<section class="page" id="portfolios"><div class="eyebrow">Audit mensuel</div><h1>Selections et contribution au CAGR</h1><p class="lede">Les portefeuilles affiches sont les positions effectivement simulees au mois de detention. L'attribution repartit exactement le log-rendement net entre actions et couts.</p><div class="controls"><label class="field"><span>Mois detenu</span><select id="holding-month"></select></label><label class="field"><span>Strategie</span><select id="holding-strategy"></select></label></div><div class="grid g3" id="holding-cards"></div><div class="panel" style="margin-top:12px"><h2>D'ou vient le CAGR ?</h2><div class="controls"><label class="field"><span>Strategie</span><select id="attr-strategy"></select></label><label class="field"><span>Niveau</span><select id="attr-level"><option value="total">Total</option><option value="year">Annee</option><option value="month">Mois</option></select></label><label class="field"><span>Periode</span><select id="attr-period"></select></label></div><div class="formula" id="attr-formula"></div><div class="chart" id="waterfall"></div><div class="table-wrap" style="margin-top:10px"><table><thead><tr><th>Composante</th><th>Contribution log annualisee</th><th>Equivalent multiplicatif</th><th>Contribution simple</th></tr></thead><tbody id="attr-table"></tbody></table></div></div></section>
<section class="page" id="methods"><div class="eyebrow">Methodologie causale</div><h1>Deux signaux, un moteur de portefeuille</h1><div class="grid g2"><article class="panel"><h2>Legacy</h2><p>Quatre analyses Optuna selectionnent des couples de moyennes mobiles exponentielles sur l'historique disponible. Les quatre paniers sont agreges par frequence de selection. La production utilise <b>Combined_Frequency</b>, sans cout de transaction dans sa convention historique.</p><p class="fine">Decision t, detention t+1. Le benchmark comparable est SPY total return calcule avec adjusted_close.</p></article><article class="panel"><h2>XGBoost H6</h2><p>Classifieur du decile superieur de surperformance future a six mois. Les variables sont les couples EMA gagnants Legacy connus avant le cutoff d'entrainement. Le modele est entraine sur le passe, calibre sur six mois, puis fige sur chaque bloc test.</p><p class="fine">Le score H6 classe les titres ; le portefeuille Top 5 ou Top 10 est detenu un mois et supporte 10 pb multiplies par le turnover. Le profil public est versionne et refuse les seuils de liquidite divergents.</p></article></div><div class="flow" style="margin-top:12px"><div class="step"><b>1 SNAPSHOT</b><span>Package open source historise et hashe.</span></div><div class="step"><b>2 FEATURES</b><span>Donnees disponibles a la date de decision.</span></div><div class="step"><b>3 TRAIN</b><span>Fenetre expansive strictement passee.</span></div><div class="step"><b>4 VALIDATION</b><span>Six mois apres le train.</span></div><div class="step"><b>5 TEST</b><span>Modele gele sur douze mois.</span></div><div class="step"><b>6 HOLDING</b><span>Selection t, rendement realise t+1.</span></div></div><div class="callout warn"><b>Deux calendriers sont necessaires.</b> Les indicateurs du modele H6 exigent une cible arrivee a maturite. Le backtest portefeuille n'exige que le rendement du mois suivant. Les mois H6 encore en attente sont scores hors echantillon mais exclus des AUC, NDCG et autres metriques de modele.</div><div class="panel"><h2>Lineage et fraicheur</h2><div class="table-wrap"><table><thead><tr><th>Jeu</th><th>Lignes</th><th>Derniere date</th><th>SHA-256</th></tr></thead><tbody id="datasets"></tbody></table></div></div><div class="panel" style="margin-top:12px"><h2>Revisions depuis le snapshot precedent</h2><p class="lede" id="revision-summary"></p><div class="table-wrap"><table><thead><tr><th>Jeu</th><th>Nouvelles lignes</th><th>Lignes retirees</th><th>Lignes historiques modifiees</th></tr></thead><tbody id="revision-rows"></tbody></table></div><p class="fine">Une ligne historique modifiee signale une revision de source. L'identite des portefeuilles est controlee separement au seuil numerique de 1e-12.</p></div></section>
<section class="page" id="boosting"><div class="eyebrow">Diagnostic hors echantillon</div><h1>Train, validation, test et SHAP exhaustif</h1><p class="lede" id="shap-lede"></p><div class="controls"><label class="field"><span>Ensemble de test</span><select id="fold"></select></label></div><div class="grid g4" id="fold-kpis"></div><div class="panel" style="margin-top:12px"><h2>Bornes du fold</h2><div class="table-wrap"><table><thead><tr><th>Train</th><th>Validation</th><th>Test</th><th>Lignes train</th><th>Lignes validation</th><th>Lignes test</th><th>Cibles evaluables</th><th>H6 en attente</th></tr></thead><tbody id="fold-calendar"></tbody></table></div></div><div class="split" style="margin-top:12px"><div class="panel"><h2>Importance SHAP du fold</h2><div class="bars" id="shap-bars"></div></div><div class="panel"><h2>Valeurs SHAP par mois test</h2><div class="controls"><label class="field"><span>Mois</span><select id="shap-month"></select></label><label class="field"><span>Variable</span><select id="shap-feature"></select></label></div><div class="chart"><canvas id="shap-canvas"></canvas></div><script>new ResizeObserver(()=>{if(window.SHAPDATA&&typeof drawShap==='function'&&document.getElementById('boosting').classList.contains('active'))drawShap(window.SHAPDATA)}).observe(document.getElementById('shap-canvas').parentElement)</script><p class="fine" id="shap-status"></p></div></div></section>
</main></div><script>const DATA=__PAYLOAD__;const COLORS={'Boosting Top 5':'#111d55','Boosting Top 10':'#0369a1','Legacy':'#9b8816','SPY total return':'#826c7f'};const $=id=>document.getElementById(id);const pct=(v,d=1)=>v==null?'—':(100*v).toFixed(d)+' %';const num=(v,d=2)=>v==null?'—':Number(v).toFixed(d);const strategies=DATA.strategies;const monthlyBy={};for(const r of DATA.monthly){(monthlyBy[r.strategy]??=[]).push(r)}for(const rows of Object.values(monthlyBy))rows.sort((a,b)=>a.holding_month.localeCompare(b.holding_month));const months=monthlyBy[strategies[0]].map(r=>r.holding_month);function options(el,values,label=x=>x){el.innerHTML=values.map(v=>`<option value="${v}">${label(v)}</option>`).join('')}options($('start'),months);options($('end'),months);$('end').value=months.at(-1);options($('holding-month'),months);$('holding-month').value=months.at(-1);options($('holding-strategy'),strategies);options($('attr-strategy'),strategies);$('aside-meta').innerHTML=`Snapshot ${DATA.meta.snapshot_id||'—'}<br>${DATA.meta.start.slice(0,7)} → ${DATA.meta.end.slice(0,7)}<br>${DATA.meta.benchmark}`;$('overview-lede').textContent=`Donnees actions et SPY disponibles jusqu'au ${DATA.meta.stock_price_max.slice(0,10)}. Dernier mois complet et comparable : ${DATA.meta.end.slice(0,7)}. ${DATA.meta.months} mois communs, memes hashes d'entree, meme calendrier et SPY total return.`;document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.page).classList.add('active')});const fieldIndex=Object.fromEntries(DATA.metric_fields.map((x,i)=>[x,i]));function currentMetrics(){return DATA.period_metrics[$('start').value+'|'+$('end').value]}function renderOverview(){let rows=currentMetrics();if(!rows)return;let n=months.indexOf($('end').value)-months.indexOf($('start').value)+1;$('period-label').textContent=`${$('start').value.slice(0,7)} → ${$('end').value.slice(0,7)} · ${n} mois`;$('kpis').innerHTML=strategies.map((s,i)=>`<article class="panel metric"><span>${s}</span><strong>${pct(rows[i][fieldIndex.cagr])}</strong><small>CAGR · Sharpe ${num(rows[i][fieldIndex.sharpe])}</small></article>`).join('');$('metrics').innerHTML=strategies.map((s,i)=>{let r=rows[i];return `<tr><td><b>${s}</b></td><td class="num">${pct(r[0])}</td><td class="num">${pct(r[1])}</td><td class="num">${pct(r[2])}</td><td class="num">${num(r[3])}</td><td class="num">${num(r[4])}</td><td class="num">${pct(r[6])}</td><td class="num">${num(r[7])}</td><td class="num">${num(r[8])}</td><td class="num">${pct(r[9])}</td></tr>`}).join('');renderWealth()}function renderWealth(){let si=months.indexOf($('start').value),ei=months.indexOf($('end').value),series=strategies.map(s=>{let w=1;return {s,pts:monthlyBy[s].slice(si,ei+1).map((r,i)=>({x:i,y:(w*=1+r.net_return)}))}});let W=1000,H=330,p=38,max=Math.max(...series.flatMap(s=>s.pts.map(x=>x.y))),min=Math.min(1,...series.flatMap(s=>s.pts.map(x=>x.y))),x=i=>p+i*(W-2*p)/Math.max(1,ei-si),y=v=>H-p-(v-min)*(H-2*p)/Math.max(.001,max-min);$('legend').innerHTML=series.map(s=>`<span><i style="background:${COLORS[s.s]}"></i>${s.s}</span>`).join('');$('wealth').innerHTML=`<svg viewBox="0 0 ${W} ${H}">${series.map(s=>`<polyline fill="none" stroke="${COLORS[s.s]}" stroke-width="2" points="${s.pts.map(q=>`${x(q.x)},${y(q.y)}`).join(' ')}"/>`).join('')}<text x="8" y="18" font-size="11">${num(max,1)}x</text><text x="8" y="${H-10}" font-size="11">${num(min,1)}x</text></svg>`}[$('start'),$('end')].forEach(e=>e.onchange=renderOverview);document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{let n=b.dataset.preset==='all'?months.length:Number(b.dataset.preset);$('start').value=months[Math.max(0,months.length-n)];$('end').value=months.at(-1);renderOverview()});function renderHoldings(){let m=$('holding-month').value,s=$('holding-strategy').value,rows=DATA.holdings.filter(r=>r.holding_month===m&&r.strategy===s).sort((a,b)=>(a.selection_rank??99)-(b.selection_rank??99));$('holding-cards').innerHTML=`<article class="panel"><h3>${s}</h3><p class="fine">Decision ${rows[0]?.decision_month?.slice(0,7)||'—'} · detention ${m.slice(0,7)}</p><div class="ticker-list">${rows.map(r=>`<div class="ticker-row"><span class="ticker">${r.ticker}</span><span>${pct(r.target_weight)}</span></div>`).join('')||'<p>Aucune position : serie de reference.</p>'}</div></article><article class="panel metric"><span>Rendement net du mois</span><strong>${pct(DATA.monthly.find(r=>r.holding_month===m&&r.strategy===s)?.net_return)}</strong><small>Apres couts selon la convention de la strategie</small></article><article class="panel metric"><span>Turnover</span><strong>${pct(DATA.monthly.find(r=>r.holding_month===m&&r.strategy===s)?.turnover)}</strong><small>${DATA.monthly.find(r=>r.holding_month===m&&r.strategy===s)?.n_positions||0} positions</small></article>`}[$('holding-month'),$('holding-strategy')].forEach(e=>e.onchange=renderHoldings);function attrPeriods(){let l=$('attr-level').value;if(l==='total')return ['Toute la periode'];if(l==='year')return [...new Set(months.map(m=>m.slice(0,4)))];return months.map(m=>m.slice(0,7))}function renderAttrPeriods(){options($('attr-period'),attrPeriods());renderAttr()}function renderAttr(){let s=$('attr-strategy').value,l=$('attr-level').value,p=$('attr-period').value,rows=DATA.attribution.filter(r=>r.strategy===s&&(l==='total'||r.holding_month.startsWith(p))),n=l==='total'?months.length:l==='year'?months.filter(m=>m.startsWith(p)).length:1,agg={};for(const r of rows){let a=agg[r.component]??={log:0,simple:0,type:r.component_type};a.log+=r.log_return_contribution;a.simple+=r.simple_return_contribution}let items=Object.entries(agg).map(([c,v])=>({c,...v,annual:12/n*v.log})).sort((a,b)=>Math.abs(b.annual)-Math.abs(a.annual));let total=items.reduce((a,b)=>a+b.annual,0),cagr=Math.expm1(total),top=items.slice(0,24),other=items.slice(24);if(other.length)top.push({c:'Autres',annual:other.reduce((a,b)=>a+b.annual,0),simple:other.reduce((a,b)=>a+b.simple,0)});$('attr-formula').textContent=`Somme des contributions log annualisees = ${pct(total,2)} ; CAGR exact = exp(${num(total,4)}) - 1 = ${pct(cagr,2)}. L'ecart ${pct(cagr-total,2)} est l'effet compose.`;$('attr-table').innerHTML=top.map(x=>`<tr><td>${x.c}</td><td class="num">${pct(x.annual,2)}</td><td class="num">${pct(Math.expm1(x.annual),2)}</td><td class="num">${pct(x.simple,2)}</td></tr>`).join('');renderWaterfall(top,cagr)}function renderWaterfall(items,cagr){let W=1000,H=330,p=45,max=Math.max(.01,...items.map(x=>Math.abs(x.annual))),bw=(W-2*p)/Math.max(1,items.length),y0=H/2,scale=(H/2-p)/max;$('waterfall').innerHTML=`<svg viewBox="0 0 ${W} ${H}"><line x1="${p}" x2="${W-p}" y1="${y0}" y2="${y0}" stroke="#94a3b8"/>${items.map((x,i)=>{let h=Math.abs(x.annual)*scale,y=x.annual>=0?y0-h:y0;return `<rect x="${p+i*bw+2}" y="${y}" width="${Math.max(2,bw-4)}" height="${h}" fill="${x.annual>=0?'#265511':'#802331'}"><title>${x.c}: ${pct(x.annual,2)}</title></rect>`}).join('')}<text x="8" y="18" font-size="11">CAGR ${pct(cagr,2)}</text></svg>`}[$('attr-strategy'),$('attr-period')].forEach(e=>e.onchange=renderAttr);$('attr-level').onchange=renderAttrPeriods;let datasets=DATA.lineage.legacy.datasets;$('datasets').innerHTML=Object.entries(datasets).map(([k,v])=>`<tr><td><b>${k}</b></td><td class="num">${v.rows??'—'}</td><td>${Object.values(v.max_temporal_values||{}).join(' · ')||'—'}</td><td><code>${(v.sha256||'').slice(0,16)}…</code></td></tr>`).join('');let revision=DATA.lineage.revision;if(revision){let conclusion=revision.conclusion;$('revision-summary').textContent=`Legacy identique a 1e-12 : ${conclusion.legacy_identical?'oui':'non'} ; predictions Boosting identiques : ${conclusion.boosting_predictions_identical?'oui':'non'}. Les revisions fournisseur restent visibles ci-dessous.`;$('revision-rows').innerHTML=Object.entries(revision.datasets).map(([k,v])=>`<tr><td><b>${k}</b></td><td class="num">${v.added_rows??'—'}</td><td class="num">${v.removed_rows??'—'}</td><td class="num">${v.changed_common_rows??(v.identical_file?0:'—')}</td></tr>`).join('')}else{$('revision-summary').textContent='Aucun audit de revision associe a ce replay.';$('revision-rows').innerHTML=''}const folds=DATA.boosting.folds;options($('fold'),folds.map(f=>f.fold),f=>'Fold '+f);function metric(f,name){let v=f[name];return v==null?'—':num(v,3)}function renderFold(){let id=Number($('fold').value),f=folds.find(x=>x.fold===id),sh=DATA.boosting.fold_shap.find(x=>x.fold===id);$('fold-kpis').innerHTML=[['AUC test','test_roc_auc'],['NDCG@10 test','test_ndcg_at_10'],['AUC train','train_roc_auc'],['AUC validation','validation_roc_auc']].map(([l,k])=>`<article class="panel metric"><span>${l}</span><strong>${metric(f,k)}</strong><small>${k}</small></article>`).join('');$('fold-calendar').innerHTML=`<tr><td>${f.train_start.slice(0,7)} → ${f.train_cutoff.slice(0,7)}</td><td>${f.validation_start.slice(0,7)} → ${f.validation_end.slice(0,7)}</td><td>${f.test_start.slice(0,7)} → ${f.test_end.slice(0,7)}</td><td class="num">${f.train_rows}</td><td class="num">${f.validation_rows}</td><td class="num">${f.test_rows}</td><td class="num">${f.mature_test_rows}</td><td class="num">${f.score_only_test_rows}</td></tr>`;let top=sh.importance.slice(0,18),mx=top[0]?.mean_abs_shap||1;$('shap-bars').innerHTML=top.map(x=>`<div class="bar-row"><span>${x.feature}</span><div class="track"><div class="fill" style="width:${100*x.mean_abs_shap/mx}%"></div></div><span>${num(x.mean_abs_shap,4)}</span></div>`).join('');let fm=DATA.boosting.prediction_months.filter(x=>x.fold===id).map(x=>x.decision_month),unique=[...new Set(fm)];options($('shap-month'),unique,m=>m.slice(0,7));loadShap()}$('fold').onchange=renderFold;$('shap-month').onchange=loadShap;$('shap-feature').onchange=()=>drawShap(window.SHAPDATA);async function loadShap(){let m=$('shap-month').value;if(!m)return;try{$('shap-status').textContent='Chargement…';let res=await fetch(new URL(`shap/${m.slice(0,7)}.json.gz`,location.href));let buf=await res.arrayBuffer(),bytes=new Uint8Array(buf),txt;if(bytes[0]===31&&bytes[1]===139){let stream=new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));txt=await new Response(stream).text()}else txt=new TextDecoder().decode(bytes);window.SHAPDATA=JSON.parse(txt);options($('shap-feature'),window.SHAPDATA.features);drawShap(window.SHAPDATA);$('shap-status').textContent=`${window.SHAPDATA.rows.length} points · exhaustif · fold ${window.SHAPDATA.rows[0]?.fold}`;}catch(e){$('shap-status').textContent='Erreur SHAP : '+e.message}}function drawShap(d){if(!d)return;let canvas=$('shap-canvas'),ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1;canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;ctx.scale(ratio,ratio);let W=rect.width,H=rect.height,p=35,i=d.features.indexOf($('shap-feature').value),pts=d.rows.map(r=>({x:r.values[i],y:r.shap[i],t:r.ticker})).filter(q=>Number.isFinite(q.x)&&Number.isFinite(q.y)),xs=pts.map(q=>q.x),ys=pts.map(q=>q.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),sx=x=>p+(x-xmin)*(W-2*p)/(xmax-xmin||1),sy=y=>H-p-(y-ymin)*(H-2*p)/(ymax-ymin||1);ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d7e0ea';ctx.beginPath();ctx.moveTo(p,sy(0));ctx.lineTo(W-p,sy(0));ctx.stroke();ctx.fillStyle='#111d55aa';for(const q of pts){ctx.beginPath();ctx.arc(sx(q.x),sy(q.y),2.3,0,Math.PI*2);ctx.fill()}ctx.fillStyle='#475569';ctx.font='11px IBM Plex Mono';ctx.fillText(num(xmin,2),p,H-10);ctx.fillText(num(xmax,2),W-p-34,H-10);ctx.fillText('SHAP',5,15)}$('shap-lede').textContent=`${DATA.boosting.fold_shap.length} ensembles walk-forward. SHAP exhaustif : chaque mois doit contenir exactement autant de points que les predictions de test du mois.`;renderOverview();renderHoldings();renderAttrPeriods();renderFold();</script></body></html>'''


def render(
    *,
    comparison_dir: Path,
    boosting_dir: Path,
    legacy_run_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    payload, sources, shap_samples, predictions = build_payload(
        comparison_dir=comparison_dir,
        boosting_dir=boosting_dir,
        legacy_run_dir=legacy_run_dir,
    )
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    shap_manifest = _write_shap_sidecars(
        samples=shap_samples,
        predictions=predictions,
        output_dir=html_dir / "shap",
    )
    report = html_dir / "alpharank_research_center.html"
    report.write_text(
        HTML.replace(
            "__PAYLOAD__",
            json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "published_same_snapshot_common_dashboard",
                "report": str(report.resolve()),
                "report_sha256": _hash(report),
                "data_lineage": payload["lineage"],
                "calendar": payload["calendar"],
                "shap": shap_manifest,
                "sources": [
                    {"path": str(path.resolve()), "sha256": _hash(path)}
                    for path in sources
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--boosting-dir", type=Path, required=True)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, manifest = render(
        comparison_dir=args.comparison_dir.resolve(),
        boosting_dir=args.boosting_dir.resolve(),
        legacy_run_dir=args.legacy_run_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(report)
    print(manifest)


if __name__ == "__main__":
    main()
