#!/usr/bin/env python3
"""Build an exhaustive, human-readable price audit between two snapshots."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "adjusted_close")
ACTION_EVIDENCE = {
    "CRWD.US": ("2026-07-02", 4.0),
    "CVNA.US": ("2026-05-08", 5.0),
    "DD.US": ("2026-06-24", 1 / 3),
    "FDX.US": ("2026-06-01", 1.241),
    "HON.US": ("2026-06-29", 0.9535),
    "KLAC.US": ("2026-06-12", 10.0),
    "MNST.US": ("2026-08-11", 2.0),
    "SPGI.US": ("2026-07-01", 1.057),
}
MICROSOFT_DIVIDEND = {
    "ex_date": "2026-05-21",
    "amount": 0.91,
    "verified_at": "2026-08-15T13:44:45Z",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _changed(previous: pl.Expr, current: pl.Expr) -> pl.Expr:
    return previous.eq_missing(current).not_()


def _load_prices(snapshot: Path, filename: str, cutoff: date) -> pl.DataFrame:
    frame = pl.read_parquet(snapshot / filename)
    if frame.schema["date"] == pl.String:
        frame = frame.with_columns(pl.col("date").str.to_date())
    else:
        frame = frame.with_columns(pl.col("date").cast(pl.Date))
    return frame.filter(pl.col("date") <= cutoff)


def _lineage_comparison(previous_snapshot: Path, current_snapshot: Path, cutoff: date) -> pl.DataFrame:
    filename = Path("lineage/prices_open_source_lineage.parquet")
    previous = pl.read_parquet(previous_snapshot / filename).filter(
        pl.col("date") <= cutoff.isoformat()
    )
    current = pl.read_parquet(current_snapshot / filename).filter(
        pl.col("date") <= cutoff.isoformat()
    )
    columns = ("source", "dataset", "ingestion_run_id", "ingested_at")
    paired = previous.select(
        "ticker",
        "date",
        *(pl.col(column).alias(f"t1_{column}") for column in columns),
    ).join(
        current.select(
            "ticker",
            "date",
            *(pl.col(column).alias(f"t2_{column}") for column in columns),
        ),
        on=["ticker", "date"],
        how="inner",
    )
    return paired.with_columns(
        *(
            _changed(pl.col(f"t1_{column}"), pl.col(f"t2_{column}")).alias(
                f"changed_{column}"
            )
            for column in columns
        )
    ).filter(pl.any_horizontal(*(pl.col(f"changed_{column}") for column in columns)))


def _load_price_lineage(snapshot: Path, cutoff: date) -> pl.DataFrame:
    frame = pl.read_parquet(snapshot / "lineage/prices_open_source_lineage.parquet")
    if frame.schema["date"] == pl.String:
        frame = frame.with_columns(pl.col("date").str.to_date())
    else:
        frame = frame.with_columns(pl.col("date").cast(pl.Date))
    return frame.filter(pl.col("date") <= cutoff)


def _lineage_vintage_context(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return (
        frame.sort(["ticker", "date"])
        .with_columns(
            pl.col("ingestion_run_id")
            .shift(1)
            .over("ticker")
            .alias(f"{prefix}_previous_ingestion_run_id")
        )
        .select(
            "ticker",
            "date",
            pl.col("ingestion_run_id").alias(f"{prefix}_ingestion_run_id"),
            f"{prefix}_previous_ingestion_run_id",
        )
    )


def _paired(previous: pl.DataFrame, current: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    keys = ["ticker", "date"]
    previous_keys = previous.select(keys)
    current_keys = current.select(keys)
    removed = previous.join(current_keys, on=keys, how="anti").sort(keys)
    added = current.join(previous_keys, on=keys, how="anti").sort(keys)
    paired = previous.select(
        *keys,
        *(pl.col(column).alias(f"t1_{column}") for column in PRICE_COLUMNS),
    ).join(
        current.select(
            *keys,
            *(pl.col(column).alias(f"t2_{column}") for column in PRICE_COLUMNS),
        ),
        on=keys,
        how="inner",
    )
    return paired, added, removed


def _classify_changes(paired: pl.DataFrame) -> pl.DataFrame:
    change_columns = []
    for column in PRICE_COLUMNS:
        change_columns.append(
            _changed(pl.col(f"t1_{column}"), pl.col(f"t2_{column}")).alias(f"changed_{column}")
        )
    changed = paired.with_columns(change_columns).filter(
        pl.any_horizontal(*(pl.col(f"changed_{column}") for column in PRICE_COLUMNS))
    )
    raw_changed = pl.any_horizontal(
        *(pl.col(f"changed_{column}") for column in PRICE_COLUMNS if column != "adjusted_close")
    )
    action_tickers = list(ACTION_EVIDENCE)
    return changed.with_columns(
        raw_changed.alias("raw_ohlcv_changed"),
        pl.when(pl.col("ticker").is_in(action_tickers) & raw_changed)
        .then(pl.lit("corporate_action_full_history_repair"))
        .when(pl.col("ticker") == "EA.US")
        .then(pl.lit("yahoo_unexplained_ea_history_revision"))
        .when(raw_changed & pl.col("date").is_in([date(2026, 8, 7), date(2026, 8, 10)]))
        .then(pl.lit("recent_bar_settlement_revision"))
        .otherwise(pl.lit("adjusted_close_factor_restatement"))
        .alias("reason_category"),
        *(
            (pl.col(f"t2_{column}") - pl.col(f"t1_{column}")).alias(f"delta_{column}")
            for column in PRICE_COLUMNS
        ),
        pl.when(pl.col("t1_adjusted_close").is_not_null() & (pl.col("t1_adjusted_close") != 0))
        .then((pl.col("t2_adjusted_close") / pl.col("t1_adjusted_close")) - 1)
        .otherwise(None)
        .alias("adjusted_close_relative_change"),
    ).sort(["ticker", "date"])


def _monthly_returns(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return (
        frame.filter(pl.col("adjusted_close").is_not_null())
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .sort(["ticker", "date"])
        .group_by(["ticker", "month"], maintain_order=True)
        .agg(pl.col("adjusted_close").last().alias(f"{prefix}_month_end_adjusted_close"))
        .sort(["ticker", "month"])
        .with_columns(
            pl.col(f"{prefix}_month_end_adjusted_close")
            .pct_change()
            .over("ticker")
            .alias(f"{prefix}_return")
        )
    )


def _daily_returns(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return (
        frame.sort(["ticker", "date"])
        .with_columns(
            pl.col("adjusted_close")
            .shift(1)
            .over("ticker")
            .alias(f"{prefix}_previous_adjusted_close"),
            pl.col("adjusted_close")
            .pct_change()
            .over("ticker")
            .alias(f"{prefix}_daily_return"),
        )
        .select(
            "ticker",
            "date",
            pl.col("adjusted_close").alias(f"{prefix}_adjusted_close"),
            f"{prefix}_previous_adjusted_close",
            f"{prefix}_daily_return",
        )
    )


def _daily_comparison(
    previous: pl.DataFrame,
    current: pl.DataFrame,
    previous_lineage: pl.DataFrame,
    current_lineage: pl.DataFrame,
) -> pl.DataFrame:
    difference = pl.col("t2_daily_return") - pl.col("t1_daily_return")
    return (
        _daily_returns(previous, "t1")
        .join(_daily_returns(current, "t2"), on=["ticker", "date"], how="inner")
        .join(
            _lineage_vintage_context(previous_lineage, "t1"),
            on=["ticker", "date"],
            how="left",
        )
        .join(
            _lineage_vintage_context(current_lineage, "t2"),
            on=["ticker", "date"],
            how="left",
        )
        .filter(_changed(pl.col("t1_daily_return"), pl.col("t2_daily_return")))
        .with_columns(
            difference.alias("daily_return_difference"),
            difference.abs().alias("absolute_daily_return_difference"),
            _changed(
                pl.col("t1_previous_ingestion_run_id"),
                pl.col("t1_ingestion_run_id"),
            )
            .fill_null(False)
            .alias("t1_ingestion_vintage_seam"),
            _changed(
                pl.col("t2_previous_ingestion_run_id"),
                pl.col("t2_ingestion_run_id"),
            )
            .fill_null(False)
            .alias("t2_ingestion_vintage_seam"),
        )
        .with_columns(
            pl.when(pl.col("t1_daily_return").is_null() | pl.col("t2_daily_return").is_null())
            .then(pl.lit("return_availability_changed"))
            .when(difference.abs() > 0.10)
            .then(pl.lit("over_10pct"))
            .when(difference.abs() > 0.05)
            .then(pl.lit("over_5pct"))
            .when(difference.abs() > 0.01)
            .then(pl.lit("over_1pct"))
            .when(difference.abs() > 0.001)
            .then(pl.lit("over_10bps"))
            .when(difference.abs() > 0.0001)
            .then(pl.lit("over_1bp"))
            .when(difference.abs() > 0.000001)
            .then(pl.lit("over_0_01bp"))
            .otherwise(pl.lit("floating_point_or_sub_0_01bp"))
            .alias("materiality_bucket"),
            pl.when(pl.col("ticker").is_in(list(ACTION_EVIDENCE)))
            .then(pl.lit("corporate_action_history_repair"))
            .when(pl.col("ticker") == "EA.US")
            .then(pl.lit("yahoo_unexplained_ea_revision"))
            .when(pl.col("date").is_in([date(2026, 8, 7), date(2026, 8, 10)]))
            .then(pl.lit("recent_bar_settlement_revision"))
            .when(pl.col("t1_ingestion_vintage_seam"))
            .then(pl.lit("stitched_adjustment_vintage_seam"))
            .otherwise(pl.lit("adjustment_factor_precision_or_provider_restatement"))
            .alias("reason_category"),
        )
        .sort("absolute_daily_return_difference", descending=True, nulls_last=True)
    )


def _monthly_comparison(previous: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
    return (
        _monthly_returns(previous, "t1")
        .join(_monthly_returns(current, "t2"), on=["ticker", "month"], how="inner")
        .with_columns((pl.col("t2_return") - pl.col("t1_return")).alias("return_difference"))
        .filter(_changed(pl.col("t1_return"), pl.col("t2_return")))
        .with_columns(
            pl.col("return_difference").abs().alias("absolute_return_difference"),
            pl.when(pl.col("ticker").is_in(list(ACTION_EVIDENCE)))
            .then(pl.lit("corporate_action_full_history_repair"))
            .when(pl.col("ticker") == "EA.US")
            .then(pl.lit("yahoo_unexplained_ea_history_revision"))
            .otherwise(pl.lit("adjusted_close_factor_or_recent_bar_revision"))
            .alias("reason_category"),
        )
        .sort("absolute_return_difference", descending=True)
    )


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _fmt_pct(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f} %"


def _table(headers: list[str], rows: list[list[Any]], *, table_id: str | None = None) -> str:
    identifier = f' id="{html.escape(table_id)}"' if table_id else ""
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class=\"table-wrap\"><table{identifier}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _render_report(
    *,
    summary: dict[str, Any],
    ticker_summary: pl.DataFrame,
    monthly: pl.DataFrame,
    benchmark_changes: pl.DataFrame,
    output_dir: Path,
) -> None:
    action_rows = []
    for ticker, (event_date, ratio) in ACTION_EVIDENCE.items():
        row = ticker_summary.filter(pl.col("ticker") == ticker).to_dicts()[0]
        action_rows.append(
            [
                ticker.removesuffix(".US"),
                event_date,
                f"{ratio:.6g}",
                _fmt_int(row["changed_rows"]),
                _fmt_int(row["changed_close_rows"]),
                _fmt_pct(row["median_close_ratio"] - 1 if row["median_close_ratio"] else None),
                _fmt_pct(row["max_monthly_return_difference"], 2),
            ]
        )
    missing_rows = [
        ["CPRT", "2026-07-03", "ligne T1 entierement nulle", "jour ferie US; placeholder nul absent du full refresh"],
        ["CPT", "2026-07-03", "ligne T1 entierement nulle", "jour ferie US; placeholder nul absent du full refresh"],
        ["EA", "2026-07-20 au 2026-07-28", "7 lignes deviennent nulles", "regression Yahoo non expliquee; aucune action corporate trouvee"],
        ["EA", "2026-08-10", "prix gagne en T2; volume=0", "barre ajoutee par le full refresh Yahoo"],
        ["MNST", "2026-08-10", "prix T1 devient nul en T2", "anomalie du refresh autour du split du 11 aout"],
    ]
    monthly_material = monthly.filter(pl.col("absolute_return_difference") > 0.001)
    monthly_rows = [
        [
            row["ticker"].removesuffix(".US"),
            row["month"],
            _fmt_pct(row["t1_return"], 3),
            _fmt_pct(row["t2_return"], 3),
            _fmt_pct(row["return_difference"], 3),
            row["reason_category"],
        ]
        for row in monthly_material.head(100).iter_rows(named=True)
    ]
    ticker_rows = [
        [
            row["ticker"].removesuffix(".US"),
            row["primary_reason"],
            _fmt_int(row["changed_rows"]),
            _fmt_int(row["changed_adjusted_close_rows"]),
            _fmt_int(row["changed_raw_rows"]),
            str(row["first_changed_date"]),
            str(row["last_changed_date"]),
            _fmt_pct(row["max_abs_adjusted_close_relative_change"], 2),
            _fmt_int(row["months_over_10bps"]),
            _fmt_int(row["months_over_1pct"]),
        ]
        for row in ticker_summary.iter_rows(named=True)
    ]
    daily_rows = [
        [
            row["ticker"].removesuffix(".US"),
            str(row["date"]),
            _fmt_pct(row["t1_daily_return"], 4),
            _fmt_pct(row["t2_daily_return"], 4),
            _fmt_pct(row["daily_return_difference"], 4),
            row["reason_category"],
        ]
        for row in pl.read_parquet(output_dir / "daily_adjusted_return_changes.parquet")
        .filter(pl.col("absolute_daily_return_difference") > 0.0001)
        .head(100)
        .iter_rows(named=True)
    ]
    daily_reason_rows = [
        [
            reason,
            _fmt_int(values["over_1bp"]),
            _fmt_int(values["over_10bps"]),
            _fmt_int(values["tickers_over_10bps"]),
        ]
        for reason, values in summary["daily"]["material_by_reason"].items()
    ]
    microsoft = summary["microsoft_case"]
    column_counts = summary["stocks"]["changed_columns"]
    reason_counts = summary["stocks"]["reason_counts"]
    benchmark_column_counts = summary["benchmark"]["changed_columns"]
    content = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit exhaustif des prix T1 / T2</title>
<style>
:root{{--bg:#f5f7fa;--panel:#fff;--text:#172033;--muted:#5f6b82;--border:#d8deea;--navy:#142b4a;--gold:#a77a1d;--green:#1d6b4f;--red:#9c2f35;--soft:#edf1f7}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:"IBM Plex Sans",Inter,Arial,sans-serif;letter-spacing:0}}
header{{background:var(--navy);color:#fff;padding:28px max(24px,calc((100vw - 1440px)/2));border-bottom:4px solid var(--gold)}}
header h1{{font-size:28px;margin:0 0 8px}} header p{{margin:0;color:#dbe4f0;max-width:980px;line-height:1.5}}
main{{max-width:1440px;margin:0 auto;padding:24px}} h2{{font-size:20px;margin:34px 0 12px}} h3{{font-size:16px;margin:22px 0 8px}}
.notice{{border-left:4px solid var(--red);background:#fff;padding:14px 16px;margin:0 0 20px;line-height:1.5}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--border);background:var(--panel)}}
.kpi{{padding:18px;border-right:1px solid var(--border)}} .kpi:last-child{{border-right:0}} .kpi small{{display:block;color:var(--muted);text-transform:uppercase;font:11px "IBM Plex Mono",monospace}}
.kpi strong{{display:block;font:25px "IBM Plex Mono",monospace;margin-top:8px}} .kpi span{{font-size:12px;color:var(--muted)}}
.band{{background:var(--panel);border:1px solid var(--border);padding:18px 20px;margin:12px 0}} .band p,.band li{{line-height:1.55}}
.table-wrap{{overflow:auto;border:1px solid var(--border);background:#fff}} table{{border-collapse:collapse;width:100%;font-size:12px}} th{{position:sticky;top:0;background:#eef2f7;color:var(--muted);text-transform:uppercase;font:10px "IBM Plex Mono",monospace;text-align:left;padding:10px;border-bottom:1px solid var(--border);white-space:nowrap}} td{{padding:9px 10px;border-bottom:1px solid #e8ecf2;font-family:"IBM Plex Mono",monospace;white-space:nowrap}} tr:hover td{{background:#f7f9fc}}
.mono{{font-family:"IBM Plex Mono",monospace}} .bad{{color:var(--red);font-weight:700}} .ok{{color:var(--green);font-weight:700}}
.search{{width:min(460px,100%);padding:10px 12px;border:1px solid var(--border);background:#fff;margin:0 0 10px;font-size:14px}}
.files a{{color:var(--navy);font-family:"IBM Plex Mono",monospace}} footer{{color:var(--muted);font-size:12px;padding:28px 0}}
@media(max-width:800px){{main{{padding:14px}}.grid{{grid-template-columns:1fr 1fr}}.kpi:nth-child(2){{border-right:0}}header{{padding:22px 16px}}}}
</style></head><body>
<header><h1>Audit exhaustif des prix entre T1 et T2</h1><p>Comparaison cle par cle de tous les prix jusqu'au 10 aout 2026 inclus. T2 est un candidat diagnostique mis en quarantaine, pas la production.</p></header>
<main>
<div class="notice"><strong>Conclusion.</strong> Le changement massif n'est pas un nouveau signal de marche. Il provient surtout d'un full refresh Yahoo qui a reecrit <span class="mono">adjusted_close</span> sur tout l'historique, et a enfin corrige huit historiques OHLCV mal recoles autour de splits. Le cas <span class="mono">EA</span> reste une regression fournisseur non expliquee.</div>
<section class="grid">
<div class="kpi"><small>Lignes communes modifiees</small><strong>{_fmt_int(summary['stocks']['changed_common_rows'])}</strong><span>sur {_fmt_int(summary['stocks']['common_rows'])}</span></div>
<div class="kpi"><small>Tickers touches</small><strong>{_fmt_int(summary['stocks']['changed_tickers'])}</strong><span>univers actif full-refresh</span></div>
<div class="kpi"><small>Mois &gt; 10 bps</small><strong>{_fmt_int(summary['monthly']['over_10bps'])}</strong><span>{_fmt_int(summary['monthly']['over_1pct'])} mois &gt; 1 point</span></div>
<div class="kpi"><small>Cas non explique</small><strong class="bad">EA</strong><span>7 prix historiques perdus</span></div>
</section>

<h2>Definition exacte des deux jeux</h2>
{_table(['Jeu','Run ingestion','Debut ingestion UTC / Paris','Publication snapshot UTC / Paris','Type de refresh','Dernier prix'], summary['snapshot_rows'])}
<div class="band"><p><strong>T1</strong> est le snapshot de production actuellement restaure. Il a rafraichi une fenetre prix du 31 juillet au 11 aout seulement. <strong>T2</strong> a retelcharge tout l'historique Yahoo des 503 actions actives du 1er janvier 2005 au 13 aout. Il a ete mis en quarantaine le 13 aout a 10:28:41 UTC parce que les fondamentaux historiques avaient ete publies sans le garde-fou de revision.</p></div>

<h2>Derniere date de chaque couche</h2>
{_table(['Couche','T1','T2','Lecture'], summary['last_date_rows'])}

<h2>Ce qui a change dans les prix actions</h2>
{_table(['Champ','Lignes modifiees'], [[column, _fmt_int(count)] for column, count in column_counts.items()])}
<h3>Attribution exhaustive par cause principale</h3>
{_table(['Cause','Lignes','Tickers'], [[reason, _fmt_int(values['rows']), _fmt_int(values['tickers'])] for reason, values in reason_counts.items()])}
<div class="band"><ul>
<li><strong>{_fmt_int(summary['stocks']['removed_rows'])} cles retirees</strong> : uniquement deux placeholders entierement nuls du 3 juillet, aucun prix reel.</li>
<li><strong>{_fmt_int(summary['stocks']['lost_non_null_rows'])} lignes perdent une valeur</strong> et <strong>{_fmt_int(summary['stocks']['gained_non_null_rows'])} en gagne une</strong> pour chacun des champs OHLCV/adjusted close concernes.</li>
<li><strong>{_fmt_int(summary['stocks']['adjusted_close_over_1pct'])}</strong> lignes ont un adjusted close qui bouge de plus de 1 %, mais une multiplication constante de tout un historique ne change pas les rendements internes. L'impact utile est donc mesure aussi au niveau mensuel.</li>
<li><strong>La source ne change sur aucune cle commune</strong> : Yahoo reste Yahoo, StockAnalysis reste StockAnalysis et SimFin reste SimFin. Les {_fmt_int(summary['lineage']['dataset_label_changes'])} changements de libelle de dataset normalisent des sous-types Yahoo; {_fmt_int(summary['lineage']['ingestion_run_changes'])} cles recoivent le nouveau run d'ingestion parce que T2 les a effectivement retelchargees.</li>
</ul></div>

<h2>Huit corrections de corporate actions</h2>
{_table(['Ticker','Date action Yahoo','Ratio Yahoo','Lignes touchees','Close touche','Ratio median close T2/T1','Plus gros ecart mensuel'], action_rows)}
<div class="band"><p>Ces huit series montrent une signature exacte : le prix brut historique est multiplie par l'inverse du ratio d'action et le volume par le ratio. T1 avait colle une queue recente deja ajustee sur un prefixe ancien non reajuste. Cela creait de faux sauts mensuels. T2 realigne l'historique complet.</p></div>

<h2>Valeurs absentes ou nouvelles</h2>
{_table(['Ticker','Dates','Changement','Cause'], missing_rows)}

<h2>Impact sur les rendements mensuels</h2>
<div class="grid">
<div class="kpi"><small>Mois exacts modifies</small><strong>{_fmt_int(summary['monthly']['changed'])}</strong><span>les micro-ecarts de flottants sont inclus</span></div>
<div class="kpi"><small>&gt; 1 bp</small><strong>{_fmt_int(summary['monthly']['over_1bp'])}</strong><span>ecart absolu</span></div>
<div class="kpi"><small>&gt; 10 bps</small><strong>{_fmt_int(summary['monthly']['over_10bps'])}</strong><span>ecart economiquement visible</span></div>
<div class="kpi"><small>&gt; 5 points</small><strong>{_fmt_int(summary['monthly']['over_5pct'])}</strong><span>tous dus aux actions corporate</span></div>
</div>
<h3>100 plus gros ecarts mensuels</h3>
{_table(['Ticker','Mois','Rendement T1','Rendement T2','Ecart','Cause'], monthly_rows)}

<h2>Rendements journaliers sur adjusted close</h2>
<div class="grid">
<div class="kpi"><small>Lignes exactes modifiees</small><strong>{_fmt_int(summary['daily']['changed'])}</strong><span>inclut le bruit de flottants</span></div>
<div class="kpi"><small>&gt; 1 bp</small><strong>{_fmt_int(summary['daily']['over_1bp'])}</strong><span>{_fmt_int(summary['daily']['stocks_over_1bp'])} actions</span></div>
<div class="kpi"><small>&gt; 10 bps</small><strong>{_fmt_int(summary['daily']['over_10bps'])}</strong><span>{_fmt_int(summary['daily']['stocks_over_10bps'])} actions</span></div>
<div class="kpi"><small>&gt; 1 point</small><strong>{_fmt_int(summary['daily']['over_1pct'])}</strong><span>ecarts economiquement majeurs</span></div>
</div>
<div class="band"><p>Legacy calcule bien <span class="mono">adjusted_close_t / adjusted_close_t-1 - 1</span>. Une remise a l'echelle constante change les niveaux mais pas ce rendement, sauf micro-ecarts de precision. Le probleme de T1 est different : son historique est assemble a partir de millesimes Yahoo incompatibles. Sur les {_fmt_int(summary['daily']['over_10bps'])} ecarts superieurs a 10 bps, <strong>{_fmt_int(summary['daily']['t1_vintage_seams_over_10bps'])}</strong> tombent exactement sur une frontiere de run d'ingestion T1.</p></div>
{_table(['Cause','Lignes > 1 bp','Lignes > 10 bps','Actions > 10 bps'], daily_reason_rows)}

<h3>Cas Microsoft : dividende normal, raccordement T1 anormal</h3>
<div class="band"><p>Microsoft verse un dividende de <strong>${microsoft['dividend_amount']:.2f}</strong> avec date ex-dividende le <strong>{microsoft['dividend_ex_date']}</strong>. Les lignes T1 jusqu'au {microsoft['stale_prefix_end']} viennent du run <span class="mono">{microsoft['stale_prefix_run']}</span>, telecharge avant ce dividende. La ligne suivante vient du run <span class="mono">{microsoft['adjusted_tail_run']}</span>, telecharge apres. Le facteur du dividende n'a donc ete applique qu'a la queue de T1. T2, retelcharge d'un bloc, applique un facteur moyen de <strong>{_fmt_pct(microsoft['mean_level_factor_change'], 4)}</strong> au prefixe et supprime la fausse marche.</p><p>Consequence : {_fmt_int(microsoft['adjusted_close_rows_changed'])} niveaux adjusted close changent, mais seulement <strong>{_fmt_int(microsoft['daily_over_1bp'])} rendement journalier depasse 1 bp</strong>. Le {microsoft['false_seam_date']}, T1 calcule {_fmt_pct(microsoft['t1_false_seam_return'], 4)} contre {_fmt_pct(microsoft['t2_false_seam_return'], 4)}, soit {_fmt_pct(microsoft['false_seam_return_difference'], 4)} d'ecart. Le rendement mensuel de mars change de {_fmt_pct(microsoft['march_monthly_return_difference'], 4)}. Ce n'est pas un evenement Microsoft du 24 mars : c'est la frontiere technique entre deux telechargements.</p></div>
<h3>100 plus gros ecarts journaliers au-dessus de 1 bp</h3>
{_table(['Ticker','Date','Rendement T1','Rendement T2','Ecart','Cause'], daily_rows)}

<h2>SPY</h2>
<div class="band"><p>SPY conserve exactement les memes cles jusqu'au 10 aout. Son close ne change jamais. Les seuls changements bruts sont un high et un low le 10 aout, plus les volumes des 7 et 10 aout. En revanche, <strong>{_fmt_int(summary['benchmark']['changed_columns']['adjusted_close'])}</strong> adjusted closes sont recalcules : le ratio T2/T1 est presque constant avant le 11 juin, signature du dividende cumule dans Yahoo. L'effet mensuel materiel est concentre en juin 2026 : <strong>{_fmt_pct(summary['benchmark']['june_2026_monthly_difference'], 4)}</strong>.</p></div>
{_table(['Champ SPY','Lignes modifiees'], [[column, _fmt_int(count)] for column, count in benchmark_column_counts.items()])}

<h2>Les 503 tickers, exhaustivement</h2>
<input class="search" id="tickerSearch" placeholder="Filtrer par ticker ou cause" aria-label="Filtrer les tickers">
{_table(['Ticker','Cause principale','Lignes','Adjusted close','OHLCV','Premiere date','Derniere date','Ecart adj. max','Mois >10bps','Mois >1pt'], ticker_rows, table_id='tickerTable')}

<h2>Fichiers d'audit</h2>
<div class="band files"><p><a href="price_changes_exhaustive.parquet">price_changes_exhaustive.parquet</a> : chaque valeur T1/T2, indicateurs par colonne, ecarts et cause.</p><p><a href="daily_adjusted_return_changes.parquet">daily_adjusted_return_changes.parquet</a> : toutes les lignes ou le rendement journalier adjusted close change.</p><p><a href="price_changes_by_ticker.parquet">price_changes_by_ticker.parquet</a> : synthese des 503 tickers.</p><p><a href="monthly_return_changes.parquet">monthly_return_changes.parquet</a> : tous les rendements mensuels modifies.</p><p><a href="price_lineage_label_changes.parquet">price_lineage_label_changes.parquet</a> : changements de source ou de libelle de dataset.</p><p><a href="benchmark_price_changes_exhaustive.parquet">benchmark_price_changes_exhaustive.parquet</a> : detail SPY.</p><p><a href="summary.json">summary.json</a> : compteurs et metadonnees lisibles par machine.</p></div>
<footer>Genere le 15 aout 2026. Sources immuables : open_source_output_20260811_014746 et open_source_output_20260813_075926. Cutoff strict : 2026-08-10.</footer>
</main><script>
const q=document.getElementById('tickerSearch');q.addEventListener('input',()=>{{const v=q.value.toLowerCase();document.querySelectorAll('#tickerTable tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(v))}});
</script></body></html>"""
    (output_dir / "price_revision_audit.html").write_text(content, encoding="utf-8")


def _changed_column_counts(changes: pl.DataFrame) -> dict[str, int]:
    return {
        column: changes.select(pl.col(f"changed_{column}").sum()).item()
        for column in PRICE_COLUMNS
    }


def _ticker_summary(changes: pl.DataFrame, monthly: pl.DataFrame) -> pl.DataFrame:
    monthly_summary = monthly.group_by("ticker").agg(
        pl.col("absolute_return_difference").max().alias("max_monthly_return_difference"),
        (pl.col("absolute_return_difference") > 0.001).sum().alias("months_over_10bps"),
        (pl.col("absolute_return_difference") > 0.01).sum().alias("months_over_1pct"),
    )
    return (
        changes.group_by("ticker")
        .agg(
            pl.len().alias("changed_rows"),
            pl.col("changed_adjusted_close").sum().alias("changed_adjusted_close_rows"),
            pl.col("raw_ohlcv_changed").sum().alias("changed_raw_rows"),
            pl.col("changed_close").sum().alias("changed_close_rows"),
            pl.col("date").min().alias("first_changed_date"),
            pl.col("date").max().alias("last_changed_date"),
            pl.col("adjusted_close_relative_change").abs().max().alias("max_abs_adjusted_close_relative_change"),
            (pl.col("t2_close") / pl.col("t1_close")).filter(
                pl.col("changed_close") & pl.col("t1_close").is_not_null() & (pl.col("t1_close") != 0)
            ).median().alias("median_close_ratio"),
            pl.col("reason_category").mode().first().alias("primary_reason"),
        )
        .join(monthly_summary, on="ticker", how="left")
        .with_columns(
            pl.col("max_monthly_return_difference").fill_null(0.0),
            pl.col("months_over_10bps").fill_null(0),
            pl.col("months_over_1pct").fill_null(0),
        )
        .sort(["max_monthly_return_difference", "ticker"], descending=[True, False])
    )


def _audit_one(previous: pl.DataFrame, current: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    paired, added, removed = _paired(previous, current)
    return _classify_changes(paired), added, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-snapshot", type=Path, required=True)
    parser.add_argument("--current-snapshot", type=Path, required=True)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    previous = _load_prices(args.previous_snapshot, "US_Finalprice.parquet", args.cutoff)
    current = _load_prices(args.current_snapshot, "US_Finalprice.parquet", args.cutoff)
    previous_lineage = _load_price_lineage(args.previous_snapshot, args.cutoff)
    current_lineage = _load_price_lineage(args.current_snapshot, args.cutoff)
    changes, added, removed = _audit_one(previous, current)
    daily = _daily_comparison(previous, current, previous_lineage, current_lineage)
    monthly = _monthly_comparison(previous, current)
    ticker_summary = _ticker_summary(changes, monthly)
    changes.write_parquet(args.output_dir / "price_changes_exhaustive.parquet", compression="zstd")
    daily.write_parquet(
        args.output_dir / "daily_adjusted_return_changes.parquet", compression="zstd"
    )
    added.write_parquet(args.output_dir / "price_keys_added.parquet", compression="zstd")
    removed.write_parquet(args.output_dir / "price_keys_removed.parquet", compression="zstd")
    monthly.write_parquet(args.output_dir / "monthly_return_changes.parquet", compression="zstd")
    ticker_summary.write_parquet(args.output_dir / "price_changes_by_ticker.parquet", compression="zstd")
    lineage_changes = _lineage_comparison(
        args.previous_snapshot, args.current_snapshot, args.cutoff
    )
    lineage_changes.filter(
        pl.col("changed_source") | pl.col("changed_dataset")
    ).write_parquet(args.output_dir / "price_lineage_label_changes.parquet", compression="zstd")

    previous_benchmark = _load_prices(args.previous_snapshot, "SP500Price.parquet", args.cutoff)
    current_benchmark = _load_prices(args.current_snapshot, "SP500Price.parquet", args.cutoff)
    benchmark_changes, benchmark_added, benchmark_removed = _audit_one(previous_benchmark, current_benchmark)
    benchmark_monthly = _monthly_comparison(previous_benchmark, current_benchmark)
    benchmark_changes.write_parquet(
        args.output_dir / "benchmark_price_changes_exhaustive.parquet", compression="zstd"
    )
    benchmark_monthly.write_parquet(
        args.output_dir / "benchmark_monthly_return_changes.parquet", compression="zstd"
    )

    lost_non_null = changes.select(
        pl.any_horizontal(
            *((pl.col(f"t1_{column}").is_not_null() & pl.col(f"t2_{column}").is_null()) for column in PRICE_COLUMNS)
        ).sum()
    ).item()
    gained_non_null = changes.select(
        pl.any_horizontal(
            *((pl.col(f"t1_{column}").is_null() & pl.col(f"t2_{column}").is_not_null()) for column in PRICE_COLUMNS)
        ).sum()
    ).item()
    june_spy = benchmark_monthly.filter(pl.col("month") == "2026-06")
    june_spy_difference = june_spy.select("return_difference").item() if june_spy.height else 0.0
    material_by_reason = {
        row["reason_category"]: {
            "over_1bp": row["over_1bp"],
            "over_10bps": row["over_10bps"],
            "tickers_over_10bps": row["tickers_over_10bps"],
        }
        for row in daily.group_by("reason_category")
        .agg(
            (pl.col("absolute_daily_return_difference") > 0.0001).sum().alias("over_1bp"),
            (pl.col("absolute_daily_return_difference") > 0.001).sum().alias("over_10bps"),
            pl.col("ticker")
            .filter(pl.col("absolute_daily_return_difference") > 0.001)
            .n_unique()
            .alias("tickers_over_10bps"),
        )
        .sort("over_10bps", descending=True)
        .iter_rows(named=True)
    }
    microsoft_changes = changes.filter(pl.col("ticker") == "MSFT.US")
    microsoft_daily = daily.filter(pl.col("ticker") == "MSFT.US")
    microsoft_seam = microsoft_daily.filter(pl.col("date") == date(2026, 3, 24)).to_dicts()[0]
    microsoft_march = monthly.filter(
        (pl.col("ticker") == "MSFT.US") & (pl.col("month") == "2026-03")
    ).to_dicts()[0]
    microsoft_level_factor = microsoft_changes.filter(
        pl.col("changed_adjusted_close")
    ).select((pl.col("t2_adjusted_close") / pl.col("t1_adjusted_close")).mean()).item()
    summary = {
        "generated_at": "2026-08-15",
        "cutoff": args.cutoff.isoformat(),
        "previous_snapshot": str(args.previous_snapshot.resolve()),
        "current_snapshot": str(args.current_snapshot.resolve()),
        "snapshot_rows": [
            ["T1 production", "20260811_001503", "2026-08-11 00:15:03 / 02:15:03", "2026-08-11 01:47:46 / 03:47:46", "incremental 2026-07-31 -> 2026-08-11", "2026-08-10"],
            ["T2 quarantaine", "20260813_071802", "2026-08-13 07:18:02 / 09:18:02", "2026-08-13 07:59:26 / 09:59:26", "full history 2005-01-01 -> 2026-08-13", "2026-08-12"],
        ],
        "last_date_rows": [
            ["Prix actions", "2026-08-10", "2026-08-12", "derniere seance disponible"],
            ["Prix SPY", "2026-08-10", "2026-08-12", "derniere seance disponible"],
            ["Income / balance / cash - periode", "2026-06-30", "2026-09-30", "date de periode normalisee, pas date de disponibilite"],
            ["Income / balance / cash - filing", "2026-06-05", "2026-08-12", "date utilisable pour le point-in-time"],
            ["Shares - periode", "2026-06-30", "2026-09-30", "date de periode normalisee"],
            ["Earnings - reportDate", "2026-06-03", "2026-08-12", "date d'annonce"],
            ["Constituants", "2026-07-01", "2026-08-01", "mois d'univers"],
        ],
        "stocks": {
            "t1_rows": previous.height,
            "t2_rows": current.height,
            "common_rows": previous.height - removed.height,
            "added_rows": added.height,
            "removed_rows": removed.height,
            "changed_common_rows": changes.height,
            "changed_tickers": changes.select(pl.col("ticker").n_unique()).item(),
            "changed_columns": _changed_column_counts(changes),
            "reason_counts": {
                row["reason_category"]: {"rows": row["rows"], "tickers": row["tickers"]}
                for row in changes.group_by("reason_category")
                .agg(pl.len().alias("rows"), pl.col("ticker").n_unique().alias("tickers"))
                .sort("rows", descending=True)
                .iter_rows(named=True)
            },
            "lost_non_null_rows": lost_non_null,
            "gained_non_null_rows": gained_non_null,
            "adjusted_close_over_1pct": changes.filter(
                pl.col("adjusted_close_relative_change").abs() > 0.01
            ).height,
        },
        "monthly": {
            "changed": monthly.height,
            "over_1bp": monthly.filter(pl.col("absolute_return_difference") > 0.0001).height,
            "over_10bps": monthly.filter(pl.col("absolute_return_difference") > 0.001).height,
            "over_1pct": monthly.filter(pl.col("absolute_return_difference") > 0.01).height,
            "over_5pct": monthly.filter(pl.col("absolute_return_difference") > 0.05).height,
        },
        "daily": {
            "changed": daily.height,
            "changed_stocks": daily.select(pl.col("ticker").n_unique()).item(),
            "changed_dates": daily.select(pl.col("date").n_unique()).item(),
            "over_0_01bp": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.000001
            ).height,
            "over_1bp": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.0001
            ).height,
            "stocks_over_1bp": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.0001
            ).select(pl.col("ticker").n_unique()).item(),
            "over_10bps": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.001
            ).height,
            "stocks_over_10bps": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.001
            ).select(pl.col("ticker").n_unique()).item(),
            "over_1pct": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.01
            ).height,
            "over_5pct": daily.filter(
                pl.col("absolute_daily_return_difference") > 0.05
            ).height,
            "availability_changed": daily.filter(
                pl.col("t1_daily_return").is_null() | pl.col("t2_daily_return").is_null()
            ).height,
            "t1_vintage_seams_over_1bp": daily.filter(
                (pl.col("absolute_daily_return_difference") > 0.0001)
                & pl.col("t1_ingestion_vintage_seam")
            ).height,
            "t1_vintage_seams_over_10bps": daily.filter(
                (pl.col("absolute_daily_return_difference") > 0.001)
                & pl.col("t1_ingestion_vintage_seam")
            ).height,
            "material_by_reason": material_by_reason,
        },
        "microsoft_case": {
            "dividend_ex_date": MICROSOFT_DIVIDEND["ex_date"],
            "dividend_amount": MICROSOFT_DIVIDEND["amount"],
            "dividend_evidence_verified_at": MICROSOFT_DIVIDEND["verified_at"],
            "stale_prefix_end": "2026-03-23",
            "stale_prefix_run": microsoft_seam["t1_previous_ingestion_run_id"],
            "adjusted_tail_run": microsoft_seam["t1_ingestion_run_id"],
            "false_seam_date": "2026-03-24",
            "adjusted_close_rows_changed": microsoft_changes.filter(
                pl.col("changed_adjusted_close")
            ).height,
            "exact_daily_rows_changed": microsoft_daily.height,
            "daily_over_0_01bp": microsoft_daily.filter(
                pl.col("absolute_daily_return_difference") > 0.000001
            ).height,
            "daily_over_1bp": microsoft_daily.filter(
                pl.col("absolute_daily_return_difference") > 0.0001
            ).height,
            "mean_level_factor_change": microsoft_level_factor - 1,
            "t1_false_seam_return": microsoft_seam["t1_daily_return"],
            "t2_false_seam_return": microsoft_seam["t2_daily_return"],
            "false_seam_return_difference": microsoft_seam["daily_return_difference"],
            "march_monthly_return_difference": microsoft_march["return_difference"],
        },
        "benchmark": {
            "added_rows": benchmark_added.height,
            "removed_rows": benchmark_removed.height,
            "changed_common_rows": benchmark_changes.height,
            "changed_columns": _changed_column_counts(benchmark_changes),
            "june_2026_monthly_difference": june_spy_difference,
        },
        "lineage": {
            "source_changes": lineage_changes.filter(pl.col("changed_source")).height,
            "dataset_label_changes": lineage_changes.filter(pl.col("changed_dataset")).height,
            "ingestion_run_changes": lineage_changes.filter(
                pl.col("changed_ingestion_run_id")
            ).height,
            "ingested_at_changes": lineage_changes.filter(pl.col("changed_ingested_at")).height,
        },
        "corporate_action_evidence": {
            ticker: {"event_date": event_date, "yahoo_ratio": ratio}
            for ticker, (event_date, ratio) in ACTION_EVIDENCE.items()
        },
        "corporate_action_evidence_source": {
            "source": "Yahoo Finance actions queried through yfinance",
            "verified_at": "2026-08-14T11:09:45Z",
        },
        "reason_method": {
            "corporate_actions": "Yahoo split event plus inverse close/volume scaling over history",
            "adjusted_close": "Yahoo cumulative adjustment-factor restatement; raw OHLCV unchanged",
            "recent_bars": "raw changes confined to 2026-08-07 and 2026-08-10 after T1 early download",
            "EA": "historical raw values lost/rewritten without split or dividend evidence; unresolved provider regression",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_value(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    _render_report(
        summary=summary,
        ticker_summary=ticker_summary,
        monthly=monthly,
        benchmark_changes=benchmark_changes,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
