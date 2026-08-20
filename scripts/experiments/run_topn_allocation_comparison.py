#!/usr/bin/env python3
"""Compare top-5 and top-10 portfolios from frozen OOS predictions."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from alpharank.multihorizon.confirmation import paired_block_bootstrap  # noqa: E402
from alpharank.multihorizon.risk import build_risk_weighted_backtest  # noqa: E402
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts  # noqa: E402
from alpharank.portfolio.comparison import reference_monthly_series  # noqa: E402
from alpharank.portfolio.performance import (  # noqa: E402
    annual_returns as common_annual_returns,
)
from alpharank.portfolio.performance import (
    legacy_report_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPEC = Path(
    "configs/research/legacy_ema_top5_vs_top10_quarantine_v7.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/multihorizon_boosting/"
    "legacy_ema_top5_vs_top10_quarantine_v7_20260726"
)

ALLOCATION_SUFFIXES = (
    ("equal", "Équipondéré", None, False),
    ("inverse_vol_h1", "Inverse vol 1m", "predicted_realized_volatility_1m", False),
    ("inverse_vol_h3", "Inverse vol 3m", "predicted_realized_volatility_3m", False),
    ("inverse_vol_h6", "Inverse vol 6m", "predicted_realized_volatility_6m", False),
    (
        "inverse_downside_h1",
        "Inverse downside 1m",
        "predicted_daily_downside_1m",
        False,
    ),
    (
        "inverse_downside_h3",
        "Inverse downside 3m",
        "predicted_daily_downside_3m",
        False,
    ),
    (
        "inverse_downside_h6",
        "Inverse downside 6m",
        "predicted_daily_downside_6m",
        False,
    ),
    (
        "inverse_vol_h3_sector2",
        "Inverse vol 3m + secteur",
        "predicted_realized_volatility_3m",
        True,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _legacy_monthly(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("model") == "Combined_Frequency")
        .select(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.col("monthly_return").alias("legacy_return"),
        )
        .unique("holding_month")
    )


def _build_allocations(
    *,
    predictions: pl.DataFrame,
    general: pl.DataFrame,
    specification: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    transaction_cost_bps = float(
        specification["trading"]["transaction_cost_bps_times_turnover"]
    )
    allocation = specification["allocations"]
    parts: list[pl.DataFrame] = []
    holding_parts: list[pl.DataFrame] = []
    for top_n in specification["experiment"]["top_n_values"]:
        for suffix, _, risk_column, sector_constrained in ALLOCATION_SUFFIXES:
            strategy = f"alpha_top{int(top_n)}_{suffix}"
            monthly, holdings = build_risk_weighted_backtest(
                predictions,
                general=general,
                strategy=strategy,
                top_n=int(top_n),
                risk_column=risk_column,
                maximum_weight=float(allocation["maximum_stock_weight"]),
                maximum_names_per_sector=(
                    int(allocation["maximum_names_per_sector"])
                    if sector_constrained
                    else None
                ),
                maximum_sector_weight=(
                    float(allocation["maximum_sector_weight"])
                    if sector_constrained
                    else None
                ),
                transaction_cost_bps=transaction_cost_bps,
            )
            parts.append(monthly)
            holding_parts.append(holdings)
    return (
        pl.concat(parts).sort(["strategy", "holding_month"]),
        pl.concat(holding_parts, how="diagonal_relaxed").sort(
            ["strategy", "holding_month", "selection_rank"]
        ),
    )


def _assert_top5_replay(
    rebuilt: pl.DataFrame,
    source_monthly: pl.DataFrame,
) -> float:
    left = rebuilt.filter(pl.col("strategy").str.starts_with("alpha_top5_"))
    joined = left.join(
        source_monthly.select(
            "strategy",
            "holding_month",
            pl.col("net_return").alias("source_net_return"),
        ),
        on=["strategy", "holding_month"],
        how="inner",
    )
    if joined.height != left.height:
        raise ValueError(
            f"Top-5 replay coverage mismatch: {joined.height} != {left.height}"
        )
    maximum_error = float(
        (joined["net_return"] - joined["source_net_return"]).abs().max()
    )
    if maximum_error > 1e-12:
        raise ValueError(f"Top-5 replay mismatch: maximum error={maximum_error}")
    return maximum_error


def _performance(monthly: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in monthly.partition_by("strategy", maintain_order=True):
        metrics = legacy_report_statistics(
            frame["net_return"].to_numpy(),
            holding_months=frame["holding_month"].to_list(),
        )
        rows.append(
            {
                "series": frame["strategy"][0],
                "role": "allocation",
                "top_n": int(frame["n_positions"].median()),
                "start": frame["holding_month"].min(),
                "end": frame["holding_month"].max(),
                "months": frame.height,
                "average_turnover": float(frame["turnover"].mean()),
                "average_maximum_position_weight": float(
                    frame["maximum_position_weight"].mean()
                ),
                "maximum_sector_weight": float(
                    frame["maximum_sector_weight"].max()
                ),
                **metrics,
            }
        )
    reference = monthly.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).sort("holding_month")
    for series, column in (
        ("Legacy", "legacy_return"),
        ("SPY total return", "benchmark_return"),
    ):
        rows.append(
            {
                "series": series,
                "role": "reference",
                "top_n": None,
                "start": reference["holding_month"].min(),
                "end": reference["holding_month"].max(),
                "months": reference.height,
                "average_turnover": None,
                "average_maximum_position_weight": None,
                "maximum_sector_weight": None,
                **legacy_report_statistics(
                    reference[column].to_numpy(),
                    holding_months=reference["holding_month"].to_list(),
                ),
            }
        )
    return pl.DataFrame(rows).sort(["role", "series"])


def _annual_returns(monthly: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in monthly.partition_by("strategy", maintain_order=True):
        yearly = common_annual_returns(
            frame["net_return"].to_numpy(),
            holding_months=frame["holding_month"].to_list(),
        )
        for year in yearly.iter_rows(named=True):
            rows.append(
                {
                    "year": int(year["year"]),
                    "months": int(year["months"]),
                    "series": frame["strategy"][0],
                    "annual_return": float(year["annual_return"]),
                }
            )
    reference = monthly.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).sort("holding_month")
    for series, column in (
        ("Legacy", "legacy_return"),
        ("SPY total return", "benchmark_return"),
    ):
        yearly = common_annual_returns(
            reference[column].to_numpy(),
            holding_months=reference["holding_month"].to_list(),
        )
        for year in yearly.iter_rows(named=True):
            rows.append(
                {
                    "year": int(year["year"]),
                    "months": int(year["months"]),
                    "series": series,
                    "annual_return": float(year["annual_return"]),
                }
            )
    return pl.DataFrame(rows).sort(["year", "series"])


def _rank_bucket_diagnostics(holdings: pl.DataFrame) -> pl.DataFrame:
    """Measure the gross contribution of ranks 1-5 and 6-10."""

    equal_top10 = holdings.filter(
        pl.col("strategy") == "alpha_top10_equal"
    )
    rows: list[dict[str, Any]] = []
    for minimum_rank, maximum_rank, bucket in (
        (1, 5, "ranks_1_5"),
        (6, 10, "ranks_6_10"),
    ):
        selected = equal_top10.filter(
            pl.col("selection_rank").is_between(
                minimum_rank,
                maximum_rank,
            )
        )
        monthly = (
            selected.group_by("holding_month")
            .agg(pl.col("future_return_1m").mean().alias("return"))
            .sort("holding_month")
        )
        stats = legacy_report_statistics(
            monthly["return"].to_numpy(),
            holding_months=monthly["holding_month"].to_list(),
        )
        rows.append(
            {
                "bucket": bucket,
                "minimum_rank": minimum_rank,
                "maximum_rank": maximum_rank,
                "stock_month_mean_return": float(
                    selected["future_return_1m"].mean()
                ),
                "stock_month_mean_excess_return": float(
                    selected["future_excess_return_1m"].mean()
                ),
                "mean_raw_alpha_score": float(selected["score"].mean()),
                **stats,
            }
        )
    return pl.DataFrame(rows)


def _cost_sensitivity(
    monthly: pl.DataFrame,
    cost_bps_values: list[float],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in monthly.partition_by("strategy", maintain_order=True):
        for cost_bps in cost_bps_values:
            net = (
                frame["gross_return"].to_numpy()
                - frame["turnover"].to_numpy() * float(cost_bps) / 10_000.0
            )
            metrics = legacy_report_statistics(
                net,
                holding_months=frame["holding_month"].to_list(),
            )
            rows.append(
                {
                    "strategy": frame["strategy"][0],
                    "cost_bps": float(cost_bps),
                    **metrics,
                }
            )
    return pl.DataFrame(rows).sort(["strategy", "cost_bps"])


def _bootstrap(
    monthly: pl.DataFrame,
    *,
    samples: int,
    block_months: int,
    seed: int,
) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for suffix, _, _, _ in ALLOCATION_SUFFIXES:
        top5 = monthly.filter(
            pl.col("strategy") == f"alpha_top5_{suffix}"
        ).select(
            "decision_month",
            pl.col("net_return").alias("top5_return"),
        )
        top10_name = f"alpha_top10_{suffix}"
        frame = (
            monthly.filter(pl.col("strategy") == top10_name)
            .join(top5, on="decision_month", how="inner")
            .sort("decision_month")
        )
        result = paired_block_bootstrap(
            frame,
            comparator_columns={
                f"alpha_top5_{suffix}": "top5_return",
                "Legacy": "legacy_return",
                "SPY total return": "benchmark_return",
            },
            samples=samples,
            block_months=block_months,
            seed=seed,
        ).with_columns(pl.lit(top10_name).alias("strategy"))
        parts.append(result)
    return pl.concat(parts, how="diagonal_relaxed")


def _promotion_gates(
    performance: pl.DataFrame,
    costs: pl.DataFrame,
    bootstrap: pl.DataFrame,
) -> dict[str, bool]:
    top5 = performance.filter(
        pl.col("series") == "alpha_top5_equal"
    ).row(0, named=True)
    top10 = performance.filter(
        pl.col("series") == "alpha_top10_equal"
    ).row(0, named=True)
    top5_50 = costs.filter(
        (pl.col("strategy") == "alpha_top5_equal")
        & (pl.col("cost_bps") == 50.0)
    ).row(0, named=True)
    top10_50 = costs.filter(
        (pl.col("strategy") == "alpha_top10_equal")
        & (pl.col("cost_bps") == 50.0)
    ).row(0, named=True)
    paired = bootstrap.filter(
        (pl.col("strategy") == "alpha_top10_equal")
        & (pl.col("comparator") == "alpha_top5_equal")
    ).row(0, named=True)
    return {
        "sharpe_higher": top10["sharpe"] > top5["sharpe"],
        "drawdown_improves_5pp": (
            top10["max_drawdown"] - top5["max_drawdown"] >= 0.05
        ),
        "cagr_loss_within_3pp": top10["cagr"] >= top5["cagr"] - 0.03,
        "bootstrap_sharpe_ci_low_positive": (
            paired["sharpe_difference_ci_low"] > 0.0
        ),
        "sharpe_higher_at_50bps": top10_50["sharpe"] > top5_50["sharpe"],
    }


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}%".replace(".", ",")


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _label(series: str) -> str:
    if series in {"Legacy", "SPY total return"}:
        return series
    top_n = "Top 10" if "top10" in series else "Top 5"
    suffix = series.split("_", 2)[2]
    label = next(
        item[1] for item in ALLOCATION_SUFFIXES if item[0] == suffix
    )
    return f"{top_n} · {label}"


def _render_html(
    *,
    performance: pl.DataFrame,
    annual: pl.DataFrame,
    bootstrap: pl.DataFrame,
    gates: dict[str, bool],
    rank_buckets: pl.DataFrame,
    top5_replay_error: float,
    output_path: Path,
) -> None:
    performance_rows = []
    order = [
        series
        for suffix, _, _, _ in ALLOCATION_SUFFIXES
        for series in (
            f"alpha_top5_{suffix}",
            f"alpha_top10_{suffix}",
        )
    ] + ["Legacy", "SPY total return"]
    for series in order:
        row = performance.filter(pl.col("series") == series).row(
            0,
            named=True,
        )
        performance_rows.append(
            [
                html.escape(_label(series)),
                _pct(row["cagr"]),
                _number(row["sharpe"]),
                _pct(row["annualized_volatility"]),
                _pct(row["max_drawdown"]),
                (
                    f'{row["worst_full_calendar_year"]} · '
                    f'{_pct(row["worst_full_calendar_year_return"])}'
                ),
                _pct(row["average_turnover"]),
            ]
        )

    delta_rows = []
    for suffix, label, _, _ in ALLOCATION_SUFFIXES:
        top5 = performance.filter(
            pl.col("series") == f"alpha_top5_{suffix}"
        ).row(0, named=True)
        top10 = performance.filter(
            pl.col("series") == f"alpha_top10_{suffix}"
        ).row(0, named=True)
        paired = bootstrap.filter(
            (pl.col("strategy") == f"alpha_top10_{suffix}")
            & (pl.col("comparator") == f"alpha_top5_{suffix}")
        ).row(0, named=True)
        delta_rows.append(
            [
                html.escape(label),
                _pct(top10["cagr"] - top5["cagr"]),
                _number(top10["sharpe"] - top5["sharpe"]),
                _pct(top10["max_drawdown"] - top5["max_drawdown"]),
                (
                    f'{_number(paired["observed_sharpe_difference"])} '
                    f'[{_number(paired["sharpe_difference_ci_low"])}, '
                    f'{_number(paired["sharpe_difference_ci_high"])}]'
                ),
            ]
        )

    annual_wide = annual.pivot(
        on="series",
        index=["year", "months"],
        values="annual_return",
    ).sort("year")
    annual_rows = []
    for row in annual_wide.iter_rows(named=True):
        annual_rows.append(
            [
                f'{row["year"]}{"*" if row["months"] < 12 else ""}',
                _pct(row["alpha_top5_equal"], 1),
                _pct(row["alpha_top10_equal"], 1),
                _pct(row["Legacy"], 1),
                _pct(row["SPY total return"], 1),
            ]
        )

    top5_equal = performance.filter(
        pl.col("series") == "alpha_top5_equal"
    ).row(0, named=True)
    top10_equal = performance.filter(
        pl.col("series") == "alpha_top10_equal"
    ).row(0, named=True)
    gate_rows = [
        [
            "Sharpe top 10 > top 5",
            "PASS" if gates["sharpe_higher"] else "FAIL",
        ],
        [
            "Drawdown amélioré d'au moins 5 points",
            "PASS" if gates["drawdown_improves_5pp"] else "FAIL",
        ],
        [
            "Perte de CAGR limitée à 3 points",
            "PASS" if gates["cagr_loss_within_3pp"] else "FAIL",
        ],
        [
            "IC 95 % bootstrap du gain de Sharpe entièrement positif",
            "PASS"
            if gates["bootstrap_sharpe_ci_low_positive"]
            else "FAIL",
        ],
        [
            "Sharpe supérieur avec 50 pb × turnover",
            "PASS" if gates["sharpe_higher_at_50bps"] else "FAIL",
        ],
    ]
    bucket_rows = []
    for row in rank_buckets.to_dicts():
        bucket_rows.append(
            [
                (
                    "Rangs 1–5"
                    if row["bucket"] == "ranks_1_5"
                    else "Rangs 6–10"
                ),
                _number(row["mean_raw_alpha_score"]),
                _pct(row["stock_month_mean_return"]),
                _pct(row["stock_month_mean_excess_return"]),
                _pct(row["cagr"]),
                _number(row["sharpe"]),
                _pct(row["max_drawdown"]),
            ]
        )
    verdict = "PASS" if all(gates.values()) else "NO-GO"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank — Top 5 contre Top 10</title>
<style>
:root{{--ink:#18232d;--muted:#64717c;--line:#dce3e8;--paper:#f4f6f7;
--card:#fff;--blue:#195f85;--red:#a53a3a;--green:#267b57}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.55 Inter,system-ui,-apple-system,sans-serif}}
main{{max-width:1250px;margin:auto;padding:46px 24px 80px}}
.eyebrow{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.13em;
text-transform:uppercase}} h1{{font-size:clamp(38px,6vw,68px);line-height:1.02;
letter-spacing:-.045em;margin:8px 0 18px}} h2{{margin:48px 0 12px;font-size:28px}}
.lede{{max-width:900px;color:var(--muted);font-size:18px}}
.strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}}
.card{{padding:20px;background:var(--card);border:1px solid var(--line);border-radius:12px}}
.card b{{display:block;font-size:27px}} .card span{{color:var(--muted)}}
.callout{{padding:20px 22px;border-left:5px solid var(--blue);background:#eaf2f6;
border-radius:8px;margin:22px 0}} .bad{{border-color:var(--red);background:#fff0ee}}
.links{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 26px}}
.links a{{padding:8px 12px;color:var(--blue);background:var(--card);
border:1px solid var(--line);border-radius:999px;font-weight:750;text-decoration:none}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);
border-radius:12px}} table{{width:100%;min-width:900px;border-collapse:collapse;
font-variant-numeric:tabular-nums}} th,td{{padding:12px 13px;text-align:right;
border-bottom:1px solid var(--line);white-space:nowrap}} th{{font-size:11px;
color:var(--muted);text-transform:uppercase;letter-spacing:.05em;background:#f9fafb}}
th:first-child,td:first-child{{text-align:left}} .method{{display:grid;
grid-template-columns:repeat(3,1fr);gap:12px}} .method p{{color:var(--muted);margin:0}}
code{{font-size:12px}} footer{{margin-top:52px;color:var(--muted);font-size:12px}}
@media(max-width:760px){{main{{padding:30px 14px 60px}}.strip,.method{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">Allocation-only · mêmes prédictions OOS · ticker quarantine v6</p>
<h1>Top 5 contre Top 10</h1>
<p class="lede">La seule variable modifiée est le nombre d'actions retenues.
Les scores alpha six mois, les prédictions de risque, les 15 folds, les EMA,
les exclusions, le calendrier et les coûts sont strictement identiques.</p>
<nav class="links"><a href="alpha_shap_and_monthly_portfolios.html">
SHAP complet et portefeuilles mensuels</a></nav>
<div class="strip">
 <div class="card"><span>Top 5 égal · CAGR</span><b>{_pct(top5_equal["cagr"])}</b></div>
 <div class="card"><span>Top 10 égal · CAGR</span><b>{_pct(top10_equal["cagr"])}</b></div>
 <div class="card"><span>Top 5 égal · Sharpe</span><b>{_number(top5_equal["sharpe"])}</b></div>
 <div class="card"><span>Top 10 égal · Sharpe</span><b>{_number(top10_equal["sharpe"])}</b></div>
</div>
<div class="callout {'bad' if verdict == 'NO-GO' else ''}">
<strong>Verdict pré-enregistré : {verdict}.</strong>
Le top 10 diversifie davantage, mais il n'est promu que s'il passe simultanément
les cinq garde-fous ci-dessous.</div>

<h2>Ce qui change — et ce qui ne change pas</h2>
<div class="method">
 <article class="card"><h3>Top 5 égal</h3><p>Les cinq meilleurs scores alpha
 bruts, 20 % par titre.</p></article>
 <article class="card"><h3>Top 10 égal</h3><p>Les dix meilleurs scores alpha
 bruts, 10 % par titre. Aucun nouveau modèle.</p></article>
 <article class="card"><h3>Pondérations risque</h3><p>Même sélection top N,
 puis inverse volatilité ou downside prédit, avec plafond de 30 % par titre.
 La variante secteur conserve deux noms maximum par secteur et 40 % maximum
 par secteur.</p></article>
</div>
<div class="callout"><strong>Contrôle de reproductibilité.</strong>
La reconstruction du top 5 depuis les prédictions sauvegardées reproduit le
run v6 avec une erreur mensuelle maximale de {top5_replay_error:.2e}.</div>

<h2>Performances sur le même calendrier</h2>
{_table(["Méthode","CAGR","Sharpe Legacy","Vol. ann.","Max DD","Pire année","Turnover moyen"], performance_rows)}
<p class="lede">Août 2011–novembre 2025, 172 mois. Rendements ML nets de
10 pb × turnover. Sharpe = (CAGR − 2 %) / volatilité annualisée.</p>

<h2>Écart Top 10 moins Top 5</h2>
{_table(["Pondération","Δ CAGR","Δ Sharpe","Δ max DD","Δ Sharpe bootstrap [IC 95 %]"], delta_rows)}

<h2>Pourquoi le Top 10 se dégrade</h2>
{_table(["Bloc de rangs","Score alpha moyen","Rendement mensuel moyen",
"Excès mensuel moyen","CAGR brut","Sharpe","Max DD"], bucket_rows)}
<p class="lede">Cette décomposition est brute, sans coûts, et utilise les
holdings du Top 10 égal. Les rangs 6–10 ne sont pas seulement moins forts :
leur bloc produit un CAGR de {_pct(rank_buckets.filter(pl.col("bucket") == "ranks_6_10")["cagr"][0])}
et un drawdown de {_pct(rank_buckets.filter(pl.col("bucket") == "ranks_6_10")["max_drawdown"][0])}.
Le turnover du Top 10 égal est même légèrement inférieur à celui du Top 5 ;
la baisse ne vient donc pas des frais, mais de la dilution du signal.</p>

<h2>Garde-fous Top 10 égal</h2>
{_table(["Test","Résultat"], gate_rows)}

<h2>Rendements annuels équipondérés</h2>
{_table(["Année","Top 5 égal","Top 10 égal","Legacy","SPY"], annual_rows)}
<p class="lede">* Année partielle : 2011 commence en août et 2025 s'arrête en
novembre. Les CSV contiennent aussi toutes les pondérations vol, downside et
secteur pour les top 5 et top 10.</p>

<h2>Limites</h2>
<div class="callout bad"><strong>Ce test ne répare pas l'univers historique.</strong>
Il isole proprement l'effet de diversification top 10, mais les réserves sur
l'identité des constituants, le multiple testing et l'absence d'un nouveau
holdout restent inchangées. Une amélioration ici est un diagnostic, pas une
validation de production.</div>
<footer>Source : risk_predictions.parquet du run v6 · test allocation-only ·
bootstrap apparié par blocs de 12 mois · rapport généré par
scripts/experiments/run_topn_allocation_comparison.py.</footer>
</main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare top-5 and top-10 frozen-score allocations."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    output_dir = args.output_dir.resolve()
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    source_run = PROJECT_ROOT / specification["source"]["risk_run"]
    data_dir = PROJECT_ROOT / specification["source"]["input_snapshot"]
    legacy_path = (
        PROJECT_ROOT / specification["source"]["legacy_monthly_returns"]
    )
    risk_predictions_path = source_run / "risk_predictions.parquet"
    source_monthly_path = source_run / "allocation_monthly.csv"
    predictions = pl.read_parquet(risk_predictions_path)
    general = pl.read_parquet(data_dir / "US_General.parquet")
    monthly, holdings = _build_allocations(
        predictions=predictions,
        general=general,
        specification=specification,
    )
    monthly = monthly.join(
        _legacy_monthly(legacy_path),
        on="holding_month",
        how="inner",
    )
    top5_replay_error = _assert_top5_replay(
        monthly,
        pl.read_csv(source_monthly_path).with_columns(
            pl.col("holding_month").cast(pl.Utf8).str.to_date()
        ),
    )
    performance = _performance(monthly)
    annual = _annual_returns(monthly)
    rank_buckets = _rank_bucket_diagnostics(holdings)
    costs = _cost_sensitivity(
        monthly,
        [
            float(value)
            for value in specification["trading"]["cost_sensitivity_bps"]
        ],
    )
    bootstrap = _bootstrap(
        monthly,
        samples=int(
            specification["validation"]["paired_block_bootstrap_samples"]
        ),
        block_months=int(
            specification["validation"]["paired_block_months"]
        ),
        seed=int(specification["validation"]["seed"]),
    )
    gates = _promotion_gates(performance, costs, bootstrap)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.write_csv(output_dir / "allocation_monthly.csv")
    holdings.write_parquet(output_dir / "allocation_holdings.parquet")
    common_monthly = monthly.with_columns(
        (pl.col("net_return") - pl.col("benchmark_return")).alias("active_return"),
        (
            (1.0 + pl.col("net_return")) / (1.0 + pl.col("benchmark_return")) - 1.0
        ).alias("relative_return"),
    )
    reference_source = common_monthly.filter(pl.col("strategy") == "alpha_top5_equal")
    common_monthly = pl.concat(
        [
            common_monthly,
            reference_monthly_series(
                reference_source,
                strategy="Legacy",
                return_column="legacy_return",
            ),
            reference_monthly_series(
                reference_source,
                strategy="SPY total return",
                return_column="benchmark_return",
            ),
        ],
        how="diagonal_relaxed",
    )
    write_common_portfolio_artifacts(
        output_dir=output_dir,
        holdings=holdings.select(
            "strategy",
            "decision_month",
            "holding_month",
            "ticker",
            "target_weight",
            "realized_return",
            "benchmark_return",
            "sector",
            "selection_rank",
            "score",
        ),
        monthly_returns=common_monthly.select(
            "strategy",
            "decision_month",
            "holding_month",
            "gross_return",
            "turnover",
            "transaction_cost",
            "net_return",
            "benchmark_return",
            "active_return",
            "relative_return",
            "n_positions",
            "maximum_position_weight",
            "maximum_sector_weight",
            "sector_count",
        ),
    )
    performance.write_csv(output_dir / "performance_legacy_convention.csv")
    annual.write_csv(output_dir / "annual_returns.csv")
    annual.pivot(
        on="series",
        index=["year", "months"],
        values="annual_return",
    ).sort("year").write_csv(output_dir / "annual_returns_wide.csv")
    costs.write_csv(output_dir / "cost_sensitivity.csv")
    bootstrap.write_csv(output_dir / "paired_block_bootstrap.csv")
    rank_buckets.write_csv(output_dir / "rank_bucket_diagnostics.csv")
    pl.DataFrame(
        [
            {"gate": key, "pass": value}
            for key, value in gates.items()
        ]
    ).write_csv(output_dir / "promotion_gates.csv")
    _render_html(
        performance=performance,
        annual=annual,
        bootstrap=bootstrap,
        gates=gates,
        rank_buckets=rank_buckets,
        top5_replay_error=top5_replay_error,
        output_path=output_dir / "html" / "top5_vs_top10.html",
    )
    manifest = {
        "research_id": specification["research_id"],
        "created_from_repository_head": _git_head(),
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "runner_sha256": _sha256(Path(__file__)),
        "source_risk_predictions": str(risk_predictions_path.resolve()),
        "source_risk_predictions_sha256": _sha256(risk_predictions_path),
        "source_risk_manifest_sha256": _sha256(source_run / "manifest.json"),
        "source_allocation_monthly_sha256": _sha256(source_monthly_path),
        "top5_replay_maximum_monthly_error": top5_replay_error,
        "only_changed_parameter": "top_n",
        "top_n_values": specification["experiment"]["top_n_values"],
        "test_start": str(monthly["holding_month"].min()),
        "test_end": str(monthly["holding_month"].max()),
        "test_months": monthly["holding_month"].n_unique(),
        "all_top10_equal_promotion_gates_pass": all(gates.values()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print((output_dir / "html" / "top5_vs_top10.html").resolve())


if __name__ == "__main__":
    main()
