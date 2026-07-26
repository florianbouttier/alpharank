#!/usr/bin/env python3
"""Render complete alpha SHAP diagnostics and monthly portfolio holdings."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import shap  # noqa: E402


DEFAULT_ALPHA_DIR = Path(
    "outputs/multihorizon_boosting/"
    "legacy_ema_long_history_ticker_quarantine_v6_20260726/"
    "classification_h06"
)
DEFAULT_ALLOCATION_DIR = Path(
    "outputs/multihorizon_boosting/"
    "legacy_ema_top5_vs_top10_quarantine_v7_20260726"
)
DEFAULT_LEGACY_HOLDINGS = Path(
    "outputs/2026-07-13/runs/20260713_201639/"
    "legacy_detailed_returns_polars.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(index: int, feature: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", feature).strip("_")
    return f"{index:03d}_{clean}.svg"


def _finite_feature_arrays(
    samples: pl.DataFrame,
    feature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = samples.select(
        "fold",
        pl.col(f"value__{feature}").cast(pl.Float64).alias("value"),
        pl.col(f"shap__{feature}").cast(pl.Float64).alias("shap"),
    ).drop_nulls(["value", "shap"])
    if frame.is_empty():
        return np.array([]), np.array([]), np.array([])
    values = frame["value"].to_numpy()
    shap_values = frame["shap"].to_numpy()
    folds = frame["fold"].to_numpy()
    finite = np.isfinite(values) & np.isfinite(shap_values)
    return values[finite], shap_values[finite], folds[finite]


def _plot_beeswarm(
    samples: pl.DataFrame,
    features: list[str],
    output_path: Path,
) -> None:
    values = np.column_stack(
        [
            samples[f"value__{feature}"].cast(pl.Float64).to_numpy()
            for feature in features
        ]
    )
    shap_values = np.column_stack(
        [
            samples[f"shap__{feature}"].cast(pl.Float64).to_numpy()
            for feature in features
        ]
    )
    plt.figure()
    shap.summary_plot(
        shap_values,
        values,
        feature_names=features,
        max_display=len(features),
        show=False,
        plot_size=(15, max(22, 0.285 * len(features) + 4)),
        alpha=0.58,
        sort=False,
    )
    figure = plt.gcf()
    figure.suptitle(
        "Classifieur alpha 6 mois — beeswarm SHAP complet",
        fontsize=18,
        y=1.002,
    )
    figure.savefig(output_path, dpi=155, bbox_inches="tight")
    plt.close(figure)


def _binned_medians(
    values: np.ndarray,
    shap_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 12:
        return np.array([]), np.array([])
    quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 13)))
    if quantiles.size < 3:
        return np.array([]), np.array([])
    centers: list[float] = []
    medians: list[float] = []
    for lower, upper in zip(quantiles[:-1], quantiles[1:], strict=True):
        mask = (values >= lower) & (
            values <= upper if upper == quantiles[-1] else values < upper
        )
        if mask.sum() < 3:
            continue
        centers.append(float(np.median(values[mask])))
        medians.append(float(np.median(shap_values[mask])))
    return np.asarray(centers), np.asarray(medians)


def _plot_feature(
    *,
    feature: str,
    values: np.ndarray,
    shap_values: np.ndarray,
    folds: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    scatter = axis.scatter(
        values,
        shap_values,
        c=folds,
        cmap="viridis",
        s=20,
        alpha=0.60,
        edgecolors="none",
    )
    centers, medians = _binned_medians(values, shap_values)
    if centers.size:
        order = np.argsort(centers)
        axis.plot(
            centers[order],
            medians[order],
            color="#b33b3b",
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label="Médiane par quantile",
        )
        axis.legend(frameon=False, loc="best")
    axis.axhline(0.0, color="#55616c", linewidth=1.0, linestyle="--")
    axis.set_title(feature, fontsize=11, loc="left")
    axis.set_xlabel("Valeur de la variable après prétraitement du fold")
    axis.set_ylabel("Contribution SHAP au score brut")
    axis.grid(alpha=0.18)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Fold OOS")
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _feature_catalog(
    *,
    samples: pl.DataFrame,
    direction: pl.DataFrame,
    assets_dir: Path,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(direction.to_dicts(), start=1):
        feature = str(metadata["feature"])
        values, shap_values, folds = _finite_feature_arrays(samples, feature)
        if not values.size:
            continue
        image_name = _slug(index, feature)
        _plot_feature(
            feature=feature,
            values=values,
            shap_values=shap_values,
            folds=folds,
            output_path=assets_dir / image_name,
        )
        rows.append(
            {
                "importance_rank": index,
                "feature": feature,
                "mean_abs_shap": float(metadata["mean_abs_shap"]),
                "value_shap_correlation": (
                    float(metadata["value_shap_correlation"])
                    if metadata["value_shap_correlation"] is not None
                    else None
                ),
                "direction": metadata["direction"],
                "sample_count": int(values.size),
                "active_folds": int(np.unique(folds).size),
                "value_q05": float(np.quantile(values, 0.05)),
                "value_median": float(np.median(values)),
                "value_q95": float(np.quantile(values, 0.95)),
                "shap_q05": float(np.quantile(shap_values, 0.05)),
                "shap_median": float(np.median(shap_values)),
                "shap_q95": float(np.quantile(shap_values, 0.95)),
                "plot_path": f"shap_alpha_features/{image_name}",
            }
        )
    return pl.DataFrame(rows).sort("importance_rank")


def _monthly_portfolios(
    *,
    allocation_holdings: pl.DataFrame,
    legacy_holdings: pl.DataFrame,
    allocation_monthly: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    algorithms = (
        allocation_holdings.filter(
            pl.col("strategy").is_in(
                ["alpha_top5_equal", "alpha_top10_equal"]
            )
        )
        .select(
            "holding_month",
            pl.when(pl.col("strategy") == "alpha_top5_equal")
            .then(pl.lit("Alpha Top 5 égal"))
            .otherwise(pl.lit("Alpha Top 10 égal"))
            .alias("portfolio"),
            "ticker",
            pl.col("portfolio_weight").alias("weight"),
            pl.col("selection_rank").alias("rank"),
            pl.col("score").alias("raw_alpha_score"),
            "calibrated_probability",
            pl.col("future_return_1m").alias("realized_return_1m"),
            "sector",
        )
    )
    minimum_month = algorithms["holding_month"].min()
    maximum_month = algorithms["holding_month"].max()
    legacy = (
        legacy_holdings.filter(
            (pl.col("portfolio_model") == "Combined_Frequency")
            & pl.col("weight_normalized").is_not_null()
        )
        .with_columns(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.lit("Legacy publié").alias("portfolio"),
            pl.col("weight_normalized").alias("weight"),
            pl.col("dr").alias("realized_return_1m"),
            pl.col("Sector").alias("sector"),
        )
        .filter(
            pl.col("holding_month").is_between(
                minimum_month,
                maximum_month,
            )
        )
        .sort(["holding_month", "weight"], descending=[False, True])
        .with_columns(
            pl.col("weight")
            .rank(method="ordinal", descending=True)
            .over("holding_month")
            .cast(pl.Int64)
            .alias("rank"),
            pl.lit(None, dtype=pl.Float64).alias("raw_alpha_score"),
            pl.lit(None, dtype=pl.Float64).alias(
                "calibrated_probability"
            ),
        )
        .select(algorithms.columns)
    )
    portfolios = pl.concat([legacy, algorithms], how="vertical").sort(
        ["holding_month", "portfolio", "rank"]
    )
    top5 = allocation_monthly.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).select(
        "holding_month",
        pl.col("net_return").alias("alpha_top5_return"),
        "legacy_return",
        pl.col("benchmark_return").alias("spy_return"),
    )
    top10 = allocation_monthly.filter(
        pl.col("strategy") == "alpha_top10_equal"
    ).select(
        "holding_month",
        pl.col("net_return").alias("alpha_top10_return"),
    )
    returns = top5.join(top10, on="holding_month", how="inner").sort(
        "holding_month"
    )
    return portfolios, returns


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):+.{digits}f}%".replace(".", ",")


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _portfolio_table(
    frame: pl.DataFrame,
    *,
    algorithm: bool,
) -> str:
    headers = ["Rang", "Ticker", "Poids", "Secteur", "Rendement réalisé"]
    if algorithm:
        headers += ["Score brut", "Probabilité calibrée"]
    rows = []
    for row in frame.sort("rank").to_dicts():
        cells = [
            str(row["rank"]),
            html.escape(str(row["ticker"])),
            _pct(row["weight"]),
            html.escape(str(row["sector"] or "Unknown")),
            _pct(row["realized_return_1m"]),
        ]
        if algorithm:
            cells += [
                _number(row["raw_alpha_score"]),
                _pct(row["calibrated_probability"]),
            ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _monthly_sections(
    portfolios: pl.DataFrame,
    returns: pl.DataFrame,
) -> str:
    parts: list[str] = []
    return_map = {
        row["holding_month"]: row for row in returns.to_dicts()
    }
    months = sorted(
        portfolios["holding_month"].unique().to_list(),
        reverse=True,
    )
    for index, month in enumerate(months):
        month_frame = portfolios.filter(pl.col("holding_month") == month)
        metrics = return_map[month]
        cards = []
        for portfolio in (
            "Legacy publié",
            "Alpha Top 5 égal",
            "Alpha Top 10 égal",
        ):
            frame = month_frame.filter(pl.col("portfolio") == portfolio)
            cards.append(
                '<article class="portfolio-card">'
                f"<h4>{html.escape(portfolio)} · {frame.height} titres</h4>"
                '<div class="inner-table">'
                + _portfolio_table(
                    frame,
                    algorithm=portfolio != "Legacy publié",
                )
                + "</div></article>"
            )
        parts.append(
            f"""<details class="month" data-month="{month:%Y-%m}" {
                "open" if index == 0 else ""
            }>
              <summary><strong>{month:%Y-%m}</strong>
                <span>Legacy {_pct(metrics["legacy_return"])} ·
                Top 5 {_pct(metrics["alpha_top5_return"])} ·
                Top 10 {_pct(metrics["alpha_top10_return"])} ·
                SPY {_pct(metrics["spy_return"])}</span>
              </summary>
              <div class="month-grid">{''.join(cards)}</div>
            </details>"""
        )
    return "".join(parts)


def _render_html(
    *,
    catalog: pl.DataFrame,
    portfolios: pl.DataFrame,
    returns: pl.DataFrame,
    beeswarm_path: str,
    output_path: Path,
) -> None:
    top_rows = []
    for row in catalog.head(20).to_dicts():
        top_rows.append(
            "<tr>"
            f'<td>{row["importance_rank"]}</td>'
            f'<td><code>{html.escape(row["feature"])}</code></td>'
            f'<td>{_number(row["mean_abs_shap"], 5)}</td>'
            f'<td>{_number(row["value_shap_correlation"], 3)}</td>'
            f'<td>{html.escape(str(row["direction"]))}</td>'
            f'<td>{row["sample_count"]}</td>'
            f'<td>{row["active_folds"]}/15</td>'
            "</tr>"
        )
    feature_cards = []
    for index, row in enumerate(catalog.to_dicts()):
        feature_cards.append(
            f"""<details class="feature-card" data-feature="{
                html.escape(str(row["feature"]).lower(), quote=True)
            }" {"open" if index < 3 else ""}>
              <summary><span class="rank">#{row["importance_rank"]}</span>
                <code>{html.escape(row["feature"])}</code>
                <span>|SHAP| moyen {_number(row["mean_abs_shap"], 5)} ·
                n={row["sample_count"]} · {row["active_folds"]}/15 folds</span>
              </summary>
              <div class="feature-body">
                <img loading="lazy" src="{html.escape(row["plot_path"], quote=True)}"
                  alt="SHAP individuel {html.escape(row["feature"], quote=True)}">
                <div class="feature-meta">
                  <p><strong>Direction globale :</strong>
                    {html.escape(str(row["direction"]))}</p>
                  <p><strong>Corrélation valeur–SHAP :</strong>
                    {_number(row["value_shap_correlation"], 3)}</p>
                  <p><strong>Valeur prétraitée P5 / médiane / P95 :</strong>
                    {_number(row["value_q05"], 3)} /
                    {_number(row["value_median"], 3)} /
                    {_number(row["value_q95"], 3)}</p>
                  <p><strong>SHAP P5 / médiane / P95 :</strong>
                    {_number(row["shap_q05"], 4)} /
                    {_number(row["shap_median"], 4)} /
                    {_number(row["shap_q95"], 4)}</p>
                </div>
              </div>
            </details>"""
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank — SHAP complet et portefeuilles mensuels</title>
<style>
:root{{--ink:#18232d;--muted:#64717c;--line:#dce3e8;--paper:#f4f6f7;
--card:#fff;--blue:#195f85;--red:#a53a3a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.55 Inter,system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:46px 24px 80px}}
.eyebrow{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.13em;
text-transform:uppercase}} h1{{font-size:clamp(38px,6vw,68px);line-height:1.02;
letter-spacing:-.045em;margin:8px 0 18px}} h2{{margin:52px 0 12px;font-size:30px}}
h3{{margin:32px 0 10px}} h4{{margin:0 0 10px}} .lede{{max-width:950px;
color:var(--muted);font-size:18px}} .callout{{padding:20px 22px;margin:24px 0;
border-left:5px solid var(--blue);background:#eaf2f6;border-radius:8px}}
.warning{{border-color:var(--red);background:#fff0ee}} .beeswarm{{max-height:78vh;
overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}}
.beeswarm img{{display:block;width:100%;min-width:900px;height:auto}}
.table-wrap,.inner-table{{overflow:auto;background:var(--card);border:1px solid var(--line);
border-radius:12px}} table{{width:100%;border-collapse:collapse;min-width:760px;
font-variant-numeric:tabular-nums}} th,td{{padding:10px 11px;border-bottom:1px solid var(--line);
text-align:right;white-space:nowrap}} th{{font-size:11px;text-transform:uppercase;
color:var(--muted);background:#f9fafb}} th:nth-child(2),td:nth-child(2){{text-align:left}}
.search{{width:100%;max-width:580px;padding:12px 14px;border:1px solid var(--line);
border-radius:10px;background:var(--card);font:inherit;margin:12px 0 18px}}
details{{background:var(--card);border:1px solid var(--line);border-radius:11px;
margin:9px 0}} summary{{cursor:pointer;padding:14px 16px;display:flex;gap:12px;
align-items:center;justify-content:space-between}} summary span{{color:var(--muted)}}
.rank{{color:var(--blue);font-weight:800}} .feature-body{{display:grid;
grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:18px;padding:0 16px 18px}}
.feature-body img{{width:100%;height:auto;border-radius:8px}} .feature-meta{{color:var(--muted)}}
.month-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;
padding:0 12px 14px}} .portfolio-card{{min-width:0}} .portfolio-card table{{min-width:650px}}
.portfolio-card h4{{padding-left:2px}} code{{font-size:12px}} footer{{margin-top:52px;
color:var(--muted);font-size:12px}} .hidden{{display:none}}
@media(max-width:900px){{main{{padding:30px 14px 60px}}.feature-body,.month-grid{{
grid-template-columns:1fr}}summary{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><main>
<p class="eyebrow">Alpha v6 · classification six mois · OOS seulement</p>
<h1>SHAP complet et portefeuilles mensuels</h1>
<p class="lede">Le dernier test Top 10 n'a entraîné aucun nouveau modèle.
Les explications ci-dessous portent donc sur le classifieur alpha v6 exact qui
produit les scores des Top 5 et Top 10.</p>
<div class="callout"><strong>Périmètre SHAP.</strong> 1 200 observations
hors échantillon, 80 par fold, et {catalog.height} variables distinctes sur
l'union des 15 folds. Une variable absente d'un fold reste manquante : aucune
valeur SHAP n'est inventée.</div>
<div class="callout warning"><strong>Unité des SHAP.</strong> Les contributions
sont dans l'espace du score brut XGBoost — la marge/log-odds — et non en points
de probabilité calibrée. Leur somme explique le score brut avant calibration
isotone.</div>

<h2>Beeswarm global — toutes les variables</h2>
<p class="lede">Chaque point est une observation OOS. Rouge = valeur élevée,
bleu = valeur faible. L'axe horizontal indique l'effet sur le score alpha brut.
Le graphique contient les {catalog.height} variables, sans limitation top N.</p>
<div class="beeswarm"><img src="{html.escape(beeswarm_path, quote=True)}"
alt="Beeswarm SHAP complet"></div>

<h2>Valeurs SHAP principales</h2>
<div class="table-wrap"><table><thead><tr><th>Rang</th><th>Variable</th>
<th>|SHAP| moyen</th><th>Corrélation</th><th>Direction</th>
<th>Observations</th><th>Folds actifs</th></tr></thead>
<tbody>{''.join(top_rows)}</tbody></table></div>

<h2>Graphique individuel pour chaque variable</h2>
<p class="lede">Le nuage relie la valeur prétraitée dans son fold à sa
contribution SHAP. La ligne rouge donne la médiane par quantile ; elle aide à
voir seuils, saturation et non-linéarités.</p>
<input id="feature-search" class="search" type="search"
placeholder="Filtrer les {catalog.height} variables…">
<div id="feature-list">{''.join(feature_cards)}</div>

<h2>Portefeuilles par mois</h2>
<p class="lede">Les {returns.height} mois communs sont affichés, du plus récent
au plus ancien. « Mon algo » est montré sous ses deux formes : le Top 5 champion
et le Top 10 testé. Les rendements Top 5/Top 10 sont nets de 10 pb × turnover.</p>
<div class="callout warning"><strong>Legacy publié.</strong> Il s'agit des
holdings Legacy historiques publiés, non du rerun intégral corrigé par la
quarantaine. Cette distinction est conservée dans chaque mois.</div>
<input id="month-search" class="search" type="search"
placeholder="Filtrer un mois, par exemple 2023-06…">
<div id="month-list">{_monthly_sections(portfolios, returns)}</div>

<footer>SHAP source : classification_h06/shap_samples.parquet · holdings algo :
allocation v7 · holdings Legacy : package validé 20260713_201639 · rapport
reproductible par scripts/experiments/render_alpha_shap_portfolio_report.py.</footer>
<script>
const bindFilter=(inputId,selector,attribute)=>{{
  const input=document.getElementById(inputId);
  input.addEventListener('input',()=>{{
    const query=input.value.trim().toLowerCase();
    document.querySelectorAll(selector).forEach(item=>{{
      item.classList.toggle('hidden',!item.dataset[attribute].includes(query));
    }});
  }});
}};
bindFilter('feature-search','.feature-card','feature');
bindFilter('month-search','.month','month');
</script>
</main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render complete alpha SHAP and monthly portfolios."
    )
    parser.add_argument("--alpha-dir", type=Path, default=DEFAULT_ALPHA_DIR)
    parser.add_argument(
        "--allocation-dir",
        type=Path,
        default=DEFAULT_ALLOCATION_DIR,
    )
    parser.add_argument(
        "--legacy-holdings",
        type=Path,
        default=DEFAULT_LEGACY_HOLDINGS,
    )
    args = parser.parse_args()

    alpha_dir = args.alpha_dir.resolve()
    allocation_dir = args.allocation_dir.resolve()
    html_dir = allocation_dir / "html"
    assets_dir = html_dir / "shap_alpha_features"
    assets_dir.mkdir(parents=True, exist_ok=True)
    samples_path = alpha_dir / "shap_samples.parquet"
    direction_path = alpha_dir / "shap_direction.csv"
    samples = pl.read_parquet(samples_path)
    direction = pl.read_csv(direction_path).sort(
        "mean_abs_shap",
        descending=True,
    )
    features = direction["feature"].to_list()
    beeswarm_path = html_dir / "alpha_shap_beeswarm_all_features.png"
    _plot_beeswarm(samples, features, beeswarm_path)
    catalog = _feature_catalog(
        samples=samples,
        direction=direction,
        assets_dir=assets_dir,
    )
    catalog.write_csv(allocation_dir / "alpha_shap_feature_catalog.csv")

    allocation_holdings_path = allocation_dir / "allocation_holdings.parquet"
    allocation_monthly_path = allocation_dir / "allocation_monthly.csv"
    portfolios, returns = _monthly_portfolios(
        allocation_holdings=pl.read_parquet(allocation_holdings_path),
        legacy_holdings=pl.read_parquet(args.legacy_holdings),
        allocation_monthly=pl.read_csv(allocation_monthly_path).with_columns(
            pl.col("holding_month").cast(pl.Utf8).str.to_date()
        ),
    )
    portfolios.write_parquet(allocation_dir / "monthly_portfolios.parquet")
    returns.write_csv(allocation_dir / "monthly_portfolio_returns.csv")
    report_path = html_dir / "alpha_shap_and_monthly_portfolios.html"
    _render_html(
        catalog=catalog,
        portfolios=portfolios,
        returns=returns,
        beeswarm_path=beeswarm_path.name,
        output_path=report_path,
    )
    manifest = {
        "report": str(report_path),
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": _sha256(Path(__file__).resolve()),
        "alpha_shap_samples": str(samples_path),
        "alpha_shap_samples_sha256": _sha256(samples_path),
        "alpha_shap_direction_sha256": _sha256(direction_path),
        "allocation_holdings_sha256": _sha256(allocation_holdings_path),
        "allocation_monthly_sha256": _sha256(allocation_monthly_path),
        "legacy_holdings_sha256": _sha256(args.legacy_holdings.resolve()),
        "shap_observations": samples.height,
        "shap_features": catalog.height,
        "individual_feature_plots": catalog.height,
        "holding_months": returns.height,
        "portfolios": [
            "Legacy publié",
            "Alpha Top 5 égal",
            "Alpha Top 10 égal",
        ],
        "shap_unit": "raw XGBoost margin/log-odds before isotonic calibration",
    }
    (allocation_dir / "alpha_shap_portfolio_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
