#!/usr/bin/env python3
"""Render annual returns for every risk-allocation method on one calendar."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import polars as pl


DEFAULT_RUN_DIR = Path(
    "outputs/multihorizon_boosting/"
    "legacy_ema_risk_overlay_ticker_quarantine_v6_20260726"
)

SERIES = (
    ("Legacy published", "Legacy", "benchmark"),
    ("SPY total return", "SPY", "benchmark"),
    ("alpha_top5_equal", "Alpha égal", "alpha"),
    ("alpha_top5_inverse_vol_h1", "Vol 1m", "risk"),
    ("alpha_top5_inverse_vol_h3", "Vol 3m", "risk"),
    ("alpha_top5_inverse_vol_h6", "Vol 6m", "risk"),
    ("alpha_top5_inverse_downside_h1", "Down 1m", "risk"),
    ("alpha_top5_inverse_downside_h3", "Down 3m", "risk"),
    ("alpha_top5_inverse_downside_h6", "Down 6m", "risk"),
    ("alpha_top5_inverse_vol_h3_sector2", "Vol 3m + secteur", "risk"),
)


def _compound(values: pl.Series) -> float:
    return float((1.0 + values.cast(pl.Float64)).product() - 1.0)


def annual_returns(monthly: pl.DataFrame) -> pl.DataFrame:
    """Compound monthly returns by calendar year on the shared test window."""

    monthly = monthly.with_columns(
        pl.col("holding_month").cast(pl.Utf8).str.to_date()
    )
    strategies = monthly["strategy"].unique(maintain_order=True).to_list()
    expected = [series for series, _, role in SERIES if role != "benchmark"]
    missing = sorted(set(expected) - set(strategies))
    if missing:
        raise ValueError(f"Missing allocation series: {missing}")

    rows: list[dict[str, object]] = []
    for strategy in strategies:
        frame = monthly.filter(pl.col("strategy") == strategy).sort(
            "holding_month"
        )
        for key, group in frame.group_by(
            pl.col("holding_month").dt.year().alias("year"),
            maintain_order=True,
        ):
            rows.append(
                {
                    "year": int(key[0]),
                    "months": group.height,
                    "series": strategy,
                    "annual_return": _compound(group["net_return"]),
                }
            )

    reference = monthly.filter(
        pl.col("strategy") == strategies[0]
    ).sort("holding_month")
    for column, label in (
        ("legacy_return", "Legacy published"),
        ("benchmark_return", "SPY total return"),
    ):
        for key, group in reference.group_by(
            pl.col("holding_month").dt.year().alias("year"),
            maintain_order=True,
        ):
            rows.append(
                {
                    "year": int(key[0]),
                    "months": group.height,
                    "series": label,
                    "annual_return": _compound(group[column]),
                }
            )

    return pl.DataFrame(rows).sort("year", "series")


def _pct(value: float) -> str:
    return f"{100.0 * value:+.1f}%".replace(".", ",")


def _cell(value: float, *, best: bool = False) -> str:
    intensity = min(abs(value) / 1.25, 1.0)
    color = "36, 132, 92" if value >= 0 else "180, 56, 56"
    best_class = " best" if best else ""
    return (
        f'<td class="return{best_class}" '
        f'style="background:rgba({color},{0.07 + 0.22 * intensity:.3f})">'
        f"{html.escape(_pct(value))}</td>"
    )


def _summary_rows(wide: pl.DataFrame) -> str:
    full = wide.filter(pl.col("months") == 12)
    rows = []
    for series, label, role in SERIES:
        if role == "benchmark":
            continue
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(label)}</strong></td>"
            f"<td>{int((full[series] > full['Legacy published']).sum())}/"
            f"{full.height}</td>"
            f"<td>{int((full[series] > full['SPY total return']).sum())}/"
            f"{full.height}</td>"
            f"<td>{int((full[series] > 0).sum())}/{full.height}</td>"
            f"<td>{_pct(float(full[series].median()))}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_report(annual: pl.DataFrame, output_path: Path) -> None:
    """Render the responsive annual-return paper."""

    wide = annual.pivot(
        on="series",
        index=["year", "months"],
        values="annual_return",
    ).sort("year")
    headers = "".join(
        f"<th>{html.escape(label)}</th>" for _, label, _ in SERIES
    )
    body_rows = []
    strategy_columns = [
        series for series, _, role in SERIES if role != "benchmark"
    ]
    for row in wide.iter_rows(named=True):
        best_value = max(float(row[column]) for column in strategy_columns)
        cells = []
        for series, _, role in SERIES:
            value = float(row[series])
            cells.append(
                _cell(
                    value,
                    best=role != "benchmark" and value == best_value,
                )
            )
        partial = " · partiel" if int(row["months"]) < 12 else ""
        body_rows.append(
            "<tr>"
            f'<th class="year">{int(row["year"])}</th>'
            f'<td class="months">{int(row["months"])} mois{partial}</td>'
            + "".join(cells)
            + "</tr>"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRank — rendements annuels de toutes les méthodes</title>
  <style>
    :root {{ --ink:#18232d; --muted:#65717c; --line:#dce3e8;
      --paper:#f4f6f7; --card:#fff; --blue:#195f85; --green:#24845c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink);
      font:15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    main {{ max-width:1500px; margin:auto; padding:46px 24px 80px; }}
    .eyebrow {{ color:var(--blue); font-size:12px; font-weight:800;
      letter-spacing:.13em; text-transform:uppercase; }}
    h1 {{ max-width:980px; margin:8px 0 18px; font-size:clamp(36px,6vw,68px);
      line-height:1.02; letter-spacing:-.045em; }}
    h2 {{ margin:48px 0 12px; font-size:28px; letter-spacing:-.025em; }}
    h3 {{ margin:0 0 8px; font-size:19px; letter-spacing:-.015em; }}
    .lede {{ max-width:860px; color:var(--muted); font-size:18px; }}
    .note {{ margin:26px 0; padding:20px 22px; background:#eaf2f6;
      border-left:4px solid var(--blue); border-radius:8px; }}
    .method-grid {{ display:grid; grid-template-columns:repeat(3,1fr);
      gap:14px; margin:18px 0; }}
    .method-card {{ padding:20px; background:var(--card);
      border:1px solid var(--line); border-radius:12px; }}
    .method-card p {{ margin:0; color:var(--muted); }}
    .method-card strong {{ color:var(--ink); }}
    .links {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 30px; }}
    .links a {{ display:inline-block; padding:8px 12px; color:var(--blue);
      background:var(--card); border:1px solid var(--line); border-radius:999px;
      font-weight:750; text-decoration:none; }}
    .flow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px;
      margin:18px 0 28px; }}
    .flow div {{ position:relative; padding:15px; background:#edf3f6;
      border-radius:10px; font-size:13px; }}
    .flow b {{ display:block; margin-bottom:4px; color:var(--blue); }}
    .method-table {{ max-height:none; margin-top:18px; }}
    .method-table table {{ min-width:1080px; }}
    .method-table th,.method-table td {{ text-align:left; white-space:normal; }}
    .method-table td:first-child {{ min-width:150px; font-weight:800; }}
    .table-wrap {{ overflow:auto; max-height:78vh; background:var(--card);
      border:1px solid var(--line); border-radius:12px; }}
    table {{ width:100%; min-width:1310px; border-collapse:separate;
      border-spacing:0; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:12px 13px; border:0; border-bottom:1px solid var(--line);
      text-align:right; white-space:nowrap; }}
    thead th {{ position:sticky; top:0; z-index:3; background:#f8fafb;
      color:var(--muted); font-size:11px; letter-spacing:.05em;
      text-transform:uppercase; }}
    .year {{ position:sticky; left:0; z-index:2; background:var(--card);
      text-align:left; }}
    thead .year {{ z-index:4; background:#f8fafb; }}
    .months {{ position:sticky; left:72px; z-index:2; background:var(--card);
      color:var(--muted); text-align:left; }}
    thead .months {{ z-index:4; background:#f8fafb; }}
    .return {{ font-weight:700; }}
    .best {{ box-shadow:inset 0 -3px 0 var(--green); }}
    .legend {{ display:flex; flex-wrap:wrap; gap:18px; margin:14px 0;
      color:var(--muted); font-size:13px; }}
    .swatch {{ display:inline-block; width:12px; height:12px; margin-right:6px;
      border-radius:3px; vertical-align:-1px; }}
    .positive {{ background:rgba(36,132,92,.22); }}
    .negative {{ background:rgba(180,56,56,.22); }}
    .summary {{ max-height:none; }}
    .summary table {{ min-width:720px; }}
    .summary th:first-child,.summary td:first-child {{ text-align:left; }}
    footer {{ margin-top:54px; color:var(--muted); font-size:12px; }}
    @media(max-width:760px) {{
      main {{ padding:30px 14px 60px; }}
      h2 {{ margin-top:40px; }}
      .method-grid,.flow {{ grid-template-columns:1fr; }}
      .year {{ min-width:62px; }}
      .months {{ left:62px; }}
    }}
  </style>
</head>
<body><main>
  <p class="eyebrow">AlphaRank · ticker quarantine v6 · calendrier commun</p>
  <h1>Rendements annuels de toutes les méthodes</h1>
  <p class="lede">Rendements mensuels composés sur août 2011–novembre 2025.
  Les allocations ML sont nettes de 10 pb multipliés par le turnover. Legacy
  et SPY sont les séries publiées sur le même calendrier.</p>
  <nav class="links">
    <a href="risk_results_paper.html">Résultats modèles, risque et SHAP</a>
    <a href="methodology_paper.html">Méthodologie complète</a>
    <a href="../../legacy_ema_top5_vs_top10_quarantine_v7_20260726/html/alpha_shap_and_monthly_portfolios.html">SHAP alpha complet et portefeuilles mensuels</a>
  </nav>
  <div class="note"><strong>Lecture prudente.</strong> 2011 ne contient que
  cinq mois et 2025 onze mois. Les années supérieures à 100 % sont correctement
  recomposées depuis le run, mais restent diagnostiques tant que l'univers
  historique des constituants n'est pas entièrement réparé.</div>

  <h2>Ce que signifie « Alpha égal »</h2>
  <div class="method-grid">
    <article class="method-card">
      <h3>Signal alpha</h3>
      <p>Un <strong>classifieur XGBoost</strong> estime si une action fera partie
      des 10 % de meilleures surperformances relatives au S&amp;P 500 sur les
      six mois suivants. Le portefeuille est classé avec le score brut du
      modèle.</p>
    </article>
    <article class="method-card">
      <h3>Sélection</h3>
      <p>À chaque fin de mois, les <strong>cinq actions au score alpha le plus
      élevé</strong> sont retenues. Le score de risque ne change jamais cet
      ordre, sauf la variante avec contrainte sectorielle.</p>
    </article>
    <article class="method-card">
      <h3>Pondération égale</h3>
      <p>« Égal » signifie simplement <strong>20 % par action</strong> pour les
      cinq titres sélectionnés. C'est la référence permettant de mesurer si les
      pondérations de risque apportent réellement quelque chose.</p>
    </article>
  </div>

  <h2>Pipeline exact du modèle</h2>
  <div class="flow">
    <div><b>1 · Univers causal</b>Quarantaine complète des 10 tickers audités,
      puis filtres de liquidité et cohérence OHLC connus au mois de décision.</div>
    <div><b>2 · Variables</b>Uniquement les EMA gagnantes de Legacy disponibles
      au cutoff d'entraînement : 40 paires relatives, 309 variables candidates.
      Aucun fondamental dans ce run.</div>
    <div><b>3 · Cible</b>Top 10 % cross-sectionnel du rendement composé à six
      mois de l'action relativement au rendement composé du S&amp;P 500.</div>
    <div><b>4 · Validation</b>Walk-forward expanding : 62 mois minimum de train,
      6 mois de validation, purge de 6 mois, 12 mois de test, pas de mélange
      temporel.</div>
    <div><b>5 · Trading</b>Décision en t, rendement détenu en t+1, rééquilibrage
      mensuel et coût de 10 pb multiplié par le turnover.</div>
  </div>
  <div class="note"><strong>Probabilité et classement.</strong> Le modèle produit
  une probabilité brute XGBoost et une probabilité calibrée par régression
  isotone pour les diagnostics. Le choix des cinq actions utilise le
  <strong>score brut</strong>, afin de préserver l'ordre appris ; la probabilité
  calibrée n'est pas utilisée pour pondérer le portefeuille.</div>

  <h2>Définition de toutes les méthodes</h2>
  <div class="table-wrap method-table">
    <table>
      <thead><tr><th>Nom affiché</th><th>Sélection des titres</th>
        <th>Pondération</th><th>Rôle</th></tr></thead>
      <tbody>
        <tr><td>Legacy</td><td>Portefeuille déterministe
          <code>Combined_Frequency</code> publié.</td><td>Fréquence de sélection
          par les quatre blocs Legacy, puis normalisation.</td><td>Baseline
          historique.</td></tr>
        <tr><td>SPY</td><td>Aucune sélection d'actions.</td><td>100 % SPY,
          adjusted close avec dividendes réinvestis.</td><td>Benchmark marché.</td></tr>
        <tr><td>Alpha égal</td><td>Top 5 du score brut du classifieur alpha
          six mois.</td><td>20 % par titre.</td><td>Référence ML sans intervention
          du modèle de risque.</td></tr>
        <tr><td>Vol 1m / 3m / 6m</td><td>Même top 5 alpha.</td><td>Inverse de la
          volatilité future prédite à l'horizon indiqué ; poids maximal de
          30 % par titre.</td><td>Réduire l'exposition aux titres annoncés comme
          les plus volatils.</td></tr>
        <tr><td>Down 1m / 3m / 6m</td><td>Même top 5 alpha.</td><td>Inverse du
          downside journalier futur prédit à l'horizon indiqué ; poids maximal
          de 30 %.</td><td>Réduire le poids des titres dont les rendements
          négatifs futurs sont annoncés comme les plus violents.</td></tr>
        <tr><td>Vol 3m + secteur</td><td>Parcourt le classement alpha et garde
          au maximum deux titres par secteur.</td><td>Inverse de la volatilité
          trois mois prédite, maximum 30 % par titre et 40 % par secteur.</td>
          <td>Tester diversification et concentration sectorielle.</td></tr>
      </tbody>
    </table>
  </div>
  <p class="lede">Les têtes de risque sont des XGBoost séparés entraînés sur les
  mêmes variables EMA. Elles prédisent la volatilité réalisée ou le downside
  journalier strictement futurs à 1, 3 ou 6 mois. Chaque mois futur exige au
  moins dix observations journalières. Elles ne choisissent pas les actions :
  elles modifient seulement les poids du top 5 alpha.</p>

  <h2>Fiche technique des boosters</h2>
  <div class="table-wrap method-table">
    <table>
      <thead><tr><th>Modèle</th><th>Cible</th><th>Paramètres principaux</th>
        <th>Usage dans le portefeuille</th><th>Résultat hors échantillon</th></tr></thead>
      <tbody>
        <tr><td>Alpha</td><td>Classification : action dans le top 10 % du
          rendement relatif futur à six mois.</td><td>XGBoost, 100 rounds
          maximum, early stopping 25, profondeur 5, eta 0,04,
          min-child-weight 20, subsample 0,80, colsample 0,75, L2 5, L1 0,2,
          seed 42.</td><td>Classe toutes les actions ; le top 5 forme chaque
          portefeuille.</td><td>15 folds, 76 534 observations test, ROC AUC
          0,589, PR AUC 0,156, lift PR 1,535×, Brier 0,091.</td></tr>
        <tr><td>Risque continu</td><td>Régression du log de la volatilité
          réalisée ou du downside journalier futurs, winsorisés aux quantiles
          1 % et 99 % du train.</td><td>Mêmes hyperparamètres XGBoost et même
          géométrie walk-forward que l'alpha.</td><td>Pondération inverse du
          risque prédit ; ne change pas la sélection alpha.</td><td>Spearman
          mensuel de 0,302 à 0,401 selon cible et horizon.</td></tr>
        <tr><td>Risque élevé</td><td>Classification : top 20 % cross-sectionnel
          de volatilité future.</td><td>XGBoost avec calibration isotone sur la
          validation.</td><td>Diagnostic seulement, non utilisé pour les poids
          présentés dans ce tableau.</td><td>ROC AUC de 0,735 à 0,767 ; PR AUC
          de 0,484 à 0,536.</td></tr>
      </tbody>
    </table>
  </div>
  <div class="note"><strong>Garde-fou de conclusion.</strong> Aucune variante
  de pondération risque ne passe simultanément tous les critères
  pré-enregistrés face à Alpha égal : Sharpe strictement supérieur,
  amélioration du drawdown d'au moins cinq points, perte de CAGR limitée à
  trois points, concentration sectorielle maximale de 40 % et robustesse à
  50 pb de coûts.</div>

  <h2>Résultats annuels</h2>
  <div class="legend">
    <span><i class="swatch positive"></i>rendement positif</span>
    <span><i class="swatch negative"></i>rendement négatif</span>
    <span><strong>Soulignement vert</strong> : meilleure allocation ML de l'année</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th class="year">Année</th><th class="months">Couverture</th>{headers}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>

  <h2>Régularité sur les 13 années complètes</h2>
  <p class="lede">Les années partielles 2011 et 2025 sont exclues de ce résumé.</p>
  <div class="table-wrap summary">
    <table>
      <thead><tr><th>Méthode</th><th>Bat Legacy</th><th>Bat SPY</th>
        <th>Années positives</th><th>Rendement annuel médian</th></tr></thead>
      <tbody>{_summary_rows(wide)}</tbody>
    </table>
  </div>
  <footer>Source : allocation_monthly.csv · holding calendar identique pour
  toutes les séries · génération reproductible par
  scripts/experiments/render_annual_allocation_returns.py.</footer>
</main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render annual allocation returns on the shared calendar."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    annual = annual_returns(pl.read_csv(args.run_dir / "allocation_monthly.csv"))
    wide = annual.pivot(
        on="series",
        index=["year", "months"],
        values="annual_return",
    ).sort("year")
    annual.write_csv(args.run_dir / "annual_returns_all_methods.csv")
    wide.write_csv(args.run_dir / "annual_returns_all_methods_wide.csv")
    output = args.run_dir / "html" / "annual_returns_all_methods.html"
    render_report(annual, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
