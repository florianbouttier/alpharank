#!/usr/bin/env python3
"""Render inception-year CAGR and calendar returns for Legacy, Boosting, and SPY."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.portfolio.performance import (  # noqa: E402
    advanced_performance_statistics,
    annual_returns,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio  # noqa: E402


STRATEGIES = ("Boosting Top 5", "Boosting Top 10", "Legacy", "SPY total return")


def _series_sources(
    common_monthly: pl.DataFrame,
    legacy_monthly: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    sources: dict[str, pl.DataFrame] = {}
    for strategy in STRATEGIES[:2]:
        sources[strategy] = common_monthly.filter(pl.col("strategy") == strategy)
    sources["Legacy"] = legacy_monthly.filter(
        pl.col("strategy") == "Combined_Frequency"
    ).with_columns(pl.lit("Legacy").alias("strategy"))
    sources["SPY total return"] = legacy_monthly.filter(
        pl.col("strategy") == "SPY total return"
    )
    return sources


def build_start_year_performance(
    common_monthly: pl.DataFrame,
    legacy_monthly: pl.DataFrame,
    *,
    first_year: int,
) -> pl.DataFrame:
    """Calculate canonical metrics from each requested January to common end."""

    common_end = common_monthly["holding_month"].max()
    sources = _series_sources(common_monthly, legacy_monthly)
    benchmark = sources["SPY total return"].select(
        "holding_month",
        pl.col("net_return").alias("benchmark_return"),
    )
    rows: list[dict] = []
    for start_year in range(first_year, common_end.year + 1):
        requested_start = date(start_year, 1, 1)
        for strategy in STRATEGIES:
            frame = (
                sources[strategy]
                .filter(
                    pl.col("holding_month").is_between(
                        requested_start,
                        common_end,
                    )
                )
                .join(benchmark, on="holding_month", how="inner")
                .sort("holding_month")
            )
            if frame.is_empty():
                continue
            effective_start = frame["holding_month"].min()
            metrics = advanced_performance_statistics(
                frame["net_return"].to_numpy(),
                benchmark_returns=frame["benchmark_return"].to_numpy(),
            )
            rows.append(
                {
                    "requested_start_year": start_year,
                    "strategy": strategy,
                    "effective_start_month": effective_start,
                    "end_month": common_end,
                    "months": frame.height,
                    "coverage": (
                        "full_from_january"
                        if effective_start == requested_start
                        else f"partial_from_{effective_start:%Y-%m}"
                    ),
                    **metrics,
                }
            )
    return pl.DataFrame(rows).sort(["requested_start_year", "strategy"])


def build_calendar_returns(
    common_monthly: pl.DataFrame,
    legacy_monthly: pl.DataFrame,
    *,
    first_year: int,
) -> pl.DataFrame:
    """Build annual returns without inventing unavailable Boosting months."""

    common_end = common_monthly["holding_month"].max()
    parts: list[pl.DataFrame] = []
    for strategy, source in _series_sources(common_monthly, legacy_monthly).items():
        frame = source.filter(
            (pl.col("holding_month") >= date(first_year, 1, 1))
            & (pl.col("holding_month") <= common_end)
        ).sort("holding_month")
        yearly = annual_returns(
            frame["net_return"].to_numpy(),
            holding_months=frame["holding_month"].to_list(),
        ).with_columns(pl.lit(strategy).alias("strategy"))
        parts.append(yearly.select("year", "strategy", "months", "is_full_calendar_year", "annual_return"))
    return pl.concat(parts, how="vertical").sort(["year", "strategy"])


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(
    *,
    common_dir: Path,
    legacy_run_dir: Path,
    output_dir: Path,
    first_year: int = 2010,
) -> Path:
    common_monthly = pl.read_parquet(common_dir / "comparison_common_monthly.parquet")
    comparison_manifest = json.loads(
        (common_dir / "manifest.json").read_text(encoding="utf-8")
    )
    transaction_cost_bps = comparison_manifest.get(
        "transaction_cost_policy", {}
    ).get(
        "bps_times_turnover",
        comparison_manifest.get("transaction_cost_bps_times_turnover", 0.0),
    )
    legacy_monthly_source = pl.read_parquet(
        legacy_run_dir / "legacy_common_monthly.parquet"
    )
    legacy_holdings = pl.read_parquet(
        legacy_run_dir / "legacy_common_holdings.parquet"
    ).filter(pl.col("strategy") == "Combined_Frequency")
    legacy_net = simulate_weighted_portfolio(
        legacy_holdings,
        transaction_cost_bps=transaction_cost_bps,
    ).with_columns(pl.lit("Combined_Frequency").alias("strategy"))
    legacy_monthly = pl.concat(
        [
            legacy_net,
            legacy_monthly_source.filter(
                pl.col("strategy") == "SPY total return"
            ),
        ],
        how="diagonal_relaxed",
    )
    start_year = build_start_year_performance(
        common_monthly,
        legacy_monthly,
        first_year=first_year,
    )
    calendar = build_calendar_returns(
        common_monthly,
        legacy_monthly,
        first_year=first_year,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start_year.write_csv(output_dir / "cagr_by_start_year.csv")
    calendar.write_csv(output_dir / "calendar_year_returns.csv")

    cagr_rows = [
        [
            str(row["requested_start_year"]),
            str(row["strategy"]),
            str(row["effective_start_month"])[:7],
            str(row["end_month"])[:7],
            str(row["months"]),
            _pct(row["cagr"]),
            _pct(row["total_return"]),
            f"{row['sharpe']:.2f}",
            _pct(row["max_drawdown"]),
            str(row["coverage"]),
        ]
        for row in start_year.to_dicts()
    ]
    annual_rows = [
        [
            str(row["year"]),
            str(row["strategy"]),
            str(row["months"]),
            _pct(row["annual_return"]),
            "complet" if row["is_full_calendar_year"] else "partiel",
        ]
        for row in calendar.to_dicts()
    ]
    report = output_dir / "performance_since_each_year.html"
    report.write_text(
        """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaRank - performance par annee de depart</title>
<style>
body{margin:0;background:#f4f6f8;color:#172033;font:14px Inter,system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 8px}h2{margin-top:32px}
p{color:#536079;max-width:1000px;line-height:1.5}table{width:100%;border-collapse:collapse;background:white}
th,td{padding:9px 10px;border:1px solid #dce2ea;text-align:right;white-space:nowrap}
th{background:#eef2f6;color:#46526a;font-size:12px;text-transform:uppercase}th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{text-align:left}
.note{border-left:4px solid #d89b21;background:#fff8e8;padding:12px 16px;color:#624713}
</style></head><body><main>
<h1>Legacy, Boosting et SPY</h1>
<p>Les CAGR sont recalcules avec le moteur commun depuis chaque annee demandee jusqu'au dernier mois realise commun. Legacy et Boosting utilisent la meme politique de 10 pb multiplies par le turnover.</p>
<p class="note">Boosting commence hors echantillon en aout 2011. Les lignes 2010 et 2011 indiquent donc explicitement une couverture partielle; aucun rendement Boosting anterieur n'est invente. Legacy et SPY utilisent leur historique disponible sur le meme snapshot.</p>
<h2>CAGR depuis chaque annee</h2>"""
        + _table(
            ["Depart demande", "Strategie", "Depart effectif", "Fin", "Mois", "CAGR", "Rendement total", "Sharpe", "Max DD", "Couverture"],
            cagr_rows,
        )
        + "<h2>Rendements calendaires</h2>"
        + _table(["Annee", "Strategie", "Mois", "Rendement", "Statut"], annual_rows)
        + "</main></body></html>\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-year", type=int, default=2010)
    args = parser.parse_args()
    print(
        render(
            common_dir=args.common_dir.resolve(),
            legacy_run_dir=args.legacy_run_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            first_year=args.first_year,
        )
    )


if __name__ == "__main__":
    main()
