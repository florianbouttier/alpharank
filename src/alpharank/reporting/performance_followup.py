"""Attach price-only follow-up to the complete canonical performance desk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.reporting._performance_report_html import render_performance_report_html
from alpharank.reporting.strategy_analysis.index import publish_report_index
from alpharank.reporting.strategy_analysis.report import write_daily_analysis_report

RECENT_METRIC_FIELDS = ["total_return", "max_drawdown"]


def _recent_window(curves: pl.DataFrame, name: str, order: list[str]) -> dict[str, Any]:
    selected = curves.filter(pl.col("window") == name)
    metrics: dict[str, dict[str, float]] = {}
    paths: dict[str, list[dict[str, Any]]] = {}
    reference_dates = selected.filter(pl.col("strategy") == order[0]).sort("date")["date"].to_list()
    for strategy in order:
        frame = selected.filter(pl.col("strategy") == strategy).sort("date")
        if frame.is_empty():
            raise ValueError(f"Missing follow-up curve for {strategy}, {name}.")
        if frame["date"].to_list() != reference_dates or frame["date"].n_unique() != frame.height:
            raise ValueError(
                f"Follow-up curves require a unique common calendar: {strategy}, {name}."
            )
        last = frame.row(-1, named=True)
        metrics[strategy] = {field: float(last[field]) for field in RECENT_METRIC_FIELDS}
        paths[strategy] = [
            {
                "date": row["date"].isoformat(),
                **{field: row[field] for field in ("wealth", "drawdown", "relative_wealth")},
            }
            for row in frame.to_dicts()
        ]
    return {
        "name": name,
        "start_date": selected["date"].min().isoformat(),
        "end_date": selected["date"].max().isoformat(),
        "metrics": metrics,
        "series": paths,
    }


def attach_frozen_followup(
    payload: dict[str, Any], curves: pl.DataFrame, *, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Add precomputed daily follow-up, preserving all historical fields and holdings."""
    order = payload["strategy_order"]
    if set(curves["strategy"].to_list()) != set(order):
        raise ValueError("Follow-up must contain exactly the canonical report's strategies.")
    names = curves["window"].unique(maintain_order=True).to_list()
    windows = [_recent_window(curves, name, order) for name in names]
    recent = {
        "metric_fields": RECENT_METRIC_FIELDS,
        "metric_definitions": {
            "total_return": "Rendement ajusté close-to-close, brut de frais, en USD, sans rééquilibrage.",
            "max_drawdown": "Perte maximale sur les clôtures quotidiennes de cette fenêtre, départ inclus.",
        },
        "latest_close": max(window["end_date"] for window in windows),
        "default_window": max(
            windows, key=lambda window: (window["end_date"], len(window["series"][order[0]]))
        )["name"],
        "holding_month": payload["current_portfolio"]["holding_month"],
        "methodology": {
            "title": "Suivi prix d'un panier figé",
            "status": "Valorisation uniquement · pas un nouveau signal",
            "summary": "Prix ajustés, dividendes inclus, en USD, bruts de frais d'entrée/sortie et sans change. Le maintien après le mois prévu est hypothétique ; aucun mois n'est ajouté au backtest net de frais.",
            "pseudo_code": [
                "Conserver les titres, poids et scores scellés, sans entraînement ni reclassement.",
                "Contrôler chaque prix et l'intégralité du calendrier commun avec SPY.",
                "Laisser dériver les poids depuis l'entrée, sans rééquilibrage entre fenêtres.",
                "Rebaser les courbes à la clôture initiale de chaque fenêtre ; conserver le départ dans le drawdown.",
                "Afficher séparément performance, drawdown quotidien et richesse relative au SPY, sans annualiser cette fenêtre courte.",
                "Un suivi de prix ne remplace pas le replay inter-vintages des modèles.",
            ],
        },
        "windows": {window["name"]: window for window in windows},
        "provenance": provenance,
    }
    return payload | {
        "recent_valuation": recent,
        "generated_at_utc": provenance["generated_at_utc"],
    }


def write_frozen_performance_report(
    payload: dict[str, Any],
    curves: pl.DataFrame,
    *,
    provenance: dict[str, Any],
    output_dir: Path,
    analysis_frequency: Literal["daily", "monthly_archive"] = "monthly_archive",
) -> dict[str, Path]:
    """Write the full autonomous desk and its owned stable index, never a reduced report."""
    combined = attach_frozen_followup(payload, curves, provenance=provenance)
    if analysis_frequency not in {"daily", "monthly_archive"}:
        raise ValueError("Unsupported analysis frequency.")
    if analysis_frequency == "daily":
        return write_daily_analysis_report(
            combined,
            output_dir=output_dir,
            generated_at_utc=datetime.fromisoformat(str(provenance["generated_at_utc"])),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "backtest_performance_report.html"
    if report.exists():
        raise ValueError(f"Refusing to overwrite an existing performance report: {report}")
    report.write_text(render_performance_report_html(combined), encoding="utf-8")
    manifest = output_dir / "backtest_performance_report_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report": {"path": str(report), "sha256": sha256_path(report)},
                "calendar": payload["calendar"],
                "strategies": payload["strategy_order"],
                "metric_fields": payload["metric_fields"],
                "lineage": payload["lineage"],
                "recent_valuation": {
                    key: value
                    for key, value in combined["recent_valuation"].items()
                    if key != "windows"
                },
                "historical_payload_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"report": report, "manifest": manifest, "index": publish_report_index(report)}
