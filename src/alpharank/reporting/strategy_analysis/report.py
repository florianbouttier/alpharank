"""Write an immutable complete report and the explicit datasets used by its local API."""

from __future__ import annotations

import json
import platform
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.reporting._performance_report_html import render_performance_report_html
from alpharank.reporting.strategy_analysis.dataset import (
    AnalysisDataset,
    prepare_analysis_dataset,
    read_performance_payload,
)
from alpharank.reporting.strategy_analysis.index import publish_report_index
from alpharank.reporting.strategy_analysis.service import AnalysisQuery, query_analysis


def write_daily_analysis_report(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    generated_at_utc: datetime,
    source_report: Path | None = None,
) -> dict[str, Path]:
    """Preserve the full historical payload and add source-backed daily capabilities."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "backtest_performance_report.html"
    if report.exists() or (output_dir / "analysis_manifest.json").exists():
        raise ValueError("Refusing to overwrite an archived analysis report: " + str(output_dir))
    if generated_at_utc.tzinfo is None:
        raise ValueError("Report generation time must include a timezone.")
    dataset = prepare_analysis_dataset(payload)
    combined = _daily_payload(dataset, generated_at_utc)
    artifacts = _write_ledgers(dataset, output_dir)
    report.write_text(render_performance_report_html(combined), encoding="utf-8")
    manifest, report_manifest = _write_manifests(
        dataset, report, artifacts, generated_at_utc, source_report
    )
    index = publish_report_index(report)
    return {
        "report": report,
        "manifest": report_manifest,
        "analysis_manifest": manifest,
        "index": index,
    }


def _daily_payload(dataset: AnalysisDataset, generated_at_utc: datetime) -> dict[str, Any]:
    source = dataset.sources["history"]
    end = date.fromisoformat(source["max_date"])
    start = max(date.fromisoformat(source["min_date"]), date(end.year, 1, 1))
    initial = query_analysis(dataset, AnalysisQuery(start, end))
    return dataset.payload | {
        "generated_at_utc": generated_at_utc.isoformat(),
        "analysis": {
            "schema_version": 1,
            "endpoint": "/api/analysis",
            "default_source": "history",
            "default_start": start.isoformat(),
            "default_end": end.isoformat(),
            "sources": {
                name: value
                for name, value in dataset.sources.items()
                if name != "monthly_reference"
            },
            "initial_view": initial,
            "coverage": dataset.history.coverage if dataset.history is not None else {},
            "price_gaps": [
                {
                    "ticker": row["ticker"],
                    "valuation_date": row["valuation_date"].isoformat(),
                    "holding_month": row["holding_month"].isoformat(),
                }
                for row in dataset.history.gaps.iter_rows(named=True)
            ]
            if dataset.history is not None
            else [],
            "runtime": "Read-only local Python engine; custom KPI require the local service.",
        },
    }


def write_monthly_archive_report(payload: dict[str, Any], *, output_dir: Path) -> dict[str, Path]:
    """Retain the historical monthly report schema, without claiming daily inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backtest_performance_report.html"
    report_path.write_text(render_performance_report_html(payload), encoding="utf-8")
    manifest_path = output_dir / "backtest_performance_report_manifest.json"
    manifest = {
        "schema_version": payload["schema_version"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": {
            "path": str(report_path.resolve()),
            "sha256": sha256_path(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "calendar": payload["calendar"],
        "strategies": payload["strategy_order"],
        "metric_fields": payload["metric_fields"],
        "portfolio_composer": {
            "method": payload["portfolio_composer"]["policy"]["method"],
            "combinations": len(payload["portfolio_composer"]["combination_masks"]),
            "metric_fields": payload["portfolio_composer"]["metric_fields"],
            "correlation_windows": len(
                payload["portfolio_composer"]["strategy_correlation_windows"]
            ),
        },
        "strategy_frontiers": {
            "method": payload["strategy_frontiers"]["policy"]["method"],
            "candidate_allocations": len(payload["strategy_frontiers"]["weights"]),
            "windows": len(payload["strategy_frontiers"]["windows"]),
            "anchor_windows": payload["strategy_frontiers"]["anchor_windows"],
            "metric_fields": payload["strategy_frontiers"]["metric_fields"],
        },
        "current_portfolio": {
            "as_of_date": payload["current_portfolio"]["as_of_date"],
            "decision_month": payload["current_portfolio"]["decision_month"],
            "holding_month": payload["current_portfolio"]["holding_month"],
            "status": payload["current_portfolio"]["status"],
            "holding_rows": len(payload["current_portfolio"]["holdings"]),
        },
        "lineage": payload["lineage"],
        "status": payload["status"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"report": report_path, "manifest": manifest_path}


def _write_ledgers(dataset: AnalysisDataset, output_dir: Path) -> dict[str, dict[str, str]]:
    artifacts = {}
    for name, frame in dataset.ledgers.items():
        filename = (
            "monthly_reference_ledger.parquet"
            if name == "monthly_reference"
            else name + "_daily_ledger.parquet"
        )
        path = output_dir / filename
        frame.write_parquet(path)
        artifacts[name] = {"path": path.name, "sha256": sha256_path(path)}
    if dataset.history is not None:
        dataset.history.monthly_reconciliation.write_parquet(
            output_dir / "monthly_reconciliation.parquet"
        )
        dataset.history.gaps.write_parquet(output_dir / "daily_price_gaps.parquet")
    return artifacts


def _code_proof() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    paths = [
        package_root / name
        for name in [
            "portfolio/performance.py",
            "portfolio/frontiers.py",
            "reporting/_performance_report_html.py",
            "reporting/analysis_ui/__init__.py",
            "reporting/analysis_ui/client.bundle.js",
            "reporting/analysis_ui/layout.css",
            "reporting/analysis_ui/package-lock.json",
        ]
    ]
    for name in ["portfolio/daily", "reporting/strategy_analysis"]:
        paths.extend(sorted((package_root / name).glob("*.py")))
    return {
        "alpharank/" + path.relative_to(package_root).as_posix(): sha256_path(path)
        for path in paths
    }


def _write_manifests(
    dataset: AnalysisDataset,
    report: Path,
    artifacts: dict[str, dict[str, str]],
    generated_at_utc: datetime,
    source_report: Path | None,
) -> tuple[Path, Path]:
    manifest = report.parent / "analysis_manifest.json"
    metadata = {
        "schema_version": 1,
        "benchmark_strategy": dataset.benchmark_strategy,
        "report": {"path": report.name, "sha256": sha256_path(report)},
        "sources": dataset.sources,
        "ledgers": artifacts,
        "coverage": dataset.history.coverage if dataset.history is not None else {},
        "publication_eligible": False,
        "model_replay": "not_rerun",
        "generated_at_utc": generated_at_utc.isoformat(),
        "code_sha256": _code_proof(),
        "runtime": {"python": platform.python_version(), "polars": pl.__version__},
        "source_report": {"name": source_report.name, "sha256": sha256_path(source_report)}
        if source_report is not None
        else None,
    }
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_manifest = report.parent / "backtest_performance_report_manifest.json"
    report_manifest.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "report": metadata["report"],
                "lineage": dataset.payload["lineage"],
                "calendar": dataset.payload["calendar"],
                "strategies": dataset.payload["strategy_order"],
                "metric_fields": dataset.payload["metric_fields"],
                "analysis_manifest_sha256": sha256_path(manifest),
                "historical_payload_preserved": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, report_manifest


def load_analysis_report(report_dir: Path) -> AnalysisDataset:
    """Load only the hash-verified artifacts captured by this specific report."""
    directory = report_dir.resolve()
    manifest = json.loads((directory / "analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported analysis manifest version.")
    report = _verified_artifact(directory, manifest["report"])
    payload = read_performance_payload(report)
    ledgers = {
        name: pl.read_parquet(_verified_artifact(directory, info))
        for name, info in manifest["ledgers"].items()
    }
    return AnalysisDataset(payload, ledgers, manifest["sources"], manifest["benchmark_strategy"])


def _verified_artifact(directory: Path, info: dict[str, str]) -> Path:
    path = (directory / info["path"]).resolve()
    if not path.is_relative_to(directory) or sha256_path(path) != info["sha256"]:
        raise ValueError("Analysis artifact path or SHA-256 is invalid: " + info["path"])
    return path
