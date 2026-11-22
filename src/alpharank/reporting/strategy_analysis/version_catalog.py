"""Capture an explicitly selected, immutable collection of complete backtest reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio.daily.contracts import DailyValuationPolicy
from alpharank.reporting.strategy_analysis.dataset import (
    AnalysisDataset,
    apply_valuation_policy,
    prepare_analysis_dataset,
    read_performance_payload,
)
from alpharank.reporting.strategy_analysis.report import (
    _verified_artifact,
    load_analysis_report,
    write_daily_analysis_report,
)
from alpharank.reporting.strategy_analysis.versions import backtest_version, economic_identity


@dataclass(frozen=True, slots=True)
class AnalysisCollection:
    """Already verified datasets/HTML, keyed only by registered economic identifiers."""

    default_version_id: str
    datasets: dict[str, AnalysisDataset]
    reports: dict[str, bytes]
    metadata: dict[str, Any]


def write_analysis_collection(
    source_report: Path,
    *,
    archive_reports: Sequence[Path],
    output_dir: Path,
    generated_at_utc: datetime,
    valuation_policy: DailyValuationPolicy | None = None,
) -> dict[str, Path]:
    """Re-render explicit economic vintages without overwriting an archive or rerunning a model."""
    if output_dir.exists():
        raise ValueError("Refusing to overwrite an archived version collection: " + str(output_dir))
    sources = [source_report, *archive_reports]
    payloads = [read_performance_payload(path) for path in sources]
    payloads[0] = apply_valuation_policy(payloads[0], valuation_policy)
    entries = [
        backtest_version(prepare_analysis_dataset(payload), source_report=path)
        for payload, path in zip(payloads, sources, strict=True)
    ]
    identities = [entry["id"] for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate economic backtest version; a UI render is not another run.")
    default = identities[0]
    captured: dict[str, dict[str, Path]] = {}
    for index in [*range(1, len(entries)), 0]:
        entry = entries[index]
        directory = output_dir if index == 0 else output_dir / "versions" / entry["id"]
        captured[entry["id"]] = write_daily_analysis_report(
            payloads[index],
            output_dir=directory,
            generated_at_utc=generated_at_utc,
            source_report=sources[index],
            version_context={
                "current_version_id": entry["id"],
                "default_version_id": default,
                "entries": entries,
            },
        )
    catalog = {
        "schema_version": 1,
        "default_version_id": default,
        "entries": [
            {
                **entry,
                **{
                    key: {
                        "path": captured[entry["id"]][key].relative_to(output_dir).as_posix(),
                        "sha256": sha256_path(captured[entry["id"]][key]),
                    }
                    for key in ["report", "analysis_manifest"]
                },
            }
            for entry in entries
        ],
    }
    path = output_dir / "versions_catalog.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return captured[default] | {"versions_catalog": path}


def _load_entry(directory: Path, entry: dict[str, Any]) -> tuple[AnalysisDataset, bytes]:
    report = _verified_artifact(directory, entry["report"])
    manifest = _verified_artifact(directory, entry["analysis_manifest"])
    if manifest.parent != report.parent or manifest.name != "analysis_manifest.json":
        raise ValueError("Version report/analysis manifest path mismatch.")
    dataset = load_analysis_report(report.parent)
    if entry["id"] != "bt-" + economic_identity(dataset.payload)[:16]:
        raise ValueError("Version identity differs from its captured economic payload.")
    context = dataset.payload["analysis"]["versions"]
    expected = next((row for row in context["entries"] if row["id"] == entry["id"]), None)
    metadata = {
        key: value for key, value in entry.items() if key not in {"report", "analysis_manifest"}
    }
    if context["current_version_id"] != entry["id"] or expected != metadata:
        raise ValueError("Version metadata differs from its hash-captured report.")
    return dataset, report.read_bytes()


def load_analysis_collection(report_dir: Path) -> AnalysisCollection:
    """Load registered relative artifacts only; never discover a run by folder recency."""
    directory = report_dir.resolve()
    path = directory / "versions_catalog.json"
    if not path.exists():
        dataset = load_analysis_report(directory)
        identity = "bt-" + economic_identity(dataset.payload)[:16]
        context = dataset.payload.get("analysis", {}).get("versions", {})
        if len(context.get("entries", [])) > 1:
            raise ValueError("A captured version collection must be served from its root catalog.")
        metadata = {
            "schema_version": 1,
            "default_version_id": identity,
            "entries": context.get("entries", []),
        }
        return AnalysisCollection(
            identity,
            {identity: dataset},
            {identity: (directory / "backtest_performance_report.html").read_bytes()},
            metadata,
        )
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(catalog) != {"schema_version", "default_version_id", "entries"}
        or catalog["schema_version"] != 1
    ):
        raise ValueError("Unsupported or invalid backtest version catalog.")
    entries = catalog["entries"]
    identities = [entry["id"] for entry in entries]
    default = catalog["default_version_id"]
    if not entries or len(set(identities)) != len(identities) or default not in identities:
        raise ValueError("Duplicate or unknown default backtest version.")
    datasets, reports = {}, {}
    expected_metadata = [
        {key: value for key, value in entry.items() if key not in {"report", "analysis_manifest"}}
        for entry in entries
    ]
    for entry in entries:
        datasets[entry["id"]], reports[entry["id"]] = _load_entry(directory, entry)
        context = datasets[entry["id"]].payload["analysis"]["versions"]
        if context["default_version_id"] != default or context["entries"] != expected_metadata:
            raise ValueError("Catalog versions/default differ from their captured reports.")
    return AnalysisCollection(default, datasets, reports, catalog)
