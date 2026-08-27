"""Build an offline SEC download explorer from one explicit ingestion run."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import polars as pl

from alpharank.reporting._sec_explorer_html import render_sec_explorer_html


@dataclass(frozen=True, slots=True)
class SecDatasetSpec:
    """One downloaded SEC-facing table exposed without row selection."""

    key: str
    filename: str
    label: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecExplorerConfig:
    """Explicit inputs for one deterministic static audit report."""

    raw_run_dir: Path
    output_dir: Path
    project_root: Path
    generated_at_utc: datetime
    initial_ticker: str = "AAPL.US"


@dataclass(frozen=True, slots=True)
class SecExplorerResult:
    """Paths and counts produced by the SEC explorer build."""

    report_path: Path
    manifest_path: Path
    run_id: str
    company_count: int
    sec_row_count: int
    report_sha256: str


DATASET_SPECS = (
    SecDatasetSpec(
        key="companyfacts",
        filename="financials_sec_companyfacts.parquet",
        label="Companyfacts — faits financiers",
        required_columns=(
            "ticker",
            "statement",
            "metric",
            "date",
            "filing_date",
            "value",
            "source_label",
            "form",
            "fiscal_period",
            "fiscal_year",
            "ingestion_run_id",
        ),
    ),
    SecDatasetSpec(
        key="filing_fallback",
        filename="financials_sec_filing.parquet",
        label="Filing XBRL — extraction de secours",
        required_columns=(
            "ticker",
            "statement",
            "metric",
            "date",
            "filing_date",
            "value",
            "accession_number",
            "form",
            "fiscal_period",
            "fiscal_year",
            "ingestion_run_id",
        ),
    ),
    SecDatasetSpec(
        key="filing_calendar",
        filename="earnings_sec_calendar.parquet",
        label="Submissions — calendrier des dépôts",
        required_columns=(
            "ticker",
            "period_end",
            "reportDate",
            "accession_number",
            "form",
            "fiscal_period",
            "fiscal_year",
            "ingestion_run_id",
        ),
    ),
    SecDatasetSpec(
        key="earnings_actuals",
        filename="earnings_sec_actuals.parquet",
        label="Companyfacts — EPS publiés",
        required_columns=(
            "ticker",
            "period_end",
            "reportDate",
            "epsActual",
            "source_label",
            "form",
            "fiscal_period",
            "fiscal_year",
            "ingestion_run_id",
        ),
    ),
    SecDatasetSpec(
        key="company_reference",
        filename="general_reference.parquet",
        label="Référentiel société téléchargé",
        required_columns=("ticker", "name", "exchange", "cik", "ingestion_run_id"),
    ),
    SecDatasetSpec(
        key="company_reference_lineage",
        filename="general_reference_lineage.parquet",
        label="Lignée du référentiel société",
        required_columns=(
            "ticker",
            "sec_name",
            "sec_exchange",
            "sec_cik",
            "sec_sic",
            "sec_sic_description",
            "ingestion_run_id",
        ),
    ),
)

SEC_FACT_DATASETS = (
    "companyfacts",
    "filing_fallback",
    "filing_calendar",
    "earnings_actuals",
)


def build_sec_fundamental_explorer(config: SecExplorerConfig) -> SecExplorerResult:
    """Render every downloaded SEC-facing row from one immutable run.

    The report does not consolidate, select, promote, or mutate facts. It raises
    when a required file, schema, finite numeric value, or run identity is not
    demonstrable.
    """

    raw_run_dir = config.raw_run_dir.resolve()
    output_dir = config.output_dir.resolve()
    project_root = config.project_root.resolve()
    run_id = raw_run_dir.parent.name
    _validate_config(config=config, raw_run_dir=raw_run_dir, run_id=run_id)

    frames = _read_and_validate_frames(raw_run_dir=raw_run_dir, run_id=run_id)
    acquisition_status = _read_json_object(raw_run_dir.parent / "acquisition_status.json")
    _validate_status_run_id(acquisition_status=acquisition_status, run_id=run_id)
    source_contract = _read_optional_json_object(
        raw_run_dir.parent / "source_refresh_contract.json"
    )
    source_files = _source_file_records(
        raw_run_dir=raw_run_dir,
        project_root=project_root,
        frames=frames,
    )
    payload = _build_payload(
        run_id=run_id,
        generated_at_utc=config.generated_at_utc,
        frames=frames,
        source_files=source_files,
        acquisition_status=acquisition_status,
        source_contract=source_contract,
    )
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    compressed_payload = gzip.compress(payload_json, compresslevel=9, mtime=0)
    encoded_payload = base64.b64encode(compressed_payload).decode("ascii")

    initial_ticker = _resolve_initial_ticker(payload=payload, requested=config.initial_ticker)
    report_html = render_sec_explorer_html(
        encoded_payload=encoded_payload,
        run_id=run_id,
        initial_ticker=initial_ticker,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    report_sha256 = _sha256(report_path)

    company_count = len(payload["companies"])
    sec_row_count = sum(int(payload["datasets"][key]["row_count"]) for key in SEC_FACT_DATASETS)
    manifest = _build_manifest(
        run_id=run_id,
        generated_at_utc=config.generated_at_utc,
        report_sha256=report_sha256,
        payload_json=payload_json,
        compressed_payload=compressed_payload,
        source_files=source_files,
        company_count=company_count,
        sec_row_count=sec_row_count,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SecExplorerResult(
        report_path=report_path,
        manifest_path=manifest_path,
        run_id=run_id,
        company_count=company_count,
        sec_row_count=sec_row_count,
        report_sha256=report_sha256,
    )


def _validate_config(*, config: SecExplorerConfig, raw_run_dir: Path, run_id: str) -> None:
    if not raw_run_dir.is_dir() or raw_run_dir.name != "raw":
        raise ValueError(f"Expected an explicit ingestion raw directory: {raw_run_dir}")
    if not run_id or run_id == "official":
        raise ValueError(f"Cannot resolve ingestion run id from {raw_run_dir}")
    if config.generated_at_utc.tzinfo is None or config.generated_at_utc.utcoffset() is None:
        raise ValueError("generated_at_utc must be timezone-aware")
    if config.generated_at_utc.utcoffset().total_seconds() != 0:
        raise ValueError("generated_at_utc must use UTC")


def _read_and_validate_frames(*, raw_run_dir: Path, run_id: str) -> dict[str, pl.DataFrame]:
    frames: dict[str, pl.DataFrame] = {}
    for spec in DATASET_SPECS:
        path = raw_run_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Required SEC explorer input is missing: {path}")
        frame = pl.read_parquet(path)
        _validate_frame(spec=spec, frame=frame, run_id=run_id)
        frames[spec.key] = frame
    return frames


def _validate_frame(*, spec: SecDatasetSpec, frame: pl.DataFrame, run_id: str) -> None:
    missing = sorted(set(spec.required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.filename} lacks required columns: {missing}")
    if frame.get_column("ticker").null_count():
        raise ValueError(f"{spec.filename}.ticker contains null raw identities")
    if frame.get_column("ingestion_run_id").null_count():
        raise ValueError(f"{spec.filename}.ingestion_run_id contains null run identities")
    run_ids = frame.get_column("ingestion_run_id").drop_nulls().unique().to_list()
    if frame.height and run_ids != [run_id]:
        raise ValueError(
            f"{spec.filename} is not sealed to run {run_id}: ingestion_run_id={run_ids}"
        )
    for column, dtype in frame.schema.items():
        if (
            dtype.is_float()
            and frame.select((pl.col(column).is_nan() | pl.col(column).is_infinite()).any()).item()
        ):
            raise ValueError(f"{spec.filename}.{column} contains non-finite raw values")


def _validate_status_run_id(*, acquisition_status: Mapping[str, object], run_id: str) -> None:
    status_run_id = str(acquisition_status.get("run_id") or "")
    if status_run_id != run_id:
        raise ValueError(
            f"acquisition_status.json belongs to run {status_run_id!r}, expected {run_id!r}"
        )


def _source_file_records(
    *,
    raw_run_dir: Path,
    project_root: Path,
    frames: Mapping[str, pl.DataFrame],
) -> list[dict[str, object]]:
    records = []
    for spec in DATASET_SPECS:
        path = raw_run_dir / spec.filename
        records.append(
            {
                "dataset": spec.key,
                "label": spec.label,
                "path": _portable_path(path=path, project_root=project_root),
                "filename": spec.filename,
                "row_count": frames[spec.key].height,
                "column_count": frames[spec.key].width,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _portable_path(*, path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def _build_payload(
    *,
    run_id: str,
    generated_at_utc: datetime,
    frames: Mapping[str, pl.DataFrame],
    source_files: list[dict[str, object]],
    acquisition_status: Mapping[str, object],
    source_contract: Mapping[str, object],
) -> dict[str, object]:
    datasets = {
        spec.key: _frame_payload(spec=spec, frame=frames[spec.key]) for spec in DATASET_SPECS
    }
    profiles = _company_profiles(frames=frames, datasets=datasets)
    source_statuses = _sec_source_statuses(acquisition_status)
    sec_contract = _sec_contract_excerpt(source_contract)
    return {
        "schema_version": 1,
        "meta": {
            "run_id": run_id,
            "generated_at_utc": generated_at_utc.isoformat(),
            "artifact_role": "static_sec_download_audit",
            "data_promotion_status": "not_applicable",
            "source_files": source_files,
            "source_statuses": source_statuses,
            "source_contract": sec_contract,
        },
        "companies": profiles,
        "datasets": datasets,
    }


def _frame_payload(*, spec: SecDatasetSpec, frame: pl.DataFrame) -> dict[str, object]:
    rows_by_ticker: dict[str, list[list[object]]] = {}
    ordered = frame.sort(
        [
            column
            for column in ("ticker", "date", "period_end", "filing_date", "reportDate")
            if column in frame.columns
        ]
    )
    for ticker_key, group in ordered.group_by("ticker", maintain_order=True):
        ticker = str(ticker_key[0] if isinstance(ticker_key, tuple) else ticker_key)
        rows_by_ticker[ticker] = [list(row) for row in group.rows()]
    return {
        "label": spec.label,
        "filename": spec.filename,
        "columns": frame.columns,
        "row_count": frame.height,
        "rows_by_ticker": rows_by_ticker,
    }


def _company_profiles(
    *,
    frames: Mapping[str, pl.DataFrame],
    datasets: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    ticker_set: set[str] = set()
    for frame in frames.values():
        ticker_set.update(str(value) for value in frame.get_column("ticker").drop_nulls().to_list())
    lineage_rows = {
        str(row["ticker"]): row
        for row in frames["company_reference_lineage"].sort("ticker").to_dicts()
    }
    reference_rows = {
        str(row["ticker"]): row for row in frames["company_reference"].sort("ticker").to_dicts()
    }
    profiles = []
    for ticker in sorted(ticker_set):
        lineage = lineage_rows.get(ticker, {})
        reference = reference_rows.get(ticker, {})
        profiles.append(
            {
                "ticker": ticker,
                "display_ticker": ticker.removesuffix(".US"),
                "name": lineage.get("sec_name") or reference.get("name") or ticker,
                "exchange": lineage.get("sec_exchange") or reference.get("exchange"),
                "cik": lineage.get("sec_cik") or reference.get("cik"),
                "sic": lineage.get("sec_sic") or reference.get("sic"),
                "sic_description": lineage.get("sec_sic_description")
                or reference.get("sic_description"),
                "sector": reference.get("Sector"),
                "industry": reference.get("industry"),
                "row_counts": {
                    key: len(value["rows_by_ticker"].get(ticker, []))
                    for key, value in datasets.items()
                },
            }
        )
    return profiles


def _sec_source_statuses(acquisition_status: Mapping[str, object]) -> list[dict[str, object]]:
    raw_sources = acquisition_status.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("acquisition_status.json lacks a sources list")
    statuses = []
    for item in raw_sources:
        if not isinstance(item, dict) or not str(item.get("source", "")).startswith("sec_"):
            continue
        statuses.append(
            {
                "source": item.get("source"),
                "status": item.get("status"),
                "downloaded_rows": item.get("downloaded_rows"),
                "failure_count": item.get("failure_count"),
                "failure_examples": item.get("failure_examples") or [],
            }
        )
    if not statuses:
        raise ValueError("acquisition_status.json contains no SEC acquisition status")
    return statuses


def _sec_contract_excerpt(source_contract: Mapping[str, object]) -> dict[str, object]:
    semantics = source_contract.get("source_semantics")
    if not isinstance(semantics, dict):
        return {}
    return {
        key: semantics[key]
        for key in ("sec_companyfacts", "sec_submissions", "sec_filing_documents")
        if key in semantics
    }


def _resolve_initial_ticker(*, payload: Mapping[str, object], requested: str) -> str:
    companies = payload["companies"]
    tickers = [str(company["ticker"]) for company in companies]
    if not tickers:
        raise ValueError("SEC explorer input contains no company")
    return requested if requested in tickers else tickers[0]


def _build_manifest(
    *,
    run_id: str,
    generated_at_utc: datetime,
    report_sha256: str,
    payload_json: bytes,
    compressed_payload: bytes,
    source_files: list[dict[str, object]],
    company_count: int,
    sec_row_count: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "family": "sec_fundamental_explorer",
        "run_id": run_id,
        "status": "validated",
        "artifact_role": "static_sec_download_audit",
        "data_promotion_status": "not_applicable",
        "generated_at_utc": generated_at_utc.isoformat(),
        "report": {
            "path": "report.html",
            "sha256": report_sha256,
            "self_contained": True,
            "external_assets": 0,
        },
        "payload": {
            "encoding": "base64+gzip+json",
            "schema_version": 1,
            "uncompressed_size_bytes": len(payload_json),
            "compressed_size_bytes": len(compressed_payload),
            "sha256": hashlib.sha256(payload_json).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed_payload).hexdigest(),
        },
        "coverage": {
            "company_count": company_count,
            "sec_row_count": sec_row_count,
            "source_file_count": len(source_files),
        },
        "source_files": source_files,
    }


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required SEC explorer input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_optional_json_object(path: Path) -> dict[str, object]:
    return _read_json_object(path) if path.is_file() else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
