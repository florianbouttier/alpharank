"""Verify an explicit cumulative SEC parent before reusing its filing versions."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from alpharank.data.ingestion.frames import _concat_or_empty
from alpharank.data.ingestion.prices import _drop_refreshed_partitions
from alpharank.data.ingestion.raw_archive_frames import file_sha256
from alpharank.data.ingestion.reference import _upsert_financial_dataset
from alpharank.data.ingestion.refresh_policy import SourceRefreshPolicy
from alpharank.data.ingestion.storage import (
    OpenSourceLivePaths,
    append_run_delta,
    read_json,
    upsert_parquet,
    write_json,
)
from alpharank.data.sources.sec_raw_versions import (
    SEC_RAW_DELTA_KEYS,
    build_sec_raw_version_candidate,
    require_consistent_sec_issuers,
)

SEC_PARENT_ENV = "ALPHARANK_SEC_CUMULATIVE_PARENT_MANIFEST"
COMPANYFACTS = "financials_sec_companyfacts.parquet"
FILING = "financials_sec_filing.parquet"
CALENDAR = "earnings_sec_calendar.parquet"
ACTUALS = "earnings_sec_actuals.parquet"
GENERAL = "general_reference_lineage.parquet"
KNOWN_DATE_COLUMNS = {
    COMPANYFACTS: "filing_date",
    FILING: "filing_date",
    CALENDAR: "reportDate",
    ACTUALS: "reportDate",
}


@dataclass(frozen=True)
class SecCumulativeParent:
    manifest: Path
    manifest_sha256: str
    file_sha256: dict[str, str]
    latest_known_date: date


@dataclass(frozen=True)
class SecRawFrames:
    companyfacts: pl.DataFrame
    filing: pl.DataFrame
    calendar: pl.DataFrame
    actuals: pl.DataFrame
    receipt: dict[str, object] = field(default_factory=dict)


def configured_sec_parent_manifest() -> Path | None:
    """Read the optional SEC parent configured for the nightly process."""
    configured = os.environ.get(SEC_PARENT_ENV, "").strip()
    return Path(configured).expanduser().resolve() if configured else None


def resolve_sec_cumulative_parent(
    manifest: Path | None, *, cutoff: str
) -> SecCumulativeParent | None:
    """Read the DATA-052 output manifest and check every input before acquisition."""
    if manifest is None:
        return None
    manifest = manifest.resolve()
    payload = read_json(manifest)
    files = payload.get("cumulative_raw_outputs") if isinstance(payload, dict) else None
    if not isinstance(files, dict) or set(files) != {*KNOWN_DATE_COLUMNS, GENERAL}:
        raise RuntimeError(f"SEC cumulative parent requires a DATA-052 output manifest: {manifest}")
    hashes = _validate_parent_files(manifest.parent, files)
    dates = [
        _latest_known_date(manifest.parent / name, column)
        for name, column in KNOWN_DATE_COLUMNS.items()
    ]
    known = max((value for value in dates if value is not None), default=None)
    if known is None or known > date.fromisoformat(cutoff):
        raise RuntimeError(
            f"SEC cumulative parent knowledge date {known} is missing or later than {cutoff}"
        )
    return SecCumulativeParent(manifest, file_sha256(manifest), hashes, known)


def _validate_parent_files(directory: Path, files: Mapping[str, object]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, record in sorted(files.items()):
        source = directory / name
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise RuntimeError(f"SEC cumulative parent has no file hash: {name}")
        if not source.is_file() or source.resolve().parent != directory:
            raise RuntimeError(
                f"SEC cumulative parent file is missing or outside its package: {name}"
            )
        digest = file_sha256(source)
        if digest != record["sha256"]:
            raise RuntimeError(f"SEC cumulative parent hash mismatch: {name}")
        hashes[name] = digest
    return hashes


def _latest_known_date(path: Path, column: str) -> date | None:
    parsed = pl.col(column).cast(pl.String).str.to_date(strict=False)
    latest, invalid = (
        pl.scan_parquet(path)
        .select(
            parsed.max().alias("latest"), (pl.col(column).is_not_null() & parsed.is_null()).any()
        )
        .collect()
        .row(0)
    )
    if invalid:
        raise RuntimeError(f"SEC cumulative parent has invalid knowledge dates: {path.name}")
    return latest


def prepare_cumulative_sec_frames(
    *,
    parent: SecCumulativeParent,
    run_directory: Path,
    companyfacts: pl.DataFrame,
    filing: pl.DataFrame,
    calendar: pl.DataFrame,
    actuals: pl.DataFrame,
    issuer_mapping: pl.DataFrame | None = None,
) -> SecRawFrames:
    """Retain a reconciled RAW candidate and its receipt inside this run only."""
    deltas = run_directory / "raw"
    for name, frame in {
        COMPANYFACTS: companyfacts,
        FILING: filing,
        CALENDAR: calendar,
        ACTUALS: actuals,
    }.items():
        if not (deltas / name).exists():
            append_run_delta(deltas / name, frame)
    if file_sha256(parent.manifest) != parent.manifest_sha256:
        raise RuntimeError("SEC cumulative parent manifest changed during acquisition")
    issuer_receipt = _verify_acquisition_issuers(parent, issuer_mapping, run_directory)
    if not (deltas / GENERAL).exists():
        empty_reference = pl.read_parquet(parent.manifest.parent / GENERAL, n_rows=0)
        append_run_delta(deltas / GENERAL, empty_reference)
    output = run_directory / "sec_cumulative"
    rebuilt = build_sec_raw_version_candidate(
        retained_raw_dir=parent.manifest.parent, run_raw_dir=deltas, output_dir=output
    )
    retained_hashes = {
        name: record["sha256"] for name, record in rebuilt["retained_raw_sources"].items()
    }
    if retained_hashes != parent.file_sha256:
        raise RuntimeError("SEC cumulative parent files changed during acquisition")
    inherited = {
        "manifest_path": str(parent.manifest),
        "manifest_sha256": parent.manifest_sha256,
        "latest_known_date": parent.latest_known_date.isoformat(),
    }
    manifest = output / "version_rebuild_manifest.json"
    write_json(manifest, {**rebuilt, "parent": inherited, "acquisition_issuers": issuer_receipt})
    return SecRawFrames(
        companyfacts=pl.read_parquet(output / COMPANYFACTS),
        filing=pl.read_parquet(output / FILING),
        calendar=pl.read_parquet(output / CALENDAR),
        actuals=pl.read_parquet(output / ACTUALS),
        receipt={
            "parent": inherited,
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        },
    )


def _verify_acquisition_issuers(
    parent: SecCumulativeParent, mapping: pl.DataFrame | None, run_directory: Path
) -> dict[str, str] | None:
    if mapping is None:
        return None
    observed = mapping.select("ticker", "cik").unique().sort(["ticker", "cik"])
    path = run_directory / "sec_cumulative_issuer_map.parquet"
    append_run_delta(path, observed)
    require_consistent_sec_issuers(pl.read_parquet(parent.manifest.parent / GENERAL), observed)
    return {"path": str(path), "sha256": file_sha256(path)}


def drop_legacy_sec_partitions(
    *,
    parent: SecCumulativeParent | None,
    paths: OpenSourceLivePaths,
    source_policy: SourceRefreshPolicy,
    mapped_tickers: set[str],
    run_failures: dict[str, list[dict[str, str]]],
    companyfacts_years: Sequence[int],
    submissions_years: Sequence[int],
) -> None:
    """Preserve the compatibility behavior when no cumulative parent was selected."""
    if parent is not None:
        return
    if source_policy.refresh_sec_companyfacts:
        successful = _successful_tickers(mapped_tickers, run_failures["sec_companyfacts"])
        _drop_sec_window(paths.raw_dir / ACTUALS, "period_end", successful, companyfacts_years)
        _drop_sec_window(paths.raw_dir / COMPANYFACTS, "date", successful, companyfacts_years)
    if source_policy.refresh_sec_submissions:
        failures = [
            item
            for item in run_failures["sec_filing"]
            if item.get("dataset") == "earnings_sec_calendar"
        ]
        successful = _successful_tickers(mapped_tickers, failures)
        _drop_sec_window(paths.raw_dir / CALENDAR, "period_end", successful, submissions_years)


def _successful_tickers(
    mapped_tickers: set[str], failures: list[dict[str, str]]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            mapped_tickers
            - {
                str(item["ticker"]).upper().removesuffix(".US")
                for item in failures
                if item.get("ticker")
            }
        )
    )


def _drop_sec_window(
    path: Path, time_column: str, tickers: tuple[str, ...], years: Sequence[int]
) -> None:
    _drop_refreshed_partitions(
        path,
        tickers=tickers,
        date_column=time_column,
        start_date=f"{min(years):04d}-01-01",
        end_date=f"{max(years):04d}-12-31",
    )


def prepare_sec_raw_frames(
    *,
    parent: SecCumulativeParent | None,
    paths: OpenSourceLivePaths,
    run_id: str,
    companyfacts_deltas: Sequence[pl.DataFrame],
    filing_deltas: Sequence[pl.DataFrame],
    calendar: pl.DataFrame,
    actuals: pl.DataFrame,
    issuer_mapping: pl.DataFrame,
) -> SecRawFrames:
    if parent is not None:
        return prepare_cumulative_sec_frames(
            parent=parent,
            run_directory=paths.run_dir(run_id),
            companyfacts=_concat_or_empty(companyfacts_deltas),
            filing=_concat_or_empty(filing_deltas),
            calendar=calendar,
            actuals=actuals,
            issuer_mapping=issuer_mapping,
        )
    stored_calendar = upsert_parquet(
        paths.raw_dir / CALENDAR,
        calendar,
        key_cols=SEC_RAW_DELTA_KEYS[CALENDAR],
        order_cols=("ingested_at",),
    )
    stored_actuals = upsert_parquet(
        paths.raw_dir / ACTUALS,
        actuals,
        key_cols=SEC_RAW_DELTA_KEYS[ACTUALS],
        order_cols=("ingested_at",),
    )
    return SecRawFrames(
        companyfacts=_upsert_financial_dataset(
            paths=paths, run_id=run_id, file_name=COMPANYFACTS, deltas=companyfacts_deltas
        ),
        filing=_upsert_financial_dataset(
            paths=paths, run_id=run_id, file_name=FILING, deltas=filing_deltas
        ),
        calendar=stored_calendar,
        actuals=stored_actuals,
    )
