"""Candidate-only reconstruction of historical SEC fundamentals."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import polars as pl
import requests

from alpharank.data.open_source.sec_mapping import (
    load_sec_historical_ticker_bridge_file,
)

SEC_PROVIDER_ERRORS = (
    requests.RequestException,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    AttributeError,
    ET.ParseError,
    pl.exceptions.PolarsError,
)


class SecCompanyFactsProtocol(Protocol):
    """External Companyfacts operations needed by the reconstruction."""

    def fetch_company_facts(self, cik: str) -> dict[str, Any]: ...

    def extract_financials(self, ticker: str, cik: str) -> pl.DataFrame: ...


class SecFilingFactsProtocol(Protocol):
    """External filing-level operations needed by the reconstruction."""

    def fetch_company_submissions(self, cik: str) -> dict[str, Any]: ...

    def extract_financials(self, ticker: str, cik: str, year: int) -> pl.DataFrame: ...


@dataclass(frozen=True, slots=True)
class HistoricalSecReconstructionConfig:
    """Explicit inputs for a non-promotable historical SEC candidate."""

    output_dir: Path
    bridge_path: Path
    retrieved_at: datetime
    tickers: tuple[str, ...] | None = None
    workers: int = 2


@dataclass(frozen=True, slots=True)
class HistoricalSecFetchResult:
    """Financial observations and audit status for one historical identity."""

    financials: pl.DataFrame
    audit: dict[str, object]


def reconstruct_historical_sec_companyfacts(
    config: HistoricalSecReconstructionConfig,
    *,
    companyfacts_client: SecCompanyFactsProtocol,
    filing_client: SecFilingFactsProtocol,
) -> Path:
    """Build a hash-addressed diagnostic candidate without promoting it."""

    bridge = _select_bridge_rows(config)
    results, failures = _fetch_all(
        bridge,
        companyfacts_client=companyfacts_client,
        filing_client=filing_client,
        workers=config.workers,
        current_year=config.retrieved_at.year,
    )
    financials = _combine_financials(results)
    audits = [result.audit for result in results]
    audit = pl.DataFrame(audits).sort("ticker") if audits else pl.DataFrame()
    return _write_candidate(
        config=config,
        bridge=bridge,
        financials=financials,
        audit=audit,
        failures=failures,
    )


def identity_name_similarity(bridge_name: str, official_names: Sequence[str]) -> float:
    """Return the maximum Jaccard similarity across official SEC names."""

    expected = _name_tokens(bridge_name)
    if not expected:
        return 0.0
    scores = []
    for name in official_names:
        observed = _name_tokens(name)
        if observed:
            scores.append(len(expected & observed) / len(expected | observed))
    return max(scores, default=0.0)


def _select_bridge_rows(config: HistoricalSecReconstructionConfig) -> pl.DataFrame:
    bridge = load_sec_historical_ticker_bridge_file(config.bridge_path)
    if config.tickers:
        requested = sorted({ticker.upper().removesuffix(".US") for ticker in config.tickers})
        bridge = bridge.filter(pl.col("ticker").is_in(requested))
    if bridge.is_empty():
        raise ValueError("No historical SEC bridge rows match the request.")
    if config.workers < 1:
        raise ValueError("workers must be at least 1")
    return bridge.sort("ticker")


def _fetch_all(
    bridge: pl.DataFrame,
    *,
    companyfacts_client: SecCompanyFactsProtocol,
    filing_client: SecFilingFactsProtocol,
    workers: int,
    current_year: int,
) -> tuple[list[HistoricalSecFetchResult], list[dict[str, object]]]:
    results: list[HistoricalSecFetchResult] = []
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_identity,
                row,
                companyfacts_client=companyfacts_client,
                filing_client=filing_client,
                current_year=current_year,
            ): row
            for row in bridge.iter_rows(named=True)
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except SEC_PROVIDER_ERRORS as error:
                failures.append({**row, "status": "failed", "error": str(error)})
    return results, sorted(failures, key=lambda row: str(row["ticker"]))


def _fetch_identity(
    row: Mapping[str, object],
    *,
    companyfacts_client: SecCompanyFactsProtocol,
    filing_client: SecFilingFactsProtocol,
    current_year: int,
) -> HistoricalSecFetchResult:
    ticker = str(row["ticker"])
    cik = str(row["cik"])
    submission_payload = filing_client.fetch_company_submissions(cik)
    financials, facts_payload, source_mode, errors = _fetch_financials(
        row,
        ticker=ticker,
        cik=cik,
        companyfacts_client=companyfacts_client,
        filing_client=filing_client,
        current_year=current_year,
    )
    official_names = _official_names(facts_payload, submission_payload)
    similarity = identity_name_similarity(str(row["name"]), official_names)
    in_window = _filter_identity_window(financials, row)
    audit = _build_audit(
        row,
        facts_payload=facts_payload,
        submission_payload=submission_payload,
        official_names=official_names,
        similarity=similarity,
        financials=financials,
        in_window=in_window,
        source_mode=source_mode,
        errors=errors,
    )
    return HistoricalSecFetchResult(financials=in_window, audit=audit)


def _fetch_financials(
    row: Mapping[str, object],
    *,
    ticker: str,
    cik: str,
    companyfacts_client: SecCompanyFactsProtocol,
    filing_client: SecFilingFactsProtocol,
    current_year: int,
) -> tuple[pl.DataFrame, dict[str, Any], str, list[str]]:
    try:
        payload = companyfacts_client.fetch_company_facts(cik)
        return companyfacts_client.extract_financials(ticker, cik), payload, "sec_companyfacts", []
    except SEC_PROVIDER_ERRORS as error:
        errors = [f"companyfacts: {error}"]

    frames: list[pl.DataFrame] = []
    for year in _identity_years(row, current_year=current_year):
        try:
            frame = filing_client.extract_financials(ticker, cik, year)
        except SEC_PROVIDER_ERRORS as error:
            errors.append(f"filing_{year}: {error}")
            continue
        if not frame.is_empty():
            frames.append(frame)
    return _concat_frames(frames), {}, "sec_filing_fallback", errors


def _build_audit(
    row: Mapping[str, object],
    *,
    facts_payload: Mapping[str, Any],
    submission_payload: Mapping[str, Any],
    official_names: Sequence[str],
    similarity: float,
    financials: pl.DataFrame,
    in_window: pl.DataFrame,
    source_mode: str,
    errors: Sequence[str],
) -> dict[str, object]:
    return {
        **row,
        "sec_entity_name": facts_payload.get("entityName"),
        "sec_submission_name": submission_payload.get("name"),
        "sec_former_names": " | ".join(official_names[2:]),
        "identity_similarity": similarity,
        "identity_status": "name_match" if similarity >= 0.5 else "manual_review_required",
        "financial_rows": financials.height,
        "financial_rows_in_window": in_window.height,
        "metrics_in_window": _column_unique_count(in_window, "metric"),
        "period_end_min_in_window": _column_extreme(in_window, "date", "min"),
        "period_end_max_in_window": _column_extreme(in_window, "date", "max"),
        "source_mode": source_mode,
        "status": "fetched" if in_window.height else "no_financial_rows",
        "errors": " | ".join(errors) or None,
    }


def _official_names(
    facts_payload: Mapping[str, Any],
    submission_payload: Mapping[str, Any],
) -> list[str]:
    former_names = submission_payload.get("formerNames", [])
    values = [
        facts_payload.get("entityName"),
        submission_payload.get("name"),
        *[former.get("name") for former in former_names if isinstance(former, Mapping)],
    ]
    return [str(value) for value in values if value]


def _identity_years(row: Mapping[str, object], *, current_year: int) -> range:
    start_year = max(2009, int(str(row.get("start_date") or "2009")[:4]))
    end_year = int(str(row.get("end_date") or current_year)[:4])
    return range(start_year, end_year + 1)


def _filter_identity_window(
    financials: pl.DataFrame,
    row: Mapping[str, object],
) -> pl.DataFrame:
    if "date" not in financials.columns:
        return financials
    filtered = financials
    if row.get("start_date"):
        filtered = filtered.filter(pl.col("date") >= str(row["start_date"]))
    if row.get("end_date"):
        filtered = filtered.filter(pl.col("date") <= str(row["end_date"]))
    return filtered


def _write_candidate(
    *,
    config: HistoricalSecReconstructionConfig,
    bridge: pl.DataFrame,
    financials: pl.DataFrame,
    audit: pl.DataFrame,
    failures: Sequence[Mapping[str, object]],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _write_artifacts(config.output_dir, financials, audit, failures)
    manifest = _build_manifest(
        config=config,
        bridge=bridge,
        financials=financials,
        audit=audit,
        failures=failures,
        artifact_paths=artifact_paths,
    )
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_artifacts(
    output_dir: Path,
    financials: pl.DataFrame,
    audit: pl.DataFrame,
    failures: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    paths = {
        "financials": output_dir / "financials_sec_companyfacts.parquet",
        "audit": output_dir / "historical_sec_mapping_audit.csv",
        "failures": output_dir / "failures.json",
    }
    financials.write_parquet(paths["financials"])
    audit.write_csv(paths["audit"])
    paths["failures"].write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return paths


def _build_manifest(
    *,
    config: HistoricalSecReconstructionConfig,
    bridge: pl.DataFrame,
    financials: pl.DataFrame,
    audit: pl.DataFrame,
    failures: Sequence[Mapping[str, object]],
    artifact_paths: Mapping[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "historical_sec_companyfacts_reconstruction_candidate",
        "retrieved_at": config.retrieved_at.astimezone().isoformat(),
        "source": "SEC Companyfacts with explicit SEC filing-level fallback",
        "bridge": {
            "path": config.bridge_path.name,
            "sha256": _hash(config.bridge_path),
            "rows": bridge.height,
        },
        "requested_mappings": bridge.height,
        "fetched_mappings": audit.height,
        "failed_mappings": len(failures),
        "mappings_with_financial_rows": _count_matching(
            audit, pl.col("financial_rows_in_window") > 0
        ),
        "identity_name_matches": _count_matching(audit, pl.col("identity_status") == "name_match"),
        "identity_manual_reviews": _count_matching(
            audit, pl.col("identity_status") == "manual_review_required"
        ),
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _hash(path),
                "rows": _artifact_rows(name, financials, audit, failures),
            }
            for name, path in artifact_paths.items()
        },
        "promotion_status": "blocked_pending_identity_review_and_package_revision_gate",
    }


def _name_tokens(value: str) -> set[str]:
    ignored = {
        "co",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "ltd",
        "plc",
        "the",
        "group",
    }
    return {
        token
        for token in re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
        if len(token) > 1 and token not in ignored
    }


def _concat_frames(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(frames, how="diagonal_relaxed") if frames else _empty_financials()


def _combine_financials(results: Sequence[HistoricalSecFetchResult]) -> pl.DataFrame:
    frames = [result.financials for result in results if not result.financials.is_empty()]
    combined = _concat_frames(frames)
    sort_columns = [
        column
        for column in ("ticker", "statement", "metric", "date", "filing_date")
        if column in combined.columns
    ]
    return combined.sort(sort_columns) if sort_columns else combined


def _empty_financials() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )


def _column_unique_count(frame: pl.DataFrame, column: str) -> int:
    return frame[column].n_unique() if column in frame.columns and not frame.is_empty() else 0


def _column_extreme(frame: pl.DataFrame, column: str, operation: str) -> object:
    if column not in frame.columns or frame.is_empty():
        return None
    return frame[column].min() if operation == "min" else frame[column].max()


def _count_matching(frame: pl.DataFrame, predicate: pl.Expr) -> int:
    return frame.filter(predicate).height if not frame.is_empty() else 0


def _artifact_rows(
    name: str,
    financials: pl.DataFrame,
    audit: pl.DataFrame,
    failures: Sequence[Mapping[str, object]],
) -> int:
    return {"financials": financials.height, "audit": audit.height, "failures": len(failures)}[name]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
