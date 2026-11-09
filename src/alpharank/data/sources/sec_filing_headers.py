"""Acquire and parse immutable SEC filing headers containing historical SICs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests

from alpharank.data.ingestion.raw_archive import (
    RawDownloadAttempt,
    record_raw_download_batch,
)

ANNUAL_FORMS = frozenset({"10-K", "10-KT", "20-F", "40-F"})
INGESTER_VERSION = "sec_filing_header_sic_v1"
NEW_YORK = ZoneInfo("America/New_York")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FilingHeaderReference:
    """One annual SEC accession whose submission header can contain a SIC."""

    cik: str
    accession_number: str
    filing_date: str
    form: str


@dataclass(frozen=True, slots=True)
class SecFilingHeaderConfig:
    """Bounded acquisition configuration for a reproducible historical run."""

    project_root: Path
    submissions_archive: Path
    headers_archive: Path
    user_agent: str
    run_id: str
    start_year: int
    end_year: int
    request_pause_seconds: float = 0.12
    timeout_seconds: int = 30
    max_retries: int = 5
    header_workers: int = 12


@dataclass(frozen=True, slots=True)
class SecFilingHeaderAcquisition:
    """Parsed issuer observations, explicit failures and RAW manifest paths."""

    observations: pl.DataFrame
    failures: pl.DataFrame
    submissions_manifest_path: Path
    headers_manifest_path: Path


@dataclass(frozen=True, slots=True)
class _CachedPayload:
    payload: bytes
    retrieved_at: str
    payload_sha256: str
    object_path: Path


class _RawRecorder:
    def __init__(
        self,
        *,
        archive_dir: Path,
        source_name: str,
        dataset_name: str,
    ) -> None:
        self.archive_dir = archive_dir.resolve()
        self.source_name = source_name
        self.dataset_name = dataset_name
        self.pending: list[RawDownloadAttempt] = []
        self._lock = threading.Lock()

    def add(self, attempt: RawDownloadAttempt) -> None:
        with self._lock:
            self.pending.append(attempt)

    def flush(self) -> None:
        if not self.pending:
            return
        record_raw_download_batch(
            archive_dir=self.archive_dir,
            source_name=self.source_name,
            dataset_name=self.dataset_name,
            ingester_version=INGESTER_VERSION,
            attempts=tuple(self.pending),
        )
        self.pending.clear()


class _RateLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self.interval_seconds
        if delay:
            time.sleep(delay)


class SecFilingHeaderClient:
    """SEC client respecting the public fair-access request rate."""

    def __init__(self, config: SecFilingHeaderConfig) -> None:
        self.config = config
        self._local = threading.local()
        self._rate_limiter = _RateLimiter(config.request_pause_seconds)

    def get(self, url: str) -> tuple[int, bytes | None, str, str | None]:
        last_error: requests.RequestException | None = None
        for attempt_index in range(self.config.max_retries):
            try:
                self._rate_limiter.wait()
                response = self._session().get(url, timeout=self.config.timeout_seconds)
                retrieved_at = datetime.now(timezone.utc).isoformat()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(30.0, 2**attempt_index)
                    time.sleep(delay)
                    continue
                if not 200 <= response.status_code <= 299:
                    return response.status_code, None, retrieved_at, response.reason
                return response.status_code, response.content, retrieved_at, None
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(30.0, 2**attempt_index))
        retrieved_at = datetime.now(timezone.utc).isoformat()
        return 599, None, retrieved_at, str(last_error or "SEC request failed")

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": self.config.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                }
            )
            self._local.session = session
        return session


def acquire_sec_filing_headers(
    ciks: tuple[str, ...],
    *,
    config: SecFilingHeaderConfig,
) -> SecFilingHeaderAcquisition:
    """Fetch annual filing headers and preserve every provider response in RAW."""

    if config.start_year > config.end_year:
        raise ValueError("SEC filing header start_year cannot exceed end_year.")
    client = SecFilingHeaderClient(config)
    submissions_cache = _load_raw_payload_cache(config.submissions_archive)
    headers_cache = _load_raw_payload_cache(config.headers_archive)
    submission_recorder = _RawRecorder(
        archive_dir=config.submissions_archive,
        source_name="sec_submissions",
        dataset_name="issuer_submissions",
    )
    header_recorder = _RawRecorder(
        archive_dir=config.headers_archive,
        source_name="sec_filings",
        dataset_name="filing_submission_headers",
    )
    observations: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    unique_ciks = tuple(sorted(set(ciks)))
    with ThreadPoolExecutor(max_workers=config.header_workers) as executor:
        for start in range(0, len(unique_ciks), 25):
            batch = unique_ciks[start : start + 25]
            futures = []
            for cik in batch:
                references = _annual_references(
                    cik,
                    client=client,
                    cache=submissions_cache,
                    recorder=submission_recorder,
                    failures=failures,
                )
                futures.extend(
                    executor.submit(
                        _fetch_header_observation,
                        reference,
                        client=client,
                        cache=headers_cache,
                        recorder=header_recorder,
                        config=config,
                    )
                    for reference in references
                )
            for future in as_completed(futures):
                observation, failure = future.result()
                if observation is not None:
                    observations.append(observation)
                if failure is not None:
                    failures.append(failure)
            submission_recorder.flush()
            header_recorder.flush()
            LOGGER.info(
                "sec_sector_history_progress issuers=%s/%s observations=%s failures=%s",
                start + len(batch),
                len(unique_ciks),
                len(observations),
                len(failures),
            )
    submission_recorder.flush()
    header_recorder.flush()
    return SecFilingHeaderAcquisition(
        observations=_observation_frame(observations),
        failures=_failure_frame(failures),
        submissions_manifest_path=config.submissions_archive / "manifests" / "latest.json",
        headers_manifest_path=config.headers_archive / "manifests" / "latest.json",
    )


def parse_sec_filing_header(
    payload: bytes,
    *,
    expected_cik: str,
) -> dict[str, str] | None:
    """Extract the matching filer's SIC and causal acceptance timestamp."""

    text = payload.decode("latin-1", errors="replace")
    acceptance_match = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", text)
    if acceptance_match is None:
        return None
    accepted_at = _acceptance_datetime_utc(acceptance_match.group(1))
    filer_sections = re.split(
        r"(?:^|\n)(?:FILER:\s*|<FILER>)",
        text,
        flags=re.MULTILINE,
    )[1:]
    normalized_expected = str(expected_cik).zfill(10)
    for section in filer_sections:
        parsed_filer = _parse_filer_section(section)
        if parsed_filer is None:
            continue
        filer_cik, sic, description = parsed_filer
        if filer_cik != normalized_expected:
            continue
        return {
            "cik": normalized_expected,
            "sic": sic,
            "sic_description": description,
            "available_at": accepted_at.isoformat(),
        }
    return None


def _parse_filer_section(section: str) -> tuple[str, str, str] | None:
    cik_match = re.search(r"(?:CENTRAL INDEX KEY:\s*|<CIK>)(\d+)", section)
    descriptive_sic = re.search(
        r"STANDARD INDUSTRIAL CLASSIFICATION:\s*(.*?)\s*\[(\d{3,4})\]",
        section,
    )
    tagged_sic = re.search(r"<ASSIGNED-SIC>(\d{3,4})", section)
    if cik_match is None or (descriptive_sic is None and tagged_sic is None):
        return None
    cik = cik_match.group(1).zfill(10)
    if descriptive_sic is not None:
        sic = descriptive_sic.group(2).zfill(4)
        description = " ".join(descriptive_sic.group(1).split())
        return cik, sic, description
    assert tagged_sic is not None
    sic = tagged_sic.group(1).zfill(4)
    return cik, sic, f"SEC SIC {sic}"


def _annual_references(
    cik: str,
    *,
    client: SecFilingHeaderClient,
    cache: dict[str, _CachedPayload],
    recorder: _RawRecorder,
    failures: list[dict[str, object]],
) -> tuple[FilingHeaderReference, ...]:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    current = _fetch_raw_payload(
        url,
        client=client,
        cache=cache,
        recorder=recorder,
        payload_format="json",
        requested_scope={"cik": cik, "request_url": url, "kind": "current"},
        use_cache=False,
    )
    if current is None:
        failures.append(_failure(cik=cik, stage="submissions", source_url=url))
        return ()
    blocks = [_decode_json(current.payload, url=url)]
    for older in blocks[0].get("filings", {}).get("files", []):
        name = older.get("name")
        if not name:
            continue
        older_url = f"https://data.sec.gov/submissions/{name}"
        cached = _fetch_raw_payload(
            older_url,
            client=client,
            cache=cache,
            recorder=recorder,
            payload_format="json",
            requested_scope={"cik": cik, "request_url": older_url, "kind": "historical"},
            use_cache=True,
        )
        if cached is None:
            failures.append(_failure(cik=cik, stage="submissions_history", source_url=older_url))
            continue
        blocks.append(_decode_json(cached.payload, url=older_url))
    references: dict[str, FilingHeaderReference] = {}
    for block_index, block in enumerate(blocks):
        filings = block.get("filings", {}).get("recent", {}) if block_index == 0 else block
        for reference in _references_from_block(cik, filings, client.config):
            references[reference.accession_number] = reference
    return tuple(sorted(references.values(), key=lambda value: value.accession_number))


def _references_from_block(
    cik: str,
    block: dict[str, object],
    config: SecFilingHeaderConfig,
) -> tuple[FilingHeaderReference, ...]:
    forms = block.get("form", [])
    accessions = block.get("accessionNumber", [])
    filing_dates = block.get("filingDate", [])
    if not isinstance(forms, list) or not isinstance(accessions, list):
        return ()
    references: list[FilingHeaderReference] = []
    for form, accession, filing_date in zip(forms, accessions, filing_dates, strict=False):
        filing_year = _year(str(filing_date))
        if str(form) not in ANNUAL_FORMS or filing_year is None:
            continue
        if not config.start_year <= filing_year <= config.end_year:
            continue
        references.append(
            FilingHeaderReference(
                cik=cik.zfill(10),
                accession_number=str(accession),
                filing_date=str(filing_date),
                form=str(form),
            )
        )
    return tuple(references)


def _fetch_header_observation(
    reference: FilingHeaderReference,
    *,
    client: SecFilingHeaderClient,
    cache: dict[str, _CachedPayload],
    recorder: _RawRecorder,
    config: SecFilingHeaderConfig,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    url = _header_url(reference)
    raw = _fetch_raw_payload(
        url,
        client=client,
        cache=cache,
        recorder=recorder,
        payload_format="sgml",
        requested_scope={
            "cik": reference.cik,
            "accession_number": reference.accession_number,
            "form": reference.form,
            "request_url": url,
        },
        use_cache=True,
    )
    if raw is None:
        return None, _failure(cik=reference.cik, stage="filing_header", source_url=url)
    parsed = parse_sec_filing_header(raw.payload, expected_cik=reference.cik)
    if parsed is None:
        return None, _failure(
            cik=reference.cik,
            stage="header_missing_matching_sic",
            source_url=url,
            accession_number=reference.accession_number,
        )
    object_path = raw.object_path
    try:
        relative_path = object_path.relative_to(config.project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("SEC RAW archives must live under the project root.") from exc
    return (
        {
            **parsed,
            "accession_number": reference.accession_number,
            "filing_date": reference.filing_date,
            "form": reference.form,
            "source_url": url,
            "retrieved_at": raw.retrieved_at,
            "raw_observation_path": relative_path,
            "raw_observation_sha256": raw.payload_sha256,
        },
        None,
    )


def _fetch_raw_payload(
    url: str,
    *,
    client: SecFilingHeaderClient,
    cache: dict[str, _CachedPayload],
    recorder: _RawRecorder,
    payload_format: str,
    requested_scope: dict[str, object],
    use_cache: bool,
) -> _CachedPayload | None:
    if use_cache and url in cache:
        return cache[url]
    status, payload, retrieved_at, error = client.get(url)
    recorder.add(
        RawDownloadAttempt(
            receipt_id=_receipt_id(client.config.run_id, url),
            request_id=url,
            retrieved_at=retrieved_at,
            response_status=status,
            payload=payload,
            payload_format=payload_format,
            requested_scope=requested_scope,
            error=error,
        )
    )
    if payload is None:
        return None
    digest = hashlib.sha256(payload).hexdigest()
    result = _CachedPayload(
        payload=payload,
        retrieved_at=retrieved_at,
        payload_sha256=digest,
        object_path=recorder.archive_dir / "objects" / digest[:2] / digest,
    )
    cache[url] = result
    return result


def _load_raw_payload_cache(archive_dir: Path) -> dict[str, _CachedPayload]:
    archive_dir = archive_dir.resolve()
    cached: dict[str, _CachedPayload] = {}
    for receipt_path in sorted((archive_dir / "receipts").glob("*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        scope = receipt.get("requested_scope", {})
        url = scope.get("request_url") if isinstance(scope, dict) else None
        object_relative = receipt.get("payload_object_path")
        if not url or not object_relative or not 200 <= int(receipt["response_status"]) <= 299:
            continue
        object_path = archive_dir / str(object_relative)
        payload = object_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != receipt.get("payload_sha256"):
            raise RuntimeError(f"SEC RAW payload hash mismatch: {object_path}")
        candidate = _CachedPayload(
            payload=payload,
            retrieved_at=str(receipt["retrieved_at"]),
            payload_sha256=digest,
            object_path=object_path,
        )
        previous = cached.get(str(url))
        if previous is None or candidate.retrieved_at > previous.retrieved_at:
            cached[str(url)] = candidate
    return cached


def _header_url(reference: FilingHeaderReference) -> str:
    accession_without_dashes = reference.accession_number.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(reference.cik)}/{accession_without_dashes}/"
        f"{reference.accession_number}.hdr.sgml"
    )


def _decode_json(payload: bytes, *, url: str) -> dict[str, object]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SEC submissions JSON: {url}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Unexpected SEC submissions payload: {url}")
    return decoded


def _acceptance_datetime_utc(value: str) -> datetime:
    local = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=NEW_YORK)
    return local.astimezone(timezone.utc)


def _receipt_id(run_id: str, url: str) -> str:
    suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return f"{run_id}_{suffix}".lower().replace("-", "_")


def _year(value: str) -> int | None:
    try:
        return int(value[:4])
    except (TypeError, ValueError):
        return None


def _failure(
    *,
    cik: str,
    stage: str,
    source_url: str,
    accession_number: str | None = None,
) -> dict[str, object]:
    return {
        "cik": cik.zfill(10),
        "accession_number": accession_number,
        "stage": stage,
        "source_url": source_url,
    }


def _observation_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "cik": pl.String,
                "accession_number": pl.String,
                "filing_date": pl.String,
                "form": pl.String,
                "sic": pl.String,
                "sic_description": pl.String,
                "available_at": pl.String,
                "source_url": pl.String,
                "retrieved_at": pl.String,
                "raw_observation_path": pl.String,
                "raw_observation_sha256": pl.String,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["cik", "available_at", "accession_number"]
    )


def _failure_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "cik": pl.String,
                "accession_number": pl.String,
                "stage": pl.String,
                "source_url": pl.String,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["cik", "stage", "source_url", "accession_number"],
        nulls_last=True,
    )
