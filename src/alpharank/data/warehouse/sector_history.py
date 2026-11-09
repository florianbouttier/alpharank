"""Build immutable point-in-time sector-history releases from RAW observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

SECTOR_HISTORY_CONTRACT_ID = "point_in_time_sector_history_v1"
SECTOR_HISTORY_CONTRACT_IDS = frozenset(
    {SECTOR_HISTORY_CONTRACT_ID, "point_in_time_sector_history_v2"}
)
SECTOR_SELECTION_POLICY_ID = "sector_observation_as_known_v1"
UNKNOWN_SECTOR_VALUES = frozenset({"", "unknown", "n/a", "none", "null"})
REQUIRED_OBSERVATION_COLUMNS = (
    "ticker",
    "Sector",
    "sector_source",
    "sector_raw_value",
    "sic",
    "sic_description",
    "mapping_rule",
    "cik",
    "ingestion_run_id",
    "ingested_at",
    "raw_observation_path",
    "raw_observation_sha256",
)
OPTIONAL_OBSERVATION_COLUMNS = (
    "source_available_at",
    "source_url",
    "accession_number",
    "filing_date",
    "form",
    "identity_id",
    "identity_valid_from",
    "identity_valid_to",
    "identity_source",
    "identity_status",
    "identity_evidence",
)


@dataclass(frozen=True, slots=True)
class SectorHistoryBuild:
    """Accepted history, retained rejections and deterministic quality summary."""

    history: pl.DataFrame
    rejections: pl.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class SectorHistoryRelease:
    """One immutable content-addressed DEF release."""

    release_dir: Path
    history_path: Path
    rejections_path: Path
    manifest_path: Path
    release_id: str
    manifest: dict[str, object]


def build_sector_history(
    observations: pl.DataFrame,
    *,
    allowed_sectors: Sequence[str],
    aliases: Mapping[str, str],
    selection_policy_id: str = SECTOR_SELECTION_POLICY_ID,
) -> SectorHistoryBuild:
    """Normalize provider observations without making them known retroactively."""

    missing = sorted(set(REQUIRED_OBSERVATION_COLUMNS) - set(observations.columns))
    if missing:
        raise ValueError("Sector observations are missing: " + ", ".join(missing))
    allowed = frozenset(str(value) for value in allowed_sectors)
    normalized = _normalize_observations(observations, aliases=aliases)
    _require_unique_observation_keys(normalized)
    accepted = normalized.filter(
        pl.col("observed_at").is_not_null()
        & pl.col("canonical_sector").is_in(sorted(allowed))
        & pl.col("sector_source").is_not_null()
        & ~pl.col("sector_source").str.to_lowercase().is_in(sorted(UNKNOWN_SECTOR_VALUES))
    )
    observation_keys = ["ticker", "observed_at"]
    if "identity_id" in normalized.columns:
        observation_keys.insert(1, "identity_id")
    rejected = normalized.join(
        accepted.select(observation_keys),
        on=observation_keys,
        how="anti",
    ).with_columns(_rejection_reason(allowed).alias("rejection_reason"))
    history = _build_accepted_history(
        accepted,
        selection_policy_id=selection_policy_id,
    )
    summary = _build_summary(normalized, history, rejected)
    return SectorHistoryBuild(
        history=history,
        rejections=rejected.sort(["observed_at", "ticker"], nulls_last=True),
        summary=summary,
    )


def materialize_sector_history_release(
    observations: pl.DataFrame,
    *,
    contract_path: Path,
    output_root: Path,
    generated_at: datetime | None = None,
    evidence_frames: Mapping[str, pl.DataFrame] | None = None,
) -> SectorHistoryRelease:
    """Write or validate one immutable release identified by its logical content."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_id") not in SECTOR_HISTORY_CONTRACT_IDS:
        raise ValueError(f"Unsupported sector-history contract: {contract.get('contract_id')}")
    build = build_sector_history(
        observations,
        allowed_sectors=contract["allowed_sectors"],
        aliases=contract["aliases"],
        selection_policy_id=contract.get("selection_policy_id", SECTOR_SELECTION_POLICY_ID),
    )
    evidence = dict(evidence_frames or {})
    _validate_evidence_names(evidence)
    release_id = (
        _release_id(build.history, evidence) if evidence else _logical_frame_hash(build.history)
    )
    release_dir = output_root / f"sector_history_{release_id[:12]}"
    history_path = release_dir / "sector_history.parquet"
    rejections_path = release_dir / "sector_observation_rejections.parquet"
    manifest_path = release_dir / "manifest.json"
    if release_dir.exists():
        manifest = _read_and_validate_existing_release(
            release_dir,
            expected_release_id=release_id,
        )
        return SectorHistoryRelease(
            release_dir=release_dir,
            history_path=history_path,
            rejections_path=rejections_path,
            manifest_path=manifest_path,
            release_id=release_id,
            manifest=manifest,
        )

    release_dir.mkdir(parents=True, exist_ok=False)
    evidence_paths = _write_release_frames(
        build,
        history_path=history_path,
        rejections_path=rejections_path,
        release_dir=release_dir,
        evidence=evidence,
    )
    manifest: dict[str, object] = {
        "contract_id": contract["contract_id"],
        "schema_version": int(contract["schema_version"]),
        "release_id": release_id,
        "generated_at": _utc_iso(generated_at),
        "selection_policy_id": contract.get("selection_policy_id", SECTOR_SELECTION_POLICY_ID),
        "grain": contract["grain"],
        "business_key": contract["business_key"],
        "time_policy": contract["time_policy"],
        "missing_policy": contract["missing_policy"],
        "quality": build.summary,
        "inputs": _input_inventory(observations),
        "files": {
            history_path.name: _file_record(history_path),
            rejections_path.name: _file_record(rejections_path),
            **{name: _file_record(path) for name, path in sorted(evidence_paths.items())},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SectorHistoryRelease(
        release_dir=release_dir,
        history_path=history_path,
        rejections_path=rejections_path,
        manifest_path=manifest_path,
        release_id=release_id,
        manifest=manifest,
    )


def _write_release_frames(
    build: SectorHistoryBuild,
    *,
    history_path: Path,
    rejections_path: Path,
    release_dir: Path,
    evidence: Mapping[str, pl.DataFrame],
) -> dict[str, Path]:
    build.history.write_parquet(history_path)
    build.rejections.write_parquet(rejections_path)
    evidence_paths: dict[str, Path] = {}
    for name, frame in sorted(evidence.items()):
        path = release_dir / name
        frame.write_parquet(path)
        evidence_paths[name] = path
    return evidence_paths


def _utc_iso(value: datetime | None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


def load_sector_observations(paths: Sequence[Path], *, root: Path) -> pl.DataFrame:
    """Load retained RAW deltas and attach stable file-level lineage."""

    frames: list[pl.DataFrame] = []
    for path in sorted(path.resolve() for path in paths):
        frame = pl.read_parquet(path)
        relative = path.relative_to(root.resolve()).as_posix()
        frames.append(
            frame.with_columns(
                pl.lit(relative).alias("raw_observation_path"),
                pl.lit(_sha256(path)).alias("raw_observation_sha256"),
            )
        )
    if not frames:
        raise ValueError("No sector observation file was found.")
    return pl.concat(frames, how="diagonal_relaxed")


def _normalize_observations(
    observations: pl.DataFrame,
    *,
    aliases: Mapping[str, str],
) -> pl.DataFrame:
    alias_frame = pl.DataFrame(
        {
            "raw_sector_alias": list(aliases),
            "aliased_sector": list(aliases.values()),
        },
        schema={"raw_sector_alias": pl.String, "aliased_sector": pl.String},
    )
    optional = [column for column in OPTIONAL_OBSERVATION_COLUMNS if column in observations.columns]
    selected = observations.select(*REQUIRED_OBSERVATION_COLUMNS, *optional)
    available_at = (
        pl.col("source_available_at")
        if "source_available_at" in selected.columns
        else pl.col("ingested_at")
    )
    return (
        selected.with_columns(
            pl.col("ticker").cast(pl.String).str.strip_chars(),
            pl.col("Sector").cast(pl.String, strict=False).str.strip_chars().alias("raw_sector"),
            pl.col("sector_source").cast(pl.String, strict=False).str.strip_chars(),
            available_at.cast(pl.String, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False)
            .alias("observed_at"),
        )
        .join(alias_frame, left_on="raw_sector", right_on="raw_sector_alias", how="left")
        .with_columns(pl.coalesce("aliased_sector", "raw_sector").alias("canonical_sector"))
        .drop("aliased_sector")
        .sort(["ticker", "observed_at", "raw_observation_path"])
    )


def _require_unique_observation_keys(frame: pl.DataFrame) -> None:
    null_keys = frame.filter(pl.col("ticker").is_null() | pl.col("observed_at").is_null())
    if not null_keys.is_empty():
        return
    keys = ["ticker", "observed_at"]
    if "identity_id" in frame.columns:
        keys.insert(1, "identity_id")
    duplicates = frame.group_by(*keys).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError("Sector observations contain duplicate ticker/observed_at keys.")


def _rejection_reason(allowed_sectors: frozenset[str]) -> pl.Expr:
    return (
        pl.when(pl.col("observed_at").is_null())
        .then(pl.lit("invalid_observed_at"))
        .when(
            pl.col("canonical_sector").is_null()
            | pl.col("canonical_sector").str.to_lowercase().is_in(sorted(UNKNOWN_SECTOR_VALUES))
        )
        .then(pl.lit("sector_not_observed"))
        .when(~pl.col("canonical_sector").is_in(sorted(allowed_sectors)))
        .then(pl.lit("sector_outside_canonical_taxonomy"))
        .otherwise(pl.lit("source_not_eligible"))
    )


def _build_accepted_history(
    accepted: pl.DataFrame,
    *,
    selection_policy_id: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in accepted.to_dicts():
        observed_at = row["observed_at"]
        source = str(row["sector_source"])
        sector = str(row["canonical_sector"])
        classification_id = _classification_id(row, sector=sector)
        rows.append(
            {
                "ticker": row["ticker"],
                "Sector": sector,
                "classification_id": classification_id,
                "source_url": _source_url(row),
                "confidence": _confidence(source),
                "observed_at": observed_at,
                "effective_at": observed_at,
                "available_at": observed_at,
                "sector_source": source,
                "sector_raw_value": row["sector_raw_value"],
                "sic": row["sic"],
                "sic_description": row["sic_description"],
                "mapping_rule": row["mapping_rule"],
                "cik": row["cik"],
                "ingestion_run_id": row["ingestion_run_id"],
                "raw_observation_path": row["raw_observation_path"],
                "raw_observation_sha256": row["raw_observation_sha256"],
                "selection_policy_id": selection_policy_id,
                "status": "observed",
                **_identity_lineage(row),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "Sector": pl.String,
                "classification_id": pl.String,
                "source_url": pl.String,
                "confidence": pl.String,
                "observed_at": pl.Datetime(time_zone="UTC"),
                "effective_at": pl.Datetime(time_zone="UTC"),
                "available_at": pl.Datetime(time_zone="UTC"),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(["ticker", "observed_at"])


def _classification_id(row: Mapping[str, object], *, sector: str) -> str:
    payload = {
        "ticker": row["ticker"],
        "sector": sector,
        "observed_at": str(row["observed_at"]),
        "sector_source": row["sector_source"],
        "sector_raw_value": row["sector_raw_value"],
        "mapping_rule": row["mapping_rule"],
        "raw_observation_sha256": row["raw_observation_sha256"],
        "identity_id": row.get("identity_id"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sector-{digest}"


def _source_url(row: Mapping[str, object]) -> str:
    if row.get("source_url"):
        return str(row["source_url"])
    source = str(row["sector_source"])
    if source == "sec_sic" and row.get("cik"):
        return f"https://data.sec.gov/submissions/CIK{str(row['cik']).zfill(10)}.json"
    ticker = str(row["ticker"]).removesuffix(".US")
    if source == "yfinance":
        return f"https://finance.yahoo.com/quote/{ticker}/profile"
    return f"provider://{source}/{ticker}"


def _confidence(source: str) -> str:
    return "provider_reported" if source == "yfinance" else "derived_from_sec_sic"


def _identity_lineage(row: Mapping[str, object]) -> dict[str, object]:
    return {
        column: row.get(column)
        for column in (
            "accession_number",
            "filing_date",
            "form",
            "identity_id",
            "identity_valid_from",
            "identity_valid_to",
            "identity_source",
            "identity_status",
            "identity_evidence",
        )
        if column in row
    }


def _build_summary(
    observations: pl.DataFrame,
    history: pl.DataFrame,
    rejections: pl.DataFrame,
) -> dict[str, object]:
    accepted_tickers = history["ticker"].n_unique() if history.height else 0
    observed_tickers = observations["ticker"].n_unique() if observations.height else 0
    return {
        "observation_rows": observations.height,
        "accepted_rows": history.height,
        "rejected_rows": rejections.height,
        "observed_tickers": observed_tickers,
        "accepted_tickers": accepted_tickers,
        "ticker_coverage_pct": (
            100.0 * accepted_tickers / observed_tickers if observed_tickers else 0.0
        ),
        "observed_at_min": _iso_or_none(history["observed_at"].min() if history.height else None),
        "observed_at_max": _iso_or_none(history["observed_at"].max() if history.height else None),
        "accepted_by_source": _counts(history, "sector_source"),
        "rejected_by_reason": _counts(rejections, "rejection_reason"),
    }


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).to_dicts()
    }


def _input_inventory(observations: pl.DataFrame) -> list[dict[str, object]]:
    return (
        observations.select("raw_observation_path", "raw_observation_sha256")
        .unique()
        .sort("raw_observation_path")
        .to_dicts()
    )


def _logical_frame_hash(frame: pl.DataFrame) -> str:
    payload = frame.write_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _release_id(
    history: pl.DataFrame,
    evidence_frames: Mapping[str, pl.DataFrame],
) -> str:
    payload = {
        "sector_history": _logical_frame_hash(history),
        "evidence": {
            name: _logical_frame_hash(frame) for name, frame in sorted(evidence_frames.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_evidence_names(evidence_frames: Mapping[str, pl.DataFrame]) -> None:
    for name in evidence_frames:
        path = Path(name)
        if path.name != name or path.suffix != ".parquet":
            raise ValueError(f"Sector release evidence name is unsafe: {name!r}")


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_and_validate_existing_release(
    release_dir: Path,
    *,
    expected_release_id: str,
) -> dict[str, object]:
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_id") != expected_release_id:
        raise RuntimeError("Existing sector-history release has another release id.")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("Existing sector-history release has no file inventory.")
    for name, record in files.items():
        if not isinstance(record, Mapping) or _sha256(release_dir / str(name)) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"Sector-history release hash mismatch: {name}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_or_none(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None
