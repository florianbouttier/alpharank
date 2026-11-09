"""Transform SEC filing-header SIC observations into sector history inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True, slots=True)
class SicSectorRule:
    """One inclusive SIC interval in declared precedence order."""

    rule_id: str
    min_sic: int
    max_sic: int
    sector: str


@dataclass(frozen=True, slots=True)
class SicSectorTaxonomy:
    """Reviewed deterministic mapping from filing SIC to AlphaRank sector."""

    taxonomy_id: str
    allowed_sectors: frozenset[str]
    rules: tuple[SicSectorRule, ...]


def load_sic_sector_taxonomy(path: Path) -> SicSectorTaxonomy:
    """Load and validate the versioned SIC-sector mapping policy."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = frozenset(str(value) for value in payload["allowed_sectors"])
    rules = tuple(
        SicSectorRule(
            rule_id=str(row["rule_id"]),
            min_sic=int(row["min_sic"]),
            max_sic=int(row["max_sic"]),
            sector=str(row["sector"]),
        )
        for row in payload["rules"]
    )
    _validate_taxonomy(allowed, rules)
    return SicSectorTaxonomy(
        taxonomy_id=str(payload["taxonomy_id"]),
        allowed_sectors=allowed,
        rules=rules,
    )


def build_sec_sic_sector_observations(
    headers: pl.DataFrame,
    identities: pl.DataFrame,
    *,
    taxonomy: SicSectorTaxonomy,
    ingestion_run_id: str,
) -> pl.DataFrame:
    """Attach issuer SIC observations to time-bounded ticker identities."""

    _require_columns(
        headers,
        {
            "cik",
            "sic",
            "sic_description",
            "available_at",
            "retrieved_at",
            "accession_number",
            "filing_date",
            "form",
            "source_url",
            "raw_observation_path",
            "raw_observation_sha256",
        },
        label="SEC filing headers",
    )
    _require_columns(
        identities,
        {
            "ticker",
            "cik",
            "identity_id",
            "identity_valid_from",
            "identity_valid_to",
            "identity_source",
            "identity_status",
            "identity_evidence",
        },
        label="sector identity map",
    )
    mapping = _sic_mapping_frame(headers, taxonomy=taxonomy)
    return (
        headers.with_columns(
            pl.col("cik").cast(pl.String).str.zfill(10),
            pl.col("sic").cast(pl.String).str.zfill(4),
        )
        .join(identities, on="cik", how="inner", validate="m:m")
        .join(mapping, on="sic", how="left", validate="m:1")
        .with_columns(
            pl.col("sector").alias("Sector"),
            pl.lit("sec_filing_header_sic").alias("sector_source"),
            pl.col("sic_description").alias("sector_raw_value"),
            pl.concat_str(
                pl.lit(taxonomy.taxonomy_id),
                pl.lit(":"),
                pl.col("rule_id"),
            ).alias("mapping_rule"),
            pl.lit(ingestion_run_id).alias("ingestion_run_id"),
            pl.col("retrieved_at").alias("ingested_at"),
            pl.col("available_at").alias("source_available_at"),
        )
        .drop("sector", "rule_id")
        .sort(["ticker", "source_available_at", "accession_number"])
    )


def map_sic_to_sector(sic: str | int, taxonomy: SicSectorTaxonomy) -> tuple[str, str] | None:
    """Return the first declared sector rule matching one numeric SIC."""

    try:
        numeric = int(sic)
    except (TypeError, ValueError):
        return None
    for rule in taxonomy.rules:
        if rule.min_sic <= numeric <= rule.max_sic:
            return rule.sector, rule.rule_id
    return None


def _sic_mapping_frame(
    headers: pl.DataFrame,
    *,
    taxonomy: SicSectorTaxonomy,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for sic in sorted(set(headers["sic"].drop_nulls().cast(pl.String).to_list())):
        match = map_sic_to_sector(sic, taxonomy)
        rows.append(
            {
                "sic": str(sic).zfill(4),
                "sector": match[0] if match is not None else None,
                "rule_id": match[1] if match is not None else "unmapped_sic",
            }
        )
    if not rows:
        return pl.DataFrame(schema={"sic": pl.String, "sector": pl.String, "rule_id": pl.String})
    return pl.from_dicts(rows, infer_schema_length=None)


def _validate_taxonomy(
    allowed: frozenset[str],
    rules: tuple[SicSectorRule, ...],
) -> None:
    if not allowed or not rules:
        raise ValueError("SIC sector taxonomy must declare sectors and rules.")
    rule_ids = [rule.rule_id for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("SIC sector taxonomy contains duplicate rule ids.")
    for rule in rules:
        if rule.min_sic > rule.max_sic:
            raise ValueError(f"SIC sector rule has an inverted interval: {rule.rule_id}")
        if rule.sector not in allowed:
            raise ValueError(f"SIC sector rule uses an undeclared sector: {rule.rule_id}")


def _require_columns(frame: pl.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing: " + ", ".join(missing))
