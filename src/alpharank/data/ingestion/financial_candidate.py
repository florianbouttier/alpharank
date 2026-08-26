"""Build and validate the consolidated financial acquisition candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from alpharank.data.ingestion.frames import _clean_financial_columns
from alpharank.data.ingestion.storage import write_json
from alpharank.data.publishing.consolidation import (
    FinancialSourceInput,
    consolidate_financial_sources_with_share_quality,
)
from alpharank.data.quality.fundamental_quality import (
    audit_fundamental_quality,
    quarantine_implausible_share_candidates,
    validate_fundamental_quality,
)


@dataclass(frozen=True)
class FinancialCandidate:
    consolidated: pl.DataFrame
    lineage: pl.DataFrame
    source_summary: pl.DataFrame


def build_financial_candidate(
    *,
    run_dir: Path,
    source_refresh_contract: dict[str, object],
    sec_companyfacts: pl.DataFrame,
    sec_filing: pl.DataFrame,
    simfin: pl.DataFrame,
    yfinance: pl.DataFrame,
) -> FinancialCandidate:
    """Consolidate sources in priority order and persist their quality gates."""

    source_frames = (
        ("sec_companyfacts", sec_companyfacts, 1),
        ("sec_filing", sec_filing, 2),
        ("simfin", simfin, 3),
        ("yfinance", yfinance, 4),
    )
    sanitized_sources: list[FinancialSourceInput] = []
    quarantine: dict[str, object] = {}
    for source_name, source_frame, priority in source_frames:
        sanitized, report = quarantine_implausible_share_candidates(
            source_frame.select(_clean_financial_columns())
        )
        quarantine[source_name] = report
        sanitized_sources.append(
            FinancialSourceInput(
                source_name=source_name,
                frame=sanitized,
                priority=priority,
            )
        )
    source_refresh_contract["share_candidate_quarantine"] = quarantine
    write_json(run_dir / "share_candidate_quarantine.json", quarantine)

    consolidated, lineage, summary, selection_quality = (
        consolidate_financial_sources_with_share_quality(sanitized_sources)
    )
    source_refresh_contract["share_selection_quality"] = selection_quality
    write_json(run_dir / "share_selection_quality.json", selection_quality)
    fundamental_quality = audit_fundamental_quality(consolidated)
    source_refresh_contract["fundamental_quality_guard"] = fundamental_quality
    write_json(run_dir / "fundamental_quality_guard.json", fundamental_quality)
    validate_fundamental_quality(fundamental_quality)
    return FinancialCandidate(consolidated, lineage, summary)
