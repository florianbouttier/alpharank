#!/usr/bin/env python3
"""Build a causal historical sector release from annual SEC filing headers."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.data.sources.sec_filing_headers import (
    SecFilingHeaderConfig,
    acquire_sec_filing_headers,
)
from alpharank.data.sources.sector_identity import build_sector_identity_map
from alpharank.data.warehouse.sector_history import materialize_sector_history_release
from alpharank.data.warehouse.sector_sic_history import (
    build_sec_sic_sector_observations,
    load_sic_sector_taxonomy,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    args = _parse_args(project_root)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    universe = pl.read_parquet(args.ticker_universe)
    if "ticker" not in universe.columns:
        raise ValueError("Ticker universe must contain a ticker column.")
    tickers = tuple(universe["ticker"].drop_nulls().unique().sort().to_list())
    identities, unresolved = build_sector_identity_map(
        tickers,
        pl.read_parquet(args.general_reference_lineage),
    )
    ciks = tuple(identities["cik"].unique().sort().to_list())
    if args.limit_issuers is not None:
        ciks = ciks[: args.limit_issuers]
        identities = identities.filter(pl.col("cik").is_in(ciks))
    acquisition = acquire_sec_filing_headers(
        ciks,
        config=SecFilingHeaderConfig(
            project_root=project_root,
            submissions_archive=args.submissions_archive,
            headers_archive=args.headers_archive,
            user_agent=args.user_agent,
            run_id=args.run_id,
            start_year=args.start_year,
            end_year=args.end_year,
            header_workers=args.header_workers,
        ),
    )
    observations = build_sec_sic_sector_observations(
        acquisition.observations,
        identities,
        taxonomy=load_sic_sector_taxonomy(args.taxonomy),
        ingestion_run_id=args.run_id,
    )
    release = materialize_sector_history_release(
        observations,
        contract_path=args.contract,
        output_root=args.output_root,
        generated_at=datetime.now(timezone.utc),
        evidence_frames={
            "sec_filing_header_observations.parquet": acquisition.observations,
            "sector_identity_map.parquet": identities,
            "sector_identity_unresolved.parquet": unresolved,
            "sec_filing_header_failures.parquet": acquisition.failures,
        },
    )
    logging.info("sector_history_release=%s", release.release_dir)
    logging.info("sector_history_summary=%s", release.manifest["quality"])


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker-universe", type=Path, required=True)
    parser.add_argument("--general-reference-lineage", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--submissions-archive",
        type=Path,
        default=project_root / "data/warehouse/raw/sec_submissions/issuer_submissions",
    )
    parser.add_argument(
        "--headers-archive",
        type=Path,
        default=project_root / "data/warehouse/raw/sec_filings/filing_headers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data/warehouse/def/sector_history",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=project_root / "configs/data_contracts/sector_history_v2.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=project_root / "configs/data_contracts/sec_sic_sector_taxonomy_v1.json",
    )
    parser.add_argument(
        "--user-agent",
        default="Florian Bouttier florianbouttier@example.com",
    )
    parser.add_argument("--limit-issuers", type=int)
    parser.add_argument("--header-workers", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
