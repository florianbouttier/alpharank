from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.data.contracts.sector_history import resolve_point_in_time_sectors
from alpharank.data.sources.sec_filing_headers import parse_sec_filing_header
from alpharank.data.sources.sector_identity import build_sector_identity_map
from alpharank.data.warehouse.sector_history import build_sector_history
from alpharank.data.warehouse.sector_sic_history import (
    build_sec_sic_sector_observations,
    load_sic_sector_taxonomy,
    map_sic_to_sector,
)


def test_sec_header_parser_selects_matching_filer_and_acceptance_time() -> None:
    payload = b"""<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20200102150405\nFILER:\n\n\tCOMPANY DATA:\n\t\tCENTRAL INDEX KEY:\t0000000001\n\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\tBANKS [6021]\nFILER:\n\n\tCOMPANY DATA:\n\t\tCENTRAL INDEX KEY:\t0000320193\n\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\tELECTRONIC COMPUTERS [3571]\n</SEC-HEADER>"""

    parsed = parse_sec_filing_header(payload, expected_cik="320193")

    assert parsed == {
        "cik": "0000320193",
        "sic": "3571",
        "sic_description": "ELECTRONIC COMPUTERS",
        "available_at": "2020-01-02T20:04:05+00:00",
    }


def test_sec_header_parser_supports_legacy_sgml_tags() -> None:
    payload = b"""<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20100219160552\n<FILER>\n<COMPANY-DATA>\n<CIK>0000001800\n<ASSIGNED-SIC>2834\n</COMPANY-DATA>\n</FILER>\n</SEC-HEADER>"""

    parsed = parse_sec_filing_header(payload, expected_cik="1800")

    assert parsed is not None
    assert parsed["sic"] == "2834"
    assert parsed["sic_description"] == "SEC SIC 2834"


def test_reviewed_historical_bridge_replaces_undated_identity() -> None:
    general = pl.DataFrame(
        {
            "ticker": ["DTV.US", "AAPL.US"],
            "cik": ["999", "320193"],
        }
    )

    identities, unresolved = build_sector_identity_map(["DTV.US", "AAPL.US"], general)

    dtv = identities.filter(pl.col("ticker") == "DTV.US").row(0, named=True)
    assert dtv["cik"] == "0000944868"
    assert dtv["identity_valid_to"].isoformat() == "2015-07-31"
    assert dtv["identity_status"] == "reviewed_historical_bridge"
    assert unresolved.is_empty()


def test_sec_share_class_punctuation_and_reviewed_overrides_resolve() -> None:
    general = pl.DataFrame(
        {
            "ticker": ["BRK-B.US"],
            "cik": ["1067983"],
        }
    )

    identities, unresolved = build_sector_identity_map(
        ["BRK.B.US", "RDDT.US"],
        general,
    )

    assert unresolved.is_empty()
    assert identities.filter(pl.col("ticker") == "BRK.B.US")["cik"].item() == "0001067983"
    reddit = identities.filter(pl.col("ticker") == "RDDT.US").row(0, named=True)
    assert reddit["cik"] == "0001713445"
    assert reddit["identity_valid_from"].isoformat() == "2026-08-18"
    assert reddit["identity_evidence"]


def test_sic_taxonomy_maps_representative_sectors() -> None:
    taxonomy = load_sic_sector_taxonomy(_taxonomy_path())

    assert map_sic_to_sector("3571", taxonomy) == ("Technology", "computers")
    assert map_sic_to_sector("2834", taxonomy) == ("Healthcare", "pharmaceuticals")
    assert map_sic_to_sector("6798", taxonomy) == ("Real Estate", "reit")
    assert map_sic_to_sector("4931", taxonomy) == ("Utilities", "utilities")


def test_sec_observation_uses_filing_acceptance_not_download_time() -> None:
    headers = _headers()
    identities = _identities()
    taxonomy = load_sic_sector_taxonomy(_taxonomy_path())

    observations = build_sec_sic_sector_observations(
        headers,
        identities,
        taxonomy=taxonomy,
        ingestion_run_id="run-1",
    )
    history = build_sector_history(
        observations,
        allowed_sectors=sorted(taxonomy.allowed_sectors),
        aliases={},
        selection_policy_id="sec_filing_header_sic_as_known_v2",
    ).history

    assert history["observed_at"].to_list() == [datetime(2010, 2, 1, 21, 0, tzinfo=timezone.utc)]
    assert history["Sector"].to_list() == ["Technology"]
    assert history["selection_policy_id"].to_list() == ["sec_filing_header_sic_as_known_v2"]


def test_sector_resolution_blocks_dates_outside_identity_interval() -> None:
    taxonomy = load_sic_sector_taxonomy(_taxonomy_path())
    observations = build_sec_sic_sector_observations(
        _headers(),
        _identities(),
        taxonomy=taxonomy,
        ingestion_run_id="run-1",
    )
    history = build_sector_history(
        observations,
        allowed_sectors=sorted(taxonomy.allowed_sectors),
        aliases={},
    ).history
    decisions = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "decision_at": [
                datetime(2011, 6, 30, 20, tzinfo=timezone.utc),
                datetime(2012, 6, 29, 20, tzinfo=timezone.utc),
            ],
        }
    )

    resolved = resolve_point_in_time_sectors(decisions, history)

    assert resolved["Sector"].to_list() == ["Technology", None]
    assert resolved["identity_interval_eligible"].to_list() == [True, False]


def test_rejection_is_scoped_to_identity_when_ticker_and_time_match() -> None:
    taxonomy = load_sic_sector_taxonomy(_taxonomy_path())
    headers = pl.concat(
        [
            _headers(),
            _headers().with_columns(
                pl.lit("0000000002").alias("cik"),
                pl.lit("9999").alias("sic"),
                pl.lit("Unknown SIC").alias("sic_description"),
            ),
        ]
    )
    identities = pl.concat(
        [
            _identities(),
            _identities().with_columns(
                pl.lit("0000000002").alias("cik"),
                pl.lit("identity-2").alias("identity_id"),
            ),
        ]
    )
    observations = build_sec_sic_sector_observations(
        headers,
        identities,
        taxonomy=taxonomy,
        ingestion_run_id="run-1",
    )

    built = build_sector_history(
        observations,
        allowed_sectors=sorted(taxonomy.allowed_sectors),
        aliases={},
    )

    assert built.history["identity_id"].to_list() == ["identity-1"]
    assert built.rejections["identity_id"].to_list() == ["identity-2"]


def _headers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cik": ["0000000001"],
            "sic": ["3571"],
            "sic_description": ["Electronic computers"],
            "available_at": ["2010-02-01T21:00:00+00:00"],
            "retrieved_at": ["2026-11-09T10:00:00+00:00"],
            "accession_number": ["0000000001-10-000001"],
            "filing_date": ["2010-02-01"],
            "form": ["10-K"],
            "source_url": ["https://www.sec.gov/example.hdr.sgml"],
            "raw_observation_path": ["data/warehouse/raw/example"],
            "raw_observation_sha256": ["a" * 64],
        }
    )


def _identities() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "cik": ["0000000001"],
            "identity_id": ["identity-1"],
            "identity_valid_from": [date(2010, 1, 1)],
            "identity_valid_to": [date(2011, 12, 31)],
            "identity_source": ["test"],
            "identity_status": ["reviewed"],
            "identity_evidence": ["test evidence"],
        }
    ).with_columns(
        pl.col("identity_valid_from").cast(pl.Date),
        pl.col("identity_valid_to").cast(pl.Date),
    )


def _taxonomy_path() -> Path:
    return Path("configs/data_contracts/sec_sic_sector_taxonomy_v1.json")
