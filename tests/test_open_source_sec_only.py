from __future__ import annotations

import polars as pl

from alpharank.data.open_source.sec_only import (
    build_sec_only_earnings,
    build_sec_only_general_reference_from_raw_lineage,
)


def test_build_sec_only_general_reference_from_raw_lineage_ignores_yahoo_fields() -> None:
    raw_lineage = pl.DataFrame(
        {
            "ticker": ["NEM.US"],
            "name": ["Newmont Corp"],
            "exchange": ["NYSE"],
            "cik": ["0001164727"],
            "source": ["open_source_general"],
            "Sector": ["Basic Materials"],
            "industry": ["Gold"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Materials"],
            "sic": ["1040"],
            "sic_description": ["Gold Ores"],
            "mapping_rule": ["yfinance:sector"],
            "selected_name_source": ["yfinance"],
            "selected_exchange_source": ["yfinance"],
            "yahoo_name": ["Newmont Corp"],
            "yahoo_exchange": ["NYSE"],
            "yahoo_sector": ["Materials"],
            "yahoo_industry": ["Gold"],
            "sec_name": ["NEWMONT CORP"],
            "sec_exchange": ["NYSE"],
            "sec_cik": ["1164727"],
            "sec_sic": ["1040"],
            "sec_sic_description": ["Gold Ores"],
        }
    )

    general_reference, lineage = build_sec_only_general_reference_from_raw_lineage(raw_lineage)

    assert general_reference["name"].to_list() == ["NEWMONT CORP"]
    assert general_reference["Sector"].to_list() == ["Basic Materials"]
    assert general_reference["sector_source"].to_list() == ["sec_sic"]
    assert lineage["selected_name_source"].to_list() == ["sec_mapping"]
    assert lineage["yahoo_name"].to_list() == [None]


def test_build_sec_only_earnings_keeps_sec_actuals_and_nulls_market_fields() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2025-03-29"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": ["2025-05-01 20:00:00"],
            "accession_number": ["0000320193-25-000001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2025-03-29"],
            "reportDate": ["2025-05-01"],
            "epsActual": [1.53],
            "source": ["sec_companyfacts"],
            "source_label": ["EarningsPerShareDiluted"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2025],
        }
    )

    consolidated, lineage, long_frame = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
    )

    assert consolidated["epsActual"].to_list() == [1.53]
    assert consolidated["epsEstimate"].to_list() == [None]
    assert consolidated["surprisePercent"].to_list() == [None]
    assert consolidated["actual_source"].to_list() == ["sec_companyfacts"]
    assert consolidated["selected_source"].to_list() == ["sec_submissions+sec_companyfacts"]
    assert lineage["sec_epsActual"].to_list() == [1.53]
    assert lineage["yahoo_epsActual"].to_list() == [None]
    assert long_frame["source"].to_list() == ["open_source_earnings"]
