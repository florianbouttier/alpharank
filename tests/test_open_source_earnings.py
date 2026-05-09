from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.benchmark import normalize_eodhd_earnings
from alpharank.data.open_source.earnings import (
    align_sec_actuals_to_calendar,
    build_sec_companyfacts_earnings_actuals,
    consolidate_earnings,
    empty_earnings_calendar_frame,
)
from alpharank.data.open_source.ingestion import (
    _canonicalize_price_tickers,
    _filter_earnings_years,
    _identify_price_history_backfill_tickers,
)
from alpharank.data.open_source.general_reference import build_general_reference
from alpharank.data.open_source.pipeline import _combine_sec_earnings_actuals
from alpharank.data.open_source.sec import _select_share_facts
from alpharank.data.open_source.sec_filing import _infer_fiscal_period


def test_consolidate_earnings_prefers_sec_calendar_and_yahoo_market_fields() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": ["2025-05-01 20:00:00"],
            "accession_number": ["0001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    yahoo_earnings = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": [None],
            "reportDate": ["2025-04-30"],
            "earningsDatetime": ["2025-04-30 21:00:00"],
            "epsEstimate": [1.40],
            "epsActual": [1.50],
            "surprisePercent": [7.0],
            "source": ["yfinance"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-05-01"],
            "epsActual": [1.45],
            "source": ["sec_companyfacts"],
            "source_label": ["EarningsPerShareDiluted"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
        }
    )

    consolidated, lineage, long_frame = consolidate_earnings(
        sec_calendar=sec_calendar,
        yahoo_earnings=yahoo_earnings,
        sec_actuals=sec_actuals,
    )

    assert consolidated.height == 1
    assert consolidated["period_end"].to_list() == ["2025-03-31"]
    assert consolidated["reportDate"].to_list() == ["2025-04-30"]
    assert consolidated["epsActual"].to_list() == [1.50]
    assert consolidated["epsEstimate"].to_list() == [1.40]
    assert consolidated["selected_source"].to_list() == ["sec_submissions+yfinance"]
    assert lineage["candidate_sources"].to_list() == ["sec_submissions | yfinance | sec_companyfacts"]
    assert lineage["sec_reportDate"].to_list() == ["2025-05-01"]
    assert long_frame.filter(pl.col("metric") == "eps_actual")["date"].to_list() == ["2025-03-31"]
    assert long_frame.filter(pl.col("metric") == "eps_actual")["filing_date"].to_list() == ["2025-04-30"]


def test_consolidate_earnings_matches_yahoo_release_before_sec_filing_when_period_end_matches() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["KEYS.US"],
            "period_end": ["2025-10-31"],
            "reportDate": ["2025-12-17"],
            "earningsDatetime": ["2025-12-17 21:00:00"],
            "accession_number": ["0002"],
            "form": ["10-Q"],
            "fiscal_period": ["Q4"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    yahoo_earnings = pl.DataFrame(
        {
            "ticker": ["KEYS.US"],
            "period_end": [None],
            "reportDate": ["2025-11-25"],
            "earningsDatetime": ["2025-11-25 21:00:00"],
            "epsEstimate": [1.82],
            "epsActual": [1.91],
            "surprisePercent": [4.95],
            "source": ["yfinance"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["KEYS.US"],
            "period_end": ["2025-10-31"],
            "reportDate": ["2025-12-17"],
            "epsActual": [1.35],
            "source": ["sec_companyfacts"],
            "source_label": ["EarningsPerShareDiluted"],
            "form": ["10-Q"],
            "fiscal_period": ["Q4"],
            "fiscal_year": [2025],
        }
    )

    consolidated, lineage, _ = consolidate_earnings(
        sec_calendar=sec_calendar,
        yahoo_earnings=yahoo_earnings,
        sec_actuals=sec_actuals,
    )

    assert consolidated["reportDate"].to_list() == ["2025-11-25"]
    assert consolidated["epsEstimate"].to_list() == [1.82]
    assert consolidated["epsActual"].to_list() == [1.91]
    assert lineage["sec_reportDate"].to_list() == ["2025-12-17"]
    assert lineage["yahoo_reportDate"].to_list() == ["2025-11-25"]


def test_build_general_reference_falls_back_to_sec_sic_mapping() -> None:
    sec_mapping = pl.DataFrame(
        {
            "ticker": ["NEM"],
            "name": ["Newmont"],
            "exchange": ["NYSE"],
            "cik": [1164727],
        }
    )
    yahoo_metadata = pl.DataFrame(
        {
            "ticker": ["NEM.US"],
            "name": ["Newmont"],
            "exchange": ["NYSE"],
            "sector_raw_value": [None],
            "industry": [None],
        }
    )
    sec_profiles = pl.DataFrame(
        {
            "ticker": ["NEM.US"],
            "cik": ["0001164727"],
            "sic": ["1040"],
            "sic_description": ["Gold Ores"],
        }
    )

    general_reference, lineage = build_general_reference(
        tickers=["NEM"],
        sec_mapping=sec_mapping,
        yahoo_metadata=yahoo_metadata,
        sec_profiles=sec_profiles,
    )

    assert general_reference["Sector"].to_list() == ["Basic Materials"]
    assert general_reference["sector_source"].to_list() == ["sec_sic"]
    assert lineage["mapping_rule"].to_list() == ["sec_sic:basic_materials"]


def test_normalize_eodhd_earnings_uses_period_end_as_date(tmp_path: Path) -> None:
    data_dir = tmp_path / "eodhd" / "output"
    data_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "reportDate": ["2025-05-01"],
            "date": ["2025-03-31"],
            "epsActual": [1.5],
            "epsEstimate": [1.4],
            "surprisePercent": [7.0],
        }
    ).write_parquet(data_dir / "US_Earnings.parquet")
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-01"], "adjusted_close": [1.0], "close": [1.0], "open": [1.0], "high": [1.0], "low": [1.0], "volume": [1.0]}).write_parquet(
        data_dir / "US_Finalprice.parquet"
    )
    (data_dir / "SP500_Constituents.csv").write_text("Date,Ticker,Name\n2025-01-01,AAPL,Apple\n", encoding="utf-8")

    normalized = normalize_eodhd_earnings(tmp_path / "eodhd", ["AAPL"], 2025)

    assert normalized.filter(pl.col("metric") == "eps_actual")["date"].to_list() == ["2025-03-31"]
    assert normalized.filter(pl.col("metric") == "eps_actual")["filing_date"].to_list() == ["2025-05-01"]


def test_filter_earnings_years_falls_back_to_report_date_when_period_end_missing() -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "MSFT.US"],
            "period_end": [None, "2024-12-31"],
            "reportDate": ["2025-05-01", "2025-01-28"],
            "earningsDatetime": ["2025-05-01 20:00:00", "2025-01-28 21:00:00"],
            "epsEstimate": [1.4, 2.8],
            "epsActual": [1.5, 2.9],
            "surprisePercent": [7.0, 3.5],
            "source": ["yfinance", "yfinance"],
            "source_label": [None, None],
            "calendar_source": [None, None],
            "actual_source": [None, None],
            "estimate_source": [None, None],
            "accession_number": [None, None],
            "form": [None, None],
            "fiscal_period": [None, None],
            "fiscal_year": [None, None],
            "dataset": ["earnings_yfinance", "earnings_yfinance"],
            "ingestion_run_id": ["run", "run"],
            "ingested_at": ["2026-03-28T14:00:00Z", "2026-03-28T14:00:00Z"],
        }
    )

    filtered = _filter_earnings_years(frame, [2025])

    assert filtered["ticker"].to_list() == ["AAPL.US"]


def test_identify_price_history_backfill_tickers_flags_recently_added_price_histories() -> None:
    existing = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "MSFT.US", "MSFT.US", "GEHC.US", "GEHC.US"],
            "date": ["2005-01-03", "2026-03-27", "2026-03-13", "2026-03-27", "2023-01-04", "2026-03-27"],
        }
    )

    result = _identify_price_history_backfill_tickers(
        requested_tickers=["AAPL", "MSFT", "GEHC", "NVDA"],
        existing_prices=existing,
        explicit_start_date="2005-01-01",
        mode="daily",
    )

    assert result == ("MSFT", "NVDA")


def test_canonicalize_price_tickers_merges_share_class_aliases() -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["BRK-B.US", "BRK.B.US", "AAPL.US"],
            "date": ["2026-03-27", "2026-03-27", "2026-03-27"],
            "source": ["yfinance", "yfinance", "yfinance"],
            "dataset": ["prices_yfinance_backfill", "prices_yfinance", "prices_yfinance"],
            "ingested_at": ["2026-03-30T21:00:00Z", "2026-03-30T21:05:00Z", "2026-03-30T21:05:00Z"],
        }
    )

    result = _canonicalize_price_tickers(frame, ticker_list=["BRK.B", "AAPL"])

    assert result["ticker"].to_list() == ["AAPL.US", "BRK.B.US"]


def test_select_share_facts_sums_share_classes_for_same_filing() -> None:
    selected = _select_share_facts(
        [
            {
                "tag": "EntityCommonStockSharesOutstanding",
                "tag_priority": 0,
                "end": "2025-11-30",
                "filed": "2026-01-10",
                "val": 289_000_000.0,
                "fp": "Q4",
                "form": "10-Q",
                "has_dimensions": True,
                "dimensions": (
                    ("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonClassAMember"),
                    ("us-gaap:StatementEquityComponentsAxis", "us-gaap:CommonStockMember"),
                ),
                "statement_class_member": "us-gaap:CommonClassAMember",
            },
            {
                "tag": "EntityCommonStockSharesOutstanding",
                "tag_priority": 0,
                "end": "2025-11-30",
                "filed": "2026-01-10",
                "val": 1_191_000_000.0,
                "fp": "Q4",
                "form": "10-Q",
                "has_dimensions": True,
                "dimensions": (
                    ("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonClassBMember"),
                    ("us-gaap:StatementEquityComponentsAxis", "us-gaap:CommonStockMember"),
                ),
                "statement_class_member": "us-gaap:CommonClassBMember",
            },
        ]
    )

    assert len(selected) == 1
    assert selected[0]["val"] == 1_480_000_000.0
    assert selected[0]["tag"] == "SummedStatementClassOfStockAxisMembers"


def test_select_share_facts_prefers_dimensionless_total_when_available() -> None:
    selected = _select_share_facts(
        [
            {
                "tag": "EntityCommonStockSharesOutstanding",
                "tag_priority": 0,
                "end": "2025-12-31",
                "filed": "2026-02-01",
                "val": 615_355_540.0,
                "fp": "Q4",
                "form": "10-K",
                "has_dimensions": False,
                "dimensions": (),
                "statement_class_member": None,
            },
            {
                "tag": "EntityCommonStockSharesOutstanding",
                "tag_priority": 0,
                "end": "2025-12-31",
                "filed": "2026-02-01",
                "val": 615_000_000.0,
                "fp": "Q4",
                "form": "10-K",
                "has_dimensions": True,
                "dimensions": (("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonClassAMember"),),
                "statement_class_member": "us-gaap:CommonClassAMember",
            },
            {
                "tag": "EntityCommonStockSharesOutstanding",
                "tag_priority": 0,
                "end": "2025-12-31",
                "filed": "2026-02-01",
                "val": 355_540.0,
                "fp": "Q4",
                "form": "10-K",
                "has_dimensions": True,
                "dimensions": (("us-gaap:StatementClassOfStockAxis", "custom:ClassXMember"),),
                "statement_class_member": "custom:ClassXMember",
            },
        ]
    )

    assert len(selected) == 1
    assert selected[0]["val"] == 615_355_540.0
    assert selected[0]["tag"] == "EntityCommonStockSharesOutstanding"


def test_sec_companyfacts_earnings_prefers_usd_per_share_unit() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "segment": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "val": 999.0,
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-05-01",
                            }
                        ],
                        "USD/shares": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "val": 1.41,
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-05-01",
                            }
                        ],
                    }
                }
            }
        }
    }

    actuals = build_sec_companyfacts_earnings_actuals(ticker="CVS", facts_payload=payload)

    assert actuals.height == 1
    assert actuals["epsActual"].to_list() == [1.41]
    assert actuals["source"].to_list() == ["sec_companyfacts"]


def test_combine_sec_earnings_actuals_keeps_filing_fallback_and_lineage() -> None:
    sec_companyfacts = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "epsActual": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )
    sec_filing = pl.DataFrame(
        {
            "ticker": ["ADT.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-04-24"],
            "epsActual": [0.15],
            "source": ["sec_filing"],
            "source_label": ["EarningsPerShareDiluted"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
        }
    )
    combined = _combine_sec_earnings_actuals(sec_companyfacts=sec_companyfacts, sec_filing=sec_filing)

    assert combined.height == 1
    assert combined["source"].to_list() == ["sec_filing"]

    sec_calendar = pl.DataFrame(
        {
            "ticker": ["ADT.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-04-24"],
            "earningsDatetime": ["2025-04-24 00:00:00"],
            "accession_number": ["0001703056-25-000069"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )

    yahoo_earnings = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "period_end": pl.String,
            "epsEstimate": pl.Float64,
            "epsActual": pl.Float64,
            "surprisePercent": pl.Float64,
            "source": pl.String,
        }
    )

    consolidated, lineage, _ = consolidate_earnings(
        sec_calendar=sec_calendar,
        yahoo_earnings=yahoo_earnings,
        sec_actuals=combined,
    )

    assert consolidated["epsActual"].to_list() == [0.15]
    assert consolidated["actual_source"].to_list() == ["sec_filing"]
    assert lineage["actual_source"].to_list() == ["sec_filing"]


def test_align_sec_actuals_to_calendar_uses_report_date_and_nearest_period() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["V.US"],
            "period_end": ["2025-09-30"],
            "reportDate": ["2025-11-06"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["V.US", "V.US"],
            "period_end": ["2024-09-30", "2025-06-30"],
            "reportDate": ["2025-11-06", "2025-11-06"],
            "epsActual": [2.98, 2.69],
            "source": ["sec_filing", "sec_filing"],
            "source_label": ["EarningsPerShareDiluted", "EarningsPerShareDiluted"],
            "form": ["10-K", "10-K"],
            "fiscal_period": ["Q4", "Q3"],
            "fiscal_year": [2025, 2025],
        }
    )

    aligned = align_sec_actuals_to_calendar(sec_calendar=sec_calendar, sec_actuals=sec_actuals)

    assert aligned.filter(pl.col("period_end") == "2025-09-30")["epsActual"].to_list() == [2.69]
    assert "aligned_reportDate" in aligned.filter(pl.col("period_end") == "2025-09-30")["source_label"].to_list()[0]


def test_infer_fiscal_period_maps_quarter_subperiods_inside_fy_filings() -> None:
    assert _infer_fiscal_period(
        filing_fp="FY",
        report_date="2025-09-30",
        start="2024-10-01",
        end="2024-12-31",
    ) == "Q1"
    assert _infer_fiscal_period(
        filing_fp="FY",
        report_date="2025-09-30",
        start="2025-01-01",
        end="2025-03-31",
    ) == "Q2"
    assert _infer_fiscal_period(
        filing_fp="FY",
        report_date="2025-09-30",
        start="2025-04-01",
        end="2025-06-30",
    ) == "Q3"
    assert _infer_fiscal_period(
        filing_fp="FY",
        report_date="2025-09-30",
        start="2025-07-01",
        end="2025-09-30",
    ) == "Q4"
