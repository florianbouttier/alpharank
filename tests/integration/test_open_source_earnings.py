from __future__ import annotations

import polars as pl

from alpharank.data.sources.earnings import (
    build_sec_companyfacts_earnings_actuals,
    consolidate_earnings,
    resolve_earnings_calendar_duplicates,
)
from alpharank.data.ingestion.frames import _with_financial_ingestion_metadata
from alpharank.data.ingestion.reference import (
    _identify_sec_filing_fallback_tickers,
    _identify_yfinance_financial_fallback_tickers,
)
from alpharank.data.open_source.sec import (
    _select_best_facts,
)
from alpharank.data.open_source.sec_filing import (
    _parse_atom_filings,
)


def test_earnings_calendar_key_is_unique() -> None:
    rows: list[dict[str, object]] = []
    for index in range(10):
        ticker = f"T{index:02d}.US"
        period_end = f"2025-{index + 1:02d}-15"
        rows.extend(
            [
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "reportDate": "2025-01-01",
                    "earningsDatetime": "2025-01-01 00:00:00",
                    "accession_number": f"bad-{index:02d}",
                    "form": "10-Q",
                    "fiscal_period": "Q1",
                    "fiscal_year": 2025,
                    "source": "sec_submissions",
                    "source_label": "invalid_pre_period",
                },
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "reportDate": f"2025-{index + 1:02d}-20",
                    "earningsDatetime": f"2025-{index + 1:02d}-20 00:00:00",
                    "accession_number": f"good-{index:02d}",
                    "form": "10-Q",
                    "fiscal_period": "Q1",
                    "fiscal_year": 2025,
                    "source": "sec_submissions",
                    "source_label": "valid_post_period",
                },
            ]
        )

    selected, audit = resolve_earnings_calendar_duplicates(pl.DataFrame(rows))
    reversed_selected, reversed_audit = resolve_earnings_calendar_duplicates(
        pl.DataFrame(list(reversed(rows)))
    )

    assert selected.height == 10
    assert selected.select("ticker", "period_end").n_unique() == 10
    assert selected["accession_number"].to_list() == [
        f"good-{index:02d}" for index in range(10)
    ]
    assert audit.height == 10
    assert set(audit["calendar_duplicate_count"]) == {2}
    assert selected.equals(reversed_selected)
    assert audit.equals(reversed_audit)


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


def test_build_sec_companyfacts_earnings_actuals_keeps_earliest_period_report() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2009-01-01",
                                "end": "2009-03-31",
                                "val": 1.0,
                                "fy": 2009,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2010-02-20",
                            },
                            {
                                "start": "2009-01-01",
                                "end": "2009-03-31",
                                "val": 9.0,
                                "fy": 2010,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2011-02-20",
                            },
                        ]
                    }
                }
            }
        }
    }

    actuals = build_sec_companyfacts_earnings_actuals(ticker="TEST", facts_payload=payload)

    assert actuals.height == 1
    assert actuals["reportDate"].to_list() == ["2010-02-20"]
    assert actuals["epsActual"].to_list() == [1.0]


def test_identify_yfinance_financial_fallback_tickers_ignores_lineage_schema_mismatches() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-03-31"],
            "source_label": [None],
        }
    )
    sec_filing = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-06-30"],
            "source_label": ["filing"],
        }
    )

    fallback_tickers = _identify_yfinance_financial_fallback_tickers(
        tickers=("AAPL",),
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_filing,
    )

    assert fallback_tickers == ("AAPL",)


def test_sec_filing_fallback_only_targets_active_tickers_without_companyfacts_rows() -> None:
    companyfacts = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2026-03-31"],
        }
    )

    fallback = _identify_sec_filing_fallback_tickers(
        tickers=("AAPL", "MSFT"),
        sec_companyfacts=companyfacts,
    )

    assert fallback == ("MSFT",)


def test_with_financial_ingestion_metadata_adds_missing_provider_lineage_columns() -> None:
    simfin_like = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-01"],
            "value": [100.0],
            "source": ["simfin"],
            "source_label": ["Revenue"],
        }
    )

    raw = _with_financial_ingestion_metadata(
        simfin_like,
        dataset="financials_simfin",
        run_id="run",
        ingested_at="2026-05-30T00:00:00Z",
    )

    assert raw["accession_number"].to_list() == [None]
    assert raw["form"].to_list() == [None]
    assert raw["fiscal_period"].to_list() == [None]
    assert raw["fiscal_year"].to_list() == [None]
    assert raw["dataset"].to_list() == ["financials_simfin"]


def test_build_sec_companyfacts_earnings_actuals_does_not_synthesize_q4_from_annual_eps() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 8.0,
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-20",
                            }
                        ]
                    }
                }
            }
        }
    }

    actuals = build_sec_companyfacts_earnings_actuals(ticker="TEST", facts_payload=payload)

    assert actuals.is_empty()


def test_build_sec_companyfacts_earnings_actuals_drops_implausible_fiscal_year_gap() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2009-10-01",
                                "end": "2009-12-31",
                                "val": -4.0,
                                "fy": 2011,
                                "fp": "Q4",
                                "form": "10-K",
                                "filed": "2012-02-20",
                            }
                        ]
                    }
                }
            }
        }
    }

    actuals = build_sec_companyfacts_earnings_actuals(ticker="TEST", facts_payload=payload)

    assert actuals.is_empty()


def test_select_best_facts_maps_quarterly_10k_to_q4() -> None:
    facts_payload = {
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "start": "2012-12-01",
                            "end": "2013-02-28",
                            "val": 103.0,
                            "fy": 2012,
                            "fp": "Q1",
                            "form": "10-K",
                            "filed": "2013-04-29",
                        }
                    ]
                }
            }
        }
    }

    selected = _select_best_facts("income_statement", ("us-gaap",), ("NetIncomeLoss",), facts_payload)

    assert len(selected) == 1
    assert selected[0]["fp"] == "Q4"


def test_parse_atom_filings_extracts_accession_and_dates() -> None:
    feed = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <content type="text/xml">
          <accession-number>0001047469-09-009634</accession-number>
          <filing-date>2009-11-06</filing-date>
          <filing-href>https://www.sec.gov/Archives/edgar/data/944868/000104746909009634/0001047469-09-009634-index.htm</filing-href>
          <filing-type>10-Q</filing-type>
        </content>
        <updated>2009-11-05T19:11:47-05:00</updated>
      </entry>
    </feed>
    """

    filings = _parse_atom_filings(feed, filing_type="10-Q")

    assert len(filings) == 1
    filing = filings[0]
    assert filing.accession_number == "0001047469-09-009634"
    assert filing.filing_date == "2009-11-06"
    assert filing.report_date == "2009-11-06"
    assert filing.form == "10-Q"
    assert filing.primary_document == "0001047469-09-009634-index.htm"


def test_select_best_facts_derives_quarters_from_cumulative_duration_facts() -> None:
    facts_payload = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2010-01-01",
                            "end": "2010-03-31",
                            "val": 100.0,
                            "fy": 2010,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2010-05-01",
                        },
                        {
                            "start": "2010-01-01",
                            "end": "2010-06-30",
                            "val": 250.0,
                            "fy": 2010,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2010-08-01",
                        },
                        {
                            "start": "2010-01-01",
                            "end": "2010-09-30",
                            "val": 390.0,
                            "fy": 2010,
                            "fp": "Q3",
                            "form": "10-Q",
                            "filed": "2010-11-01",
                        },
                        {
                            "start": "2010-01-01",
                            "end": "2010-12-31",
                            "val": 520.0,
                            "fy": 2010,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2011-02-20",
                        },
                    ]
                }
            }
        }
    }

    selected = _select_best_facts("income_statement", ("us-gaap",), ("Revenues",), facts_payload)
    by_end = {row["end"]: row for row in selected}

    assert by_end["2010-03-31"]["val"] == 100.0
    assert by_end["2010-06-30"]["val"] == 150.0
    assert by_end["2010-09-30"]["val"] == 140.0
    assert by_end["2010-12-31"]["val"] == 130.0


def test_select_best_facts_derives_q4_using_next_quarter_end_not_annual_end() -> None:
    facts_payload = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2022-12-01",
                            "end": "2023-02-28",
                            "val": 100.0,
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-04-01",
                        },
                        {
                            "start": "2023-03-01",
                            "end": "2023-05-31",
                            "val": 150.0,
                            "fy": 2023,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2023-07-01",
                        },
                        {
                            "start": "2023-06-01",
                            "end": "2023-08-31",
                            "val": 140.0,
                            "fy": 2023,
                            "fp": "Q3",
                            "form": "10-Q",
                            "filed": "2023-10-01",
                        },
                        {
                            "start": "2022-12-01",
                            "end": "2024-01-31",
                            "val": 520.0,
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2024-02-20",
                        },
                    ]
                }
            }
        }
    }

    selected = _select_best_facts("income_statement", ("us-gaap",), ("Revenues",), facts_payload)
    q4 = [row for row in selected if row["fp"] == "Q4"][0]

    assert q4["end"] == "2023-11-30"
    assert q4["start"] == "2023-09-01"
    assert q4["val"] == 130.0


def test_q4_derivation_keeps_same_preferred_tag_as_direct_quarters() -> None:
    def fact(start, end, value, fp, form, filed):
        return {
            "start": start,
            "end": end,
            "val": value,
            "fy": 2011,
            "fp": fp,
            "form": form,
            "filed": filed,
        }

    facts_payload = {
        "us-gaap": {
            "PreferredNetIncome": {
                "units": {
                    "USD": [
                        fact("2010-11-01", "2011-01-31", 193.0, "Q1", "10-Q", "2011-03-09"),
                        fact("2011-02-01", "2011-04-30", 200.0, "Q2", "10-Q", "2011-06-07"),
                        fact("2011-05-01", "2011-07-31", 330.0, "Q3", "10-Q", "2011-09-07"),
                        fact("2010-11-01", "2011-10-31", 1012.0, "FY", "10-K", "2011-12-16"),
                        fact("2011-01-01", "2011-12-31", 684.0, "FY", "10-K", "2012-02-01"),
                    ]
                }
            },
            "FallbackNetIncome": {
                "units": {
                    "USD": [
                        fact("2010-11-01", "2011-10-31", 684.0, "FY", "10-K", "2011-12-16"),
                    ]
                }
            },
        }
    }

    selected = _select_best_facts(
        "income_statement",
        ("us-gaap",),
        ("PreferredNetIncome", "FallbackNetIncome"),
        facts_payload,
    )
    q4 = next(row for row in selected if row["fp"] == "Q4")

    assert q4["tag"] == "PreferredNetIncome_derived_q4"
    assert q4["val"] == 289.0


def test_select_best_facts_derives_missing_q1_from_q2_cumulative_and_q2_direct() -> None:
    facts_payload = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2010-04-01",
                            "end": "2010-06-30",
                            "val": 150.0,
                            "fy": 2010,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2010-08-01",
                        },
                        {
                            "start": "2010-01-01",
                            "end": "2010-06-30",
                            "val": 250.0,
                            "fy": 2010,
                            "fp": "Q2",
                            "form": "10-Q",
                            "filed": "2010-08-01",
                        },
                        {
                            "start": "2010-01-01",
                            "end": "2010-09-30",
                            "val": 390.0,
                            "fy": 2010,
                            "fp": "Q3",
                            "form": "10-Q",
                            "filed": "2010-11-01",
                        },
                    ]
                }
            }
        }
    }

    selected = _select_best_facts("income_statement", ("us-gaap",), ("Revenues",), facts_payload)
    by_end = {row["end"]: row for row in selected}

    assert by_end["2010-03-31"]["val"] == 100.0
    assert by_end["2010-03-31"]["fp"] == "Q1"
    assert by_end["2010-06-30"]["val"] == 150.0
