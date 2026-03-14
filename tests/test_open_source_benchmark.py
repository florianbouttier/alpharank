import pandas as pd
import polars as pl
import xml.etree.ElementTree as ET

from alpharank.data.open_source.benchmark import (
    _build_financial_comparison_table,
    _build_price_comparison_table,
    build_audited_metric_catalog,
    build_error_summary_tables,
    build_financial_alignment,
    build_price_alignment,
)
from alpharank.data.open_source.consolidation import FinancialSourceInput, consolidate_financial_sources
from alpharank.data.open_source.config import METRIC_SPECS
from alpharank.data.open_source.pipeline import (
    _identify_sec_filing_fallback_tickers,
    _identify_yfinance_financial_fallback_tickers,
)
from alpharank.data.open_source.sec import _select_best_facts
from alpharank.data.open_source.sec_filing import _derive_missing_total_liabilities, _parse_contexts, _parse_document_focus
from alpharank.data.open_source.simfin import _extract_metric_frames
from alpharank.data.open_source.yahoo import _extract_statement_frame


def test_build_price_alignment_computes_diffs() -> None:
    eodhd = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US"],
            "date": ["2025-01-02", "2025-01-03"],
            "adjusted_close": [100.0, 101.0],
            "close": [100.0, 101.0],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.0, 99.0],
            "volume": [10.0, 11.0],
        }
    )
    yahoo = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US"],
            "date": ["2025-01-02", "2025-01-03"],
            "adjusted_close": [100.5, 100.0],
            "close": [100.5, 100.0],
            "open": [99.5, 99.0],
            "high": [101.5, 101.0],
            "low": [98.5, 98.0],
            "volume": [10.0, 11.0],
        }
    )

    result = build_price_alignment(eodhd, yahoo)

    assert result.height == 2
    assert result["match_status"].to_list() == ["matched", "matched"]
    assert result["value_diff"].to_list() == [0.5, -1.0]


def test_build_financial_alignment_keeps_filing_date_diff() -> None:
    eodhd = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-12-31"],
            "filing_date": ["2026-01-30"],
            "value": [124_300_000_000.0],
            "source": ["eodhd"],
            "source_label": ["totalRevenue"],
        }
    )
    sec = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-12-31"],
            "filing_date": ["2026-01-31"],
            "value": [124_300_000_000.0],
            "source": ["sec_companyfacts"],
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2026],
        }
    )

    result = build_financial_alignment(eodhd, sec, "sec_companyfacts")

    assert result.height == 1
    assert result["match_status"].item() == "matched"
    assert result["date_diff_days"].item() == 0


def test_extract_statement_frame_normalizes_yfinance_wide_frame() -> None:
    wide = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [100.0, 40.0, 12.0],
            pd.Timestamp("2025-09-30"): [90.0, 35.0, 10.0],
        },
        index=["Total Revenue", "Net Income", "Capital Expenditure"],
    )

    income = _extract_statement_frame("AAPL", "income_statement", wide)
    cash_flow = _extract_statement_frame("AAPL", "cash_flow", wide)

    assert income.select("metric").to_series().to_list() == ["revenue", "revenue", "net_income", "net_income"]
    assert income.filter(pl.col("metric") == "revenue")["value"].to_list() == [100.0, 90.0]
    assert cash_flow.filter(pl.col("metric") == "capital_expenditures")["value"].to_list() == [12.0, 10.0]


def test_select_best_facts_prefers_quarterly_duration_and_tag_priority() -> None:
    facts = {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "start": "2025-10-01",
                            "end": "2025-12-31",
                            "val": 10.0,
                            "filed": "2026-01-30",
                            "form": "10-Q",
                            "fy": 2026,
                            "fp": "Q1",
                        }
                    ]
                }
            },
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2025-01-01",
                            "end": "2025-12-31",
                            "val": 99.0,
                            "filed": "2026-02-01",
                            "form": "10-K",
                            "fy": 2025,
                            "fp": "FY",
                        }
                    ]
                }
            },
        }
    }

    selected = _select_best_facts(
        "income_statement",
        ("us-gaap",),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
        facts,
    )

    assert len(selected) == 1
    assert selected[0]["val"] == 10.0


def test_select_best_facts_derives_q4_from_annual_fact() -> None:
    facts = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2024-09-29",
                            "end": "2024-12-28",
                            "val": 100.0,
                            "filed": "2025-01-31",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q1",
                        },
                        {
                            "start": "2024-12-29",
                            "end": "2025-03-29",
                            "val": 110.0,
                            "filed": "2025-05-02",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q2",
                        },
                        {
                            "start": "2025-03-30",
                            "end": "2025-06-28",
                            "val": 120.0,
                            "filed": "2025-08-01",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q3",
                        },
                        {
                            "start": "2024-09-29",
                            "end": "2025-09-27",
                            "val": 460.0,
                            "filed": "2025-10-31",
                            "form": "10-K",
                            "fy": 2025,
                            "fp": "FY",
                        },
                    ]
                }
            }
        }
    }

    selected = _select_best_facts("income_statement", ("us-gaap",), ("Revenues",), facts)
    q4 = [row for row in selected if row["end"] == "2025-09-27"]

    assert len(q4) == 1
    assert q4[0]["fp"] == "Q4"
    assert q4[0]["val"] == 130.0


def test_select_best_facts_prefers_period_end_share_fact() -> None:
    facts = {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2025-04-18",
                            "val": 14935826000.0,
                            "filed": "2025-05-02",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q2",
                        }
                    ]
                }
            },
            "CommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2025-03-29",
                            "val": 14939315000.0,
                            "filed": "2025-05-02",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q2",
                        }
                    ]
                }
            },
        }
    }

    selected = _select_best_facts(
        "shares",
        ("dei",),
        ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
        facts,
    )

    assert len(selected) == 1
    assert selected[0]["end"] == "2025-03-29"
    assert selected[0]["tag"] == "CommonStockSharesOutstanding"


def test_select_best_facts_calendarizes_frame_when_available() -> None:
    facts = {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "end": "2025-04-24",
                            "val": 7432543865.0,
                            "filed": "2025-04-30",
                            "form": "10-Q",
                            "fy": 2025,
                            "fp": "Q3",
                            "frame": "CY2025Q1I",
                        }
                    ]
                }
            }
        }
    }

    selected = _select_best_facts(
        "shares",
        ("dei",),
        ("EntityCommonStockSharesOutstanding",),
        facts,
    )

    assert len(selected) == 1
    assert selected[0]["end"] == "2025-03-31"


def test_build_error_summary_tables_applies_threshold_pct() -> None:
    price_alignment = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "AAPL.US"],
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "source": ["yfinance", "yfinance", "yfinance"],
            "statement": ["price", "price", "price"],
            "metric": ["adjusted_close", "adjusted_close", "adjusted_close"],
            "match_status": ["matched", "matched", "eodhd_only"],
            "diff_pct": [0.2, 0.8, None],
            "date_diff_days": [0, 0, None],
        }
    )
    financial_alignment = pl.DataFrame(
        {
            "source": ["yfinance", "yfinance", "sec_companyfacts"],
            "ticker": ["AAPL.US", "AAPL.US", "MSFT.US"],
            "statement": ["income_statement", "income_statement", "balance_sheet"],
            "metric": ["revenue", "revenue", "total_assets"],
            "match_status": ["matched", "open_only", "matched"],
            "diff_pct": [0.6, None, 0.1],
            "date_diff_days": [0, None, 0],
        }
    )

    (
        price_summary,
        statement_summary,
        metric_summary,
        ticker_summary,
        ticker_metric_summary,
        price_ticker_summary,
        price_ticker_metric_summary,
    ) = build_error_summary_tables(
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        threshold_pct=0.5,
    )

    assert price_summary["error_rows"].item() == 1
    assert price_summary["error_rate_pct"].item() == 50.0
    assert price_summary["ok_rows"].item() == 1
    assert price_summary["eodhd_rows"].item() == 3
    assert price_summary["open_rows"].item() == 2
    assert statement_summary.filter(pl.col("source") == "yfinance")["error_rows"].item() == 1
    assert metric_summary.filter(pl.col("metric") == "revenue")["error_rows"].item() == 1
    assert ticker_summary.filter(pl.col("ticker") == "AAPL.US")["error_rows"].item() == 1
    assert ticker_metric_summary.filter((pl.col("ticker") == "AAPL.US") & (pl.col("metric") == "revenue"))["error_rows"].item() == 1
    assert price_ticker_summary.filter(pl.col("ticker") == "AAPL.US")["error_rows"].item() == 1
    assert price_ticker_metric_summary.filter(pl.col("ticker") == "AAPL.US")["matched_rows"].item() == 2


def test_build_financial_alignment_matches_nearest_quarter_end() -> None:
    eodhd = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["balance_sheet"],
            "metric": ["total_assets"],
            "date": ["2025-12-31"],
            "filing_date": ["2026-01-30"],
            "value": [100.0],
            "source": ["eodhd"],
            "source_label": ["totalAssets"],
        }
    )
    sec = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["balance_sheet"],
            "metric": ["total_assets"],
            "date": ["2025-12-27"],
            "filing_date": ["2026-01-30"],
            "value": [100.0],
            "source": ["sec_companyfacts"],
            "source_label": ["Assets"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2026],
        }
    )

    result = build_financial_alignment(eodhd, sec, "sec_companyfacts", tolerance_days=10)

    assert result["match_status"].item() == "matched"
    assert result["date_diff_days"].item() == -4


def test_build_audited_metric_catalog_lists_enabled_sources() -> None:
    catalog = build_audited_metric_catalog(
        include_yfinance_financials=False,
        include_yfinance_earnings=False,
        include_sec_filing_financials=True,
        include_simfin_financials=True,
        include_open_source_consolidated=True,
    )

    assert catalog.filter((pl.col("statement") == "income_statement") & (pl.col("metric") == "net_income") & (pl.col("source") == "sec_companyfacts")).height == 1
    assert catalog.filter((pl.col("statement") == "income_statement") & (pl.col("metric") == "net_income") & (pl.col("source") == "sec_filing")).height == 1
    assert catalog.filter(pl.col("source") == "yfinance_earnings").is_empty()
    assert catalog.filter((pl.col("statement") == "price") & (pl.col("metric") == "adjusted_close") & (pl.col("source") == "yfinance")).height == 1
    assert catalog.filter((pl.col("statement") == "income_statement") & (pl.col("metric") == "net_income") & (pl.col("source") == "simfin")).height == 1
    assert catalog.filter((pl.col("statement") == "income_statement") & (pl.col("metric") == "net_income") & (pl.col("source") == "open_source_consolidated")).height == 1


def test_comparison_tables_include_side_by_side_values_and_status() -> None:
    price_alignment = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "AAPL.US"],
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "eodhd_adjusted_close": [100.0, 100.0, 100.0],
            "yahoo_adjusted_close": [100.2, 101.0, None],
            "source": ["yfinance", "yfinance", "yfinance"],
            "statement": ["price", "price", "price"],
            "metric": ["adjusted_close", "adjusted_close", "adjusted_close"],
            "match_status": ["matched", "matched", "eodhd_only"],
            "value_diff": [0.2, 1.0, None],
            "diff_pct": [0.2, 1.0, None],
            "date_diff_days": [0, 0, None],
        }
    )
    price_table = _build_price_comparison_table(price_alignment, threshold_pct=0.5)

    assert price_table["comparison_status"].to_list() == [
        "within_threshold",
        "threshold_breach",
        "missing_in_open_source",
    ]
    assert price_table["eodhd_adjusted_close"].to_list() == [100.0, 100.0, 100.0]
    assert price_table["yfinance_adjusted_close"].to_list() == [100.2, 101.0, None]

    financial_alignment = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "AAPL.US"],
            "source": ["sec_companyfacts", "sec_companyfacts", "sec_companyfacts"],
            "statement": ["income_statement", "income_statement", "income_statement"],
            "metric": ["net_income", "net_income", "net_income"],
            "date": ["2025-03-31", "2025-06-30", "2025-09-30"],
            "match_status": ["matched", "matched", "open_only"],
            "eodhd_value": [10.0, 10.0, None],
            "open_value": [10.0, 11.0, 12.0],
            "value_diff": [0.0, 1.0, None],
            "diff_pct": [0.0, 10.0, None],
            "eodhd_filing_date": ["2025-05-01", "2025-08-01", None],
            "open_filing_date": ["2025-05-02", "2025-08-02", "2025-11-02"],
            "date_diff_days": [1, 1, None],
            "eodhd_source_label": ["netIncome", "netIncome", None],
            "open_source_label": ["NetIncomeLoss", "NetIncomeLoss", "NetIncomeLoss"],
        }
    )
    financial_table = _build_financial_comparison_table(financial_alignment, threshold_pct=0.5)

    assert financial_table["comparison_status"].to_list() == [
        "within_threshold",
        "threshold_breach",
        "missing_in_eodhd",
    ]
    assert financial_table["eodhd_value"].to_list() == [10.0, 10.0, None]
    assert financial_table["open_value"].to_list() == [10.0, 11.0, 12.0]


def test_extract_metric_frames_normalizes_simfin_quarterly_data() -> None:
    spec = next(spec for spec in METRIC_SPECS if spec.metric == "revenue")
    dataset_frames = {
        "income_statement": pl.DataFrame(
            {
                "Ticker": ["AAPL", "AAPL"],
                "Report Date": ["2025-03-31", "2025-06-30"],
                "Publish Date": ["2025-05-02", "2025-08-01"],
                "Fiscal Period": ["Q1", "Q2"],
                "Fiscal Year": [2025, 2025],
                "Revenue": [100.0, 110.0],
            }
        )
    }

    result = _extract_metric_frames(spec=spec, dataset_frames=dataset_frames, year=2025)

    assert result.height == 2
    assert result["ticker"].to_list() == ["AAPL.US", "AAPL.US"]
    assert result["date"].to_list() == ["2025-03-31", "2025-06-30"]
    assert result["filing_date"].to_list() == ["2025-05-02", "2025-08-01"]
    assert result["value"].to_list() == [100.0, 110.0]
    assert result["source"].unique().to_list() == ["simfin"]


def test_consolidate_financial_sources_prefers_sec_and_fills_simfin_gaps() -> None:
    sec = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-02"],
            "value": [100.0],
            "source": ["sec_companyfacts"],
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
        }
    )
    simfin = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US"],
            "statement": ["income_statement", "income_statement"],
            "metric": ["revenue", "revenue"],
            "date": ["2025-03-31", "2025-06-30"],
            "filing_date": ["2025-05-02", "2025-08-01"],
            "value": [101.0, 110.0],
            "source": ["simfin", "simfin"],
            "source_label": ["Revenue", "Revenue"],
            "form": [None, None],
            "fiscal_period": ["Q1", "Q2"],
            "fiscal_year": [2025, 2025],
        }
    )

    consolidated, lineage, summary = consolidate_financial_sources(
        [
            FinancialSourceInput("sec_companyfacts", sec, priority=1),
            FinancialSourceInput("simfin", simfin, priority=2),
        ]
    )

    assert consolidated.height == 2
    assert consolidated["source"].unique().to_list() == ["open_source_consolidated"]
    assert consolidated.sort("date")["date"].to_list() == ["2025-03-31", "2025-06-30"]
    assert consolidated.sort("date")["value"].to_list() == [100.0, 110.0]
    assert consolidated.sort("date")["selected_source"].to_list() == ["sec_companyfacts", "simfin"]
    assert consolidated.sort("date")["fallback_used"].to_list() == [False, True]
    assert lineage.height == 3
    assert summary.filter(pl.col("selected_source") == "simfin")["selected_rows"].item() == 1


def test_consolidate_financial_sources_handles_missing_lineage_columns() -> None:
    sec = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-02"],
            "value": [100.0],
            "source": ["sec_companyfacts"],
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
        }
    )
    yfinance = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-06-30"],
            "filing_date": [None],
            "value": [110.0],
            "source": ["yfinance"],
            "source_label": ["Total Revenue"],
        }
    )

    consolidated, lineage, _ = consolidate_financial_sources(
        [
            FinancialSourceInput("sec_companyfacts", sec, priority=1),
            FinancialSourceInput("yfinance", yfinance, priority=3),
        ]
    )

    assert consolidated.height == 2
    assert consolidated.sort("date")["selected_source"].to_list() == ["sec_companyfacts", "yfinance"]
    assert consolidated.sort("date")["candidate_source_count"].to_list() == [1, 1]
    assert lineage.filter(pl.col("source") == "yfinance")["selected_form"].to_list() == [None]


def test_identify_yfinance_financial_fallback_tickers_returns_only_incomplete_sec_tickers() -> None:
    quarter_dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    rows: list[dict[str, object]] = []
    for spec in [spec for spec in METRIC_SPECS if spec.statement != "earnings" and spec.yfinance_rows]:
        for date, period in zip(quarter_dates, ["Q1", "Q2", "Q3", "Q4"], strict=True):
            rows.append(
                {
                    "ticker": "AAPL.US",
                    "statement": spec.statement,
                    "metric": spec.metric,
                    "date": date,
                    "filing_date": None,
                    "value": 1.0,
                    "source": "sec_companyfacts",
                    "source_label": spec.sec_tags[0] if spec.sec_tags else "derived",
                    "form": "10-Q",
                    "fiscal_period": period,
                    "fiscal_year": 2025,
                }
            )
    for date, period in zip(quarter_dates[:3], ["Q1", "Q2", "Q3"], strict=True):
        rows.append(
            {
                "ticker": "MSFT.US",
                "statement": "income_statement",
                "metric": "revenue",
                "date": date,
                "filing_date": None,
                "value": 1.0,
                "source": "sec_companyfacts",
                "source_label": "Revenue",
                "form": "10-Q",
                "fiscal_period": period,
                "fiscal_year": 2025,
            }
        )
    sec = pl.DataFrame(rows)

    fallback = _identify_yfinance_financial_fallback_tickers(
        tickers=("AAPL", "MSFT", "NVDA"),
        sec_companyfacts=sec,
        sec_filing=pl.DataFrame(schema=sec.schema),
    )

    assert fallback == ("MSFT", "NVDA")


def test_identify_sec_filing_fallback_tickers_targets_incomplete_companyfacts() -> None:
    rows = [
        {
            "ticker": "AAPL.US",
            "statement": "income_statement",
            "metric": "revenue",
            "date": date,
            "filing_date": "2025-05-01",
            "value": 1.0,
            "source": "sec_companyfacts",
            "source_label": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "form": "10-Q",
            "fiscal_period": period,
            "fiscal_year": 2025,
        }
        for date, period in zip(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"], ["Q1", "Q2", "Q3", "Q4"], strict=True)
    ]
    sec = pl.DataFrame(rows)

    fallback = _identify_sec_filing_fallback_tickers(("AAPL", "MSFT"), sec)

    assert fallback == ("AAPL", "MSFT")


def test_parse_contexts_marks_dimensional_contexts() -> None:
    xml = """
    <xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
      <context id="c1">
        <entity><identifier scheme="http://www.sec.gov/CIK">1</identifier></entity>
        <period><startDate>2025-01-01</startDate><endDate>2025-03-31</endDate></period>
      </context>
      <context id="c2">
        <entity>
          <identifier scheme="http://www.sec.gov/CIK">1</identifier>
          <segment><xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">foo:Bar</xbrldi:explicitMember></segment>
        </entity>
        <period><instant>2025-03-31</instant></period>
      </context>
    </xbrl>
    """
    contexts = _parse_contexts(ET.fromstring(xml))

    assert contexts["c1"]["has_dimensions"] is False
    assert contexts["c1"]["end"] == "2025-03-31"
    assert contexts["c2"]["has_dimensions"] is True
    assert contexts["c2"]["instant"] == "2025-03-31"


def test_parse_document_focus_extracts_fiscal_year_and_period() -> None:
    xml = """
    <xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:dei="http://xbrl.sec.gov/dei/2024">
      <dei:DocumentFiscalYearFocus contextRef="c1">2025</dei:DocumentFiscalYearFocus>
      <dei:DocumentFiscalPeriodFocus contextRef="c1">Q2</dei:DocumentFiscalPeriodFocus>
    </xbrl>
    """
    root = ET.fromstring(xml)

    fiscal_year, fiscal_period = _parse_document_focus(root)

    assert fiscal_year == 2025
    assert fiscal_period == "Q2"


def test_derive_missing_total_liabilities_from_assets_and_equity() -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["OXY.US", "OXY.US"],
            "statement": ["balance_sheet", "balance_sheet"],
            "metric": ["total_assets", "stockholders_equity"],
            "date": ["2025-03-31", "2025-03-31"],
            "filing_date": ["2025-05-07", "2025-05-07"],
            "value": [84_967_000_000.0, 34_712_000_000.0],
            "source": ["sec_filing", "sec_filing"],
            "source_label": ["Assets", "StockholdersEquity"],
            "form": ["10-Q", "10-Q"],
            "fiscal_period": ["Q1", "Q1"],
            "fiscal_year": [2025, 2025],
        }
    )

    derived = _derive_missing_total_liabilities(frame).sort("metric")

    assert derived.height == 3
    assert derived.filter(pl.col("metric") == "total_liabilities")["value"].item() == 50_255_000_000.0
    assert derived.filter(pl.col("metric") == "total_liabilities")["source_label"].item() == "derived_from_assets_minus_stockholders_equity"
