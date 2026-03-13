import pandas as pd
import polars as pl

from alpharank.data.open_source.benchmark import (
    build_error_summary_tables,
    build_financial_alignment,
    build_price_alignment,
)
from alpharank.data.open_source.sec import _select_best_facts
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
    assert result["adjusted_close_diff"].to_list() == [0.5, -1.0]


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
    assert result["filing_date_diff_days"].item() == 1


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

    selected = _select_best_facts(
        "income_statement",
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
        facts,
    )

    assert len(selected) == 1
    assert selected[0]["val"] == 10.0


def test_build_error_summary_tables_applies_threshold_pct() -> None:
    price_alignment = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "AAPL.US"],
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "match_status": ["matched", "matched", "eodhd_only"],
            "adjusted_close_diff_bps": [20.0, 80.0, None],
        }
    )
    financial_alignment = pl.DataFrame(
        {
            "open_source": ["yfinance", "yfinance", "sec_companyfacts"],
            "statement": ["income_statement", "income_statement", "balance_sheet"],
            "metric": ["revenue", "revenue", "total_assets"],
            "match_status": ["matched", "open_only", "matched"],
            "value_diff_bps": [60.0, None, 10.0],
        }
    )

    price_summary, statement_summary, metric_summary = build_error_summary_tables(
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        threshold_pct=0.5,
    )

    assert price_summary["error_rows"].item() == 1
    assert price_summary["error_rate_pct"].item() == 50.0
    assert statement_summary.filter(pl.col("source") == "yfinance")["error_rows"].item() == 1
    assert metric_summary.filter(pl.col("metric") == "revenue")["error_rows"].item() == 1
