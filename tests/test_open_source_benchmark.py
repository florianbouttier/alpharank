import pandas as pd
import polars as pl

from alpharank.data.open_source.benchmark import (
    _build_financial_comparison_table,
    _build_price_comparison_table,
    build_audited_metric_catalog,
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
    )

    assert catalog.filter((pl.col("statement") == "income_statement") & (pl.col("metric") == "net_income") & (pl.col("source") == "sec_companyfacts")).height == 1
    assert catalog.filter(pl.col("source") == "yfinance_earnings").is_empty()
    assert catalog.filter((pl.col("statement") == "price") & (pl.col("metric") == "adjusted_close") & (pl.col("source") == "yfinance")).height == 1


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
