from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.financial_audit import build_financial_statement_audit_dashboard


def test_build_financial_statement_audit_dashboard_writes_expected_outputs(tmp_path: Path) -> None:
    eodhd_dir = tmp_path / "eodhd_output"
    open_dir = tmp_path / "open_output"
    lineage_dir = open_dir / "lineage"
    out_dir = tmp_path / "audit"
    eodhd_dir.mkdir()
    lineage_dir.mkdir(parents=True)

    income_eodhd = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "BBB.US"],
            "date": ["2020-03-31", "2020-06-30", "2020-03-31"],
            "filing_date": ["2020-05-01", "2020-08-01", "2020-05-03"],
            "totalRevenue": [100.0, 200.0, 300.0],
        }
    )
    income_open = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "date": ["2020-03-31", "2020-06-30"],
            "filing_date": ["2020-05-01", "2020-08-01"],
            "totalRevenue": [101.5, 200.0],
        }
    )
    share_eodhd = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "date": ["2020-Q1"],
            "dateFormatted": ["2020-03-31"],
            "shares": [10.0],
        }
    )
    share_open = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "date": ["2020-03-31"],
            "dateFormatted": ["2020-03-31"],
            "shares": [10.0],
        }
    )
    lineage = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "AAA.US"],
            "statement": ["income_statement", "income_statement", "shares"],
            "metric": ["revenue", "revenue", "outstanding_shares"],
            "date": ["2020-03-31", "2020-06-30", "2020-03-31"],
            "selected_source": ["sec_companyfacts", "sec_companyfacts", "sec_filing"],
            "selected_source_label": ["Revenue", "Revenue", "CommonStockSharesOutstanding"],
            "candidate_sources": ["sec_companyfacts", "sec_companyfacts", "sec_filing"],
            "selected_fiscal_period": ["Q1", "Q2", "Q1"],
            "selected_fiscal_year": [2020, 2020, 2020],
        }
    )

    income_eodhd.write_parquet(eodhd_dir / "US_Income_statement.parquet")
    income_open.write_parquet(open_dir / "US_Income_statement.parquet")
    share_eodhd.write_parquet(eodhd_dir / "US_share.parquet")
    share_open.write_parquet(open_dir / "US_share.parquet")
    lineage.write_parquet(lineage_dir / "financials_open_source_consolidated.parquet")

    for file_name in ["US_Balance_sheet.parquet", "US_Cash_flow.parquet"]:
        pl.DataFrame({"ticker": [], "date": [], "filing_date": []}).write_parquet(eodhd_dir / file_name)
        pl.DataFrame({"ticker": [], "date": [], "filing_date": []}).write_parquet(open_dir / file_name)

    result = build_financial_statement_audit_dashboard(
        eodhd_dir=eodhd_dir,
        open_source_dir=open_dir,
        output_dir=out_dir,
        start_date="2020-01-01",
        end_date="2020-12-31",
        threshold_pct=1.0,
    )

    assert result.dashboard_path.exists()
    assert result.summary_md_path.exists()
    assert result.summary_json_path.exists()
    assert result.total_rows == 4
    assert result.matched_rows == 3
    assert result.error_rows == 1
    assert result.missing_open_rows == 1
    assert result.reference_rows == 4

    summary = result.summary_md_path.read_text()
    assert "Rows above threshold: `1`" in summary
    assert "Missing in open-source: `1`" in summary
    assert "SEC filed" in summary
    assert "Vendor / non-SEC" in summary

    dashboard_html = result.dashboard_path.read_text()
    assert "Source Mode" in dashboard_html
    assert "Vendor / non-SEC" in dashboard_html

    alignment = pl.read_parquet(result.alignment_path)
    aaa_revenue = alignment.filter(
        (pl.col("ticker") == "AAA.US")
        & (pl.col("statement") == "income_statement")
        & (pl.col("metric") == "revenue")
        & (pl.col("date") == "2020-03-31")
    ).row(0, named=True)
    assert aaa_revenue["selected_source"] == "sec_companyfacts"
    assert aaa_revenue["selected_source_group"] == "sec"
    assert aaa_revenue["selected_source_group_label"] == "SEC filed"
    assert aaa_revenue["issue_kind"] == "error_gt_threshold"
