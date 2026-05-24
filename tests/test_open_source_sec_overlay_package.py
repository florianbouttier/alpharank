from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "open_source" / "build_sec_overlay_package.py"
SPEC = importlib.util.spec_from_file_location("build_sec_overlay_package", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

merge_earnings = MODULE.merge_earnings
merge_earnings_long = MODULE.merge_earnings_long
merge_financials = MODULE.merge_financials


def test_merge_financials_keeps_primary_and_fills_missing_quarter_from_secondary() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2024-03-31"],
            "filing_date": ["2024-05-01"],
            "value": [100.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q1"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["income_statement", "income_statement"],
            "metric": ["revenue", "revenue"],
            "date": ["2024-03-31", "2024-06-30"],
            "filing_date": ["2024-05-02", "2024-08-01"],
            "value": [999.0, 120.0],
            "selected_fiscal_year": [2024, 2024],
            "selected_fiscal_period": ["Q1", "Q2"],
        }
    )

    merged, merged_lineage, audit = merge_financials(
        primary_consolidated=primary,
        secondary_consolidated=secondary,
        primary_lineage=primary,
        secondary_lineage=secondary,
    )

    assert merged.height == 2
    assert merged.filter(pl.col("selected_fiscal_period") == "Q1")["value"].to_list() == [100.0]
    assert merged.filter(pl.col("selected_fiscal_period") == "Q2")["value"].to_list() == [120.0]
    assert merged_lineage.height == 2
    assert set(audit["overlay_origin"].to_list()) == {"primary_snapshot", "secondary_candidate"}


def test_merge_earnings_keeps_primary_and_fills_missing_quarter_from_secondary() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "period_end": ["2024-03-31"],
            "reportDate": ["2024-05-01"],
            "epsActual": [1.0],
            "fiscal_year": [2024],
            "fiscal_period": ["Q1"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "period_end": ["2024-03-31", "2024-06-30"],
            "reportDate": ["2024-05-02", "2024-08-01"],
            "epsActual": [9.0, 2.0],
            "fiscal_year": [2024, 2024],
            "fiscal_period": ["Q1", "Q2"],
        }
    )

    merged, merged_lineage, audit = merge_earnings(
        primary_consolidated=primary,
        secondary_consolidated=secondary,
        primary_lineage=primary,
        secondary_lineage=secondary,
    )

    assert merged.height == 2
    assert merged.filter(pl.col("fiscal_period") == "Q1")["epsActual"].to_list() == [1.0]
    assert merged.filter(pl.col("fiscal_period") == "Q2")["epsActual"].to_list() == [2.0]
    assert merged_lineage.height == 2
    assert set(audit["overlay_origin"].to_list()) == {"primary_snapshot", "secondary_candidate"}


def test_merge_financials_does_not_duplicate_same_date_with_conflicting_quarter_labels() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2024-09-30"],
            "filing_date": ["2024-11-01"],
            "value": [100.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q3"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2024-09-30"],
            "filing_date": ["2024-11-02"],
            "value": [101.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q4"],
        }
    )

    merged, merged_lineage, audit = merge_financials(
        primary_consolidated=primary,
        secondary_consolidated=secondary,
        primary_lineage=primary,
        secondary_lineage=secondary,
    )

    assert merged.height == 1
    assert merged["value"].to_list() == [100.0]
    assert merged_lineage.height == 1
    assert audit["rows"].sum() == 1


def test_merge_earnings_does_not_duplicate_same_period_end_with_conflicting_quarter_labels() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "period_end": ["2024-09-30"],
            "reportDate": ["2024-11-01"],
            "epsActual": [1.0],
            "fiscal_year": [2024],
            "fiscal_period": ["Q3"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "period_end": ["2024-09-30"],
            "reportDate": ["2024-11-02"],
            "epsActual": [1.1],
            "fiscal_year": [2024],
            "fiscal_period": ["Q4"],
        }
    )

    merged, merged_lineage, audit = merge_earnings(
        primary_consolidated=primary,
        secondary_consolidated=secondary,
        primary_lineage=primary,
        secondary_lineage=secondary,
    )

    assert merged.height == 1
    assert merged["epsActual"].to_list() == [1.0]
    assert merged_lineage.height == 1
    assert audit["rows"].sum() == 1


def test_merge_financials_preserves_existing_overlay_origins_across_multiple_secondaries() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2024-03-31"],
            "filing_date": ["2024-05-01"],
            "value": [100.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q1"],
        }
    )
    secondary_one = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2024-06-30"],
            "filing_date": ["2024-08-01"],
            "value": [120.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q2"],
        }
    )
    secondary_two = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2024-09-30"],
            "filing_date": ["2024-11-01"],
            "value": [140.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q3"],
        }
    )

    merged_one, merged_lineage_one, _ = merge_financials(
        primary_consolidated=primary,
        secondary_consolidated=secondary_one,
        primary_lineage=primary,
        secondary_lineage=secondary_one,
        secondary_origin="secondary_candidate:fix1",
    )
    merged_two, merged_lineage_two, audit_two = merge_financials(
        primary_consolidated=merged_one,
        secondary_consolidated=secondary_two,
        primary_lineage=merged_lineage_one,
        secondary_lineage=secondary_two,
        secondary_origin="secondary_candidate:fix2",
    )

    assert merged_two["selected_fiscal_period"].to_list() == ["Q1", "Q2", "Q3"]
    assert set(merged_lineage_two["overlay_origin"].to_list()) == {
        "primary_snapshot",
        "secondary_candidate:fix1",
        "secondary_candidate:fix2",
    }
    assert set(audit_two["overlay_origin"].to_list()) == {
        "primary_snapshot",
        "secondary_candidate:fix1",
        "secondary_candidate:fix2",
    }


def test_merge_earnings_long_fills_missing_quarter_metric_rows_from_secondary() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["earnings"],
            "metric": ["eps_actual"],
            "date": ["2024-03-31"],
            "filing_date": ["2024-05-01"],
            "value": [1.0],
            "selected_fiscal_year": [2024],
            "selected_fiscal_period": ["Q1"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["earnings", "earnings"],
            "metric": ["eps_actual", "eps_actual"],
            "date": ["2024-03-31", "2024-06-30"],
            "filing_date": ["2024-05-02", "2024-08-01"],
            "value": [9.0, 2.0],
            "selected_fiscal_year": [2024, 2024],
            "selected_fiscal_period": ["Q1", "Q2"],
        }
    )

    merged = merge_earnings_long(
        primary_long=primary,
        secondary_long=secondary,
        secondary_origin="secondary_candidate:fix1",
    )

    assert merged.height == 2
    assert merged.filter(pl.col("selected_fiscal_period") == "Q1")["value"].to_list() == [1.0]
    assert merged.filter(pl.col("selected_fiscal_period") == "Q2")["value"].to_list() == [2.0]
    assert set(merged["overlay_origin"].to_list()) == {"primary_snapshot", "secondary_candidate:fix1"}


def test_merge_earnings_long_falls_back_to_date_merge_when_fiscal_columns_are_missing() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["earnings"],
            "metric": ["eps_actual"],
            "date": ["2024-03-31"],
            "filing_date": ["2024-05-01"],
            "value": [1.0],
            "source": ["open_source_earnings"],
            "source_label": ["sec_companyfacts"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["earnings", "earnings"],
            "metric": ["eps_actual", "eps_actual"],
            "date": ["2024-03-31", "2024-06-30"],
            "filing_date": ["2024-05-02", "2024-08-01"],
            "value": [9.0, 2.0],
            "source": ["open_source_earnings", "open_source_earnings"],
            "source_label": ["sec_filing", "sec_filing"],
        }
    )

    merged = merge_earnings_long(
        primary_long=primary,
        secondary_long=secondary,
        secondary_origin="secondary_candidate:legacy",
    )

    assert merged.height == 2
    assert merged.filter(pl.col("date") == "2024-03-31")["value"].to_list() == [1.0]
    assert merged.filter(pl.col("date") == "2024-06-30")["value"].to_list() == [2.0]
    assert set(merged["overlay_origin"].to_list()) == {"primary_snapshot", "secondary_candidate:legacy"}
