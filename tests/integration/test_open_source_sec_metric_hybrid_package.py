from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.utils.module_loading import load_module_from_path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "open_source"
_HYBRID_PACKAGE_MODULE = load_module_from_path(
    "alpharank_test_build_sec_metric_hybrid_package",
    SCRIPT_DIR / "build_sec_metric_hybrid_package.py",
)
merge_earnings_prefer_non_null_actuals = (
    _HYBRID_PACKAGE_MODULE.merge_earnings_prefer_non_null_actuals
)


def test_merge_earnings_prefer_non_null_actuals_replaces_null_primary_quarter() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "period_end": ["2023-03-31", "2023-06-30"],
            "reportDate": ["2023-05-01", "2023-08-01"],
            "earningsDatetime": [None, None],
            "epsActual": [None, 1.2],
            "epsEstimate": [None, None],
            "surprisePercent": [None, None],
            "selected_source": ["sec_companyfacts", "sec_companyfacts"],
            "candidate_sources": [None, None],
            "calendar_source": ["sec", "sec"],
            "actual_source": [None, "sec_companyfacts"],
            "estimate_source": [None, None],
            "surprise_source": [None, None],
            "source_label": ["sec_companyfacts", "sec_companyfacts"],
            "accession_number": ["a1", "a2"],
            "form": ["10-Q", "10-Q"],
            "fiscal_period": ["Q1", "Q2"],
            "fiscal_year": [2023, 2023],
            "overlay_origin": ["primary", "primary"],
        }
    )
    secondary = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "period_end": ["2023-03-31", "2023-09-30"],
            "reportDate": ["2023-05-02", "2023-11-01"],
            "earningsDatetime": [None, None],
            "epsActual": [0.9, 1.5],
            "epsEstimate": [None, None],
            "surprisePercent": [None, None],
            "selected_source": ["sec_filing", "sec_filing"],
            "candidate_sources": [None, None],
            "calendar_source": ["sec", "sec"],
            "actual_source": ["sec_filing", "sec_filing"],
            "estimate_source": [None, None],
            "surprise_source": [None, None],
            "source_label": ["sec_filing", "sec_filing"],
            "accession_number": ["b1", "b2"],
            "form": ["10-Q", "10-Q"],
            "fiscal_period": ["Q1", "Q3"],
            "fiscal_year": [2023, 2023],
            "overlay_origin": ["fallback", "fallback"],
        }
    )

    merged, merged_lineage, audit = merge_earnings_prefer_non_null_actuals(
        primary_consolidated=primary,
        secondary_consolidated=secondary,
        primary_lineage=primary,
        secondary_lineage=secondary,
    )

    assert merged.height == 3
    assert merged_lineage.height == 3
    assert merged.filter((pl.col("fiscal_period") == "Q1") & pl.col("epsActual").is_not_null()).height == 1
    assert merged.filter((pl.col("fiscal_period") == "Q1")).row(0, named=True)["selected_source"] == "sec_filing"
    assert merged.filter(pl.col("fiscal_period") == "Q3").row(0, named=True)["epsActual"] == 1.5
    assert set(audit.get_column("overlay_origin").to_list()) == {"fallback", "primary"}
