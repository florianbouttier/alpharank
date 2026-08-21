from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.quality.fundamental_quality import (
    audit_fundamental_quality,
    quarantine_implausible_share_candidates,
    validate_fundamental_quality,
)


def test_fundamental_quality_rejects_share_unit_discontinuity() -> None:
    financials = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US"],
            "statement": ["shares", "shares"],
            "metric": ["outstanding_shares", "outstanding_shares"],
            "date": ["2009-07-31", "2009-10-31"],
            "value": [344_000_000.0, 344.0],
            "selected_source": ["sec_companyfacts", "sec_companyfacts"],
        }
    )

    report = audit_fundamental_quality(financials)

    assert report["share_scale_discontinuity_count"] == 1
    with pytest.raises(RuntimeError, match="share_scale_discontinuities=1"):
        validate_fundamental_quality(report)


def test_fundamental_quality_accepts_large_but_plausible_split() -> None:
    financials = pl.DataFrame(
        {
            "ticker": ["TEST.US", "TEST.US"],
            "statement": ["shares", "shares"],
            "metric": ["outstanding_shares", "outstanding_shares"],
            "date": ["2025-03-31", "2025-06-30"],
            "value": [2_000_000.0, 200_000_000.0],
        }
    )

    report = audit_fundamental_quality(financials)

    assert report["quality_failures_detected"] is False
    validate_fundamental_quality(report)


def test_share_candidate_quarantine_preserves_raw_and_excludes_unit_errors() -> None:
    source = pl.DataFrame(
        {
            "ticker": ["A.US"] * 4,
            "statement": ["shares"] * 4,
            "metric": ["outstanding_shares"] * 4,
            "date": ["2009-03-31", "2009-06-30", "2009-09-30", "2009-12-31"],
            "filing_date": ["2009-05-01", "2009-08-01", "2009-11-01", "2010-02-01"],
            "value": [344_000_000.0, 344.0, 350_000_000.0, 350_000_000_000_000.0],
            "source": ["sec_companyfacts"] * 4,
            "source_label": ["EntityCommonStockSharesOutstanding"] * 4,
        }
    )

    cleaned, report = quarantine_implausible_share_candidates(source)

    assert source.height == 4
    assert cleaned["value"].to_list() == [344_000_000.0, 350_000_000.0]
    assert report["quarantined_rows"] == 2


def test_share_candidate_quarantine_has_no_false_examples_with_null_filing_dates() -> None:
    source = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US"],
            "statement": ["shares", "shares"],
            "metric": ["outstanding_shares", "outstanding_shares"],
            "date": ["2025-03-31", "2025-06-30"],
            "filing_date": [None, None],
            "value": [300_000_000.0, 301_000_000.0],
            "source": ["yfinance", "yfinance"],
            "source_label": ["Ordinary Shares Number", "Ordinary Shares Number"],
        }
    )

    _, report = quarantine_implausible_share_candidates(source)

    assert report["quarantined_rows"] == 0
    assert report["quarantined_examples"] == []
