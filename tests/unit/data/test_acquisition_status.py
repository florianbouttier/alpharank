from __future__ import annotations

from alpharank.data.ingestion.acquisition_status import (
    build_acquisition_status,
    build_price_publication_guard,
)


def test_acquisition_status_separates_download_from_publication_gate() -> None:
    report = build_acquisition_status(
        run_id="run_1",
        source_rows={"yahoo_prices": 10, "sec_companyfacts": 4},
        source_failures={},
        price_gate_report={
            "passed": False,
            "blocking_reasons": ["unreviewed_historical_return_revisions"],
        },
    )

    statuses = {item["source"]: item["status"] for item in report["sources"]}
    assert report["phase"] == "all_declared_sources_attempted_before_publication_decision"
    assert statuses == {
        "yahoo_prices": "downloaded_quarantined",
        "sec_companyfacts": "downloaded",
    }
    assert not report["price_publication_gate_passed"]


def test_acquisition_status_names_reconciled_provider_revisions() -> None:
    report = build_acquisition_status(
        run_id="run_2",
        source_rows={"yahoo_prices": 10},
        source_failures={},
        price_gate_report={
            "passed": True,
            "blocking_reasons": [],
            "resolved_provider_blocking_reasons": [
                "unreviewed_historical_return_revisions"
            ],
        },
    )

    assert report["sources"][0]["status"] == "downloaded_revisions_reconciled"


def test_price_publication_guard_combines_deferred_price_controls() -> None:
    report = build_price_publication_guard(
        {
            "price_revision_guard": {
                "passed": True,
                "blocking_reasons": [],
                "resolved_provider_blocking_reasons": [
                    "unreviewed_historical_return_revisions"
                ],
            },
            "price_extreme_move_guard": {
                "passed": False,
                "blocking_reasons": ["unreviewed_extreme_adjusted_price_moves"],
            },
        }
    )

    assert report["revision_guard_passed"] is True
    assert report["extreme_move_guard_passed"] is False
    assert report["blocking_reasons"] == ["unreviewed_extreme_adjusted_price_moves"]
    assert report["passed"] is False
