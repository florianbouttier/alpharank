from __future__ import annotations

from alpharank.data.ingestion.acquisition_status import build_acquisition_status


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
