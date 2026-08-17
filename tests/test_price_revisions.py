from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    PRICE_LINEAGE_COLUMNS,
)
from alpharank.data.prices.revisions import build_price_revision_package


def _lineage(adjusted_close: list[float]) -> pl.DataFrame:
    dates = [date(2020, 1, 2), date(2020, 1, 3)]
    return pl.DataFrame(
        {
            "date": dates,
            "open": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 101.0],
            "close": [100.0, 102.0],
            "volume": [1_000.0, 1_100.0],
            "adjusted_close": adjusted_close,
            "ticker": ["OLD.US", "OLD.US"],
            "source": ["eodhd_frozen_history", "eodhd_frozen_history"],
            "dataset": ["prices_eodhd_frozen_seed"] * 2,
            "ingestion_run_id": ["seed"] * 2,
            "ingested_at": ["2026-08-01T00:00:00Z"] * 2,
            "source_vintage_id": ["seed"] * 2,
            "return_source_vintage_id": ["seed"] * 2,
            "adjustment_policy_version": [ADJUSTMENT_POLICY_VERSION] * 2,
            "adjustment_bridge_factor": [1.0] * 2,
            "eodhd_seed_sha256": ["seed-hash"] * 2,
            "correction_overlay_id": [None, None],
        }
    ).select(PRICE_LINEAGE_COLUMNS)


def _event(revision_type: str = "vendor_correction") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "revision_id": [f"old-{revision_type}-v1"],
            "revision_type": [revision_type],
            "ticker": ["OLD.US"],
            "effective_date": [date(2020, 1, 2)],
            "known_at": ["2026-08-16T09:00:00Z"],
            "affected_from": [date(2020, 1, 2)],
            "affected_through": [date(2020, 1, 2)],
            "source": ["issuer"],
            "source_url": ["https://example.com/correction"],
            "reason": ["Reviewed historical adjustment"],
        }
    )


def test_price_revision_requires_new_vintage() -> None:
    previous = _lineage([100.0, 102.0])
    candidate = _lineage([50.0, 102.0])

    with pytest.raises(ValueError, match="new package vintage"):
        build_price_revision_package(
            previous_lineage=previous,
            candidate_lineage=candidate,
            revision_events=_event(),
            previous_vintage_id="prices-v1",
            new_vintage_id="prices-v1",
            package_known_at="2026-08-17T00:00:00Z",
        )


@pytest.mark.parametrize(
    "revision_type", ["stock_split", "cash_dividend", "vendor_correction"]
)
def test_reviewed_price_revision_creates_immutable_diff(
    revision_type: str,
) -> None:
    previous = _lineage([100.0, 102.0])
    candidate = _lineage([50.0, 102.0])

    package = build_price_revision_package(
        previous_lineage=previous,
        candidate_lineage=candidate,
        revision_events=_event(revision_type),
        previous_vintage_id="prices-v1",
        new_vintage_id="prices-v2",
        package_known_at="2026-08-17T00:00:00Z",
    )

    assert previous["adjusted_close"].to_list() == [100.0, 102.0]
    assert candidate["correction_overlay_id"].to_list() == [None, None]
    assert package.report["previous_vintage_id"] == "prices-v1"
    assert package.report["new_vintage_id"] == "prices-v2"
    assert package.report["changed_rows"] == 1
    assert package.report["revision_diff_required"] is True
    assert package.report["previous_lineage_sha256"] != package.report[
        "new_lineage_sha256"
    ]
    assert package.revision_diff["changed_columns"].to_list() == [
        ["adjusted_close"]
    ]
    assert package.revision_diff["revision_source_url"].to_list() == [
        "https://example.com/correction"
    ]
    assert package.lineage["correction_overlay_id"].to_list() == [
        f"old-{revision_type}-v1",
        None,
    ]


def test_price_revision_rejects_evidence_known_after_package() -> None:
    event = _event().with_columns(pl.lit("2026-08-18T00:00:00Z").alias("known_at"))

    with pytest.raises(ValueError, match="not known at package creation"):
        build_price_revision_package(
            previous_lineage=_lineage([100.0, 102.0]),
            candidate_lineage=_lineage([50.0, 102.0]),
            revision_events=event,
            previous_vintage_id="prices-v1",
            new_vintage_id="prices-v2",
            package_known_at="2026-08-17T00:00:00Z",
        )


def test_price_revision_rejects_unexplained_historical_change() -> None:
    unrelated = _event().with_columns(pl.lit("OTHER.US").alias("ticker"))

    with pytest.raises(ValueError, match="lacks reviewed revision evidence"):
        build_price_revision_package(
            previous_lineage=_lineage([100.0, 102.0]),
            candidate_lineage=_lineage([50.0, 102.0]),
            revision_events=unrelated,
            previous_vintage_id="prices-v1",
            new_vintage_id="prices-v2",
            package_known_at="2026-08-17T00:00:00Z",
        )
