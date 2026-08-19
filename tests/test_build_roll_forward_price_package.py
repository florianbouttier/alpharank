from __future__ import annotations

import polars as pl
import pytest

from scripts.open_source.build_roll_forward_price_package import (
    _resolve_active_resolution_vintage_id,
)


def test_builder_binds_audited_carries_to_full_ingestion_run() -> None:
    fresh_yahoo = pl.DataFrame(
        {
            "ticker": ["AVB.US", "AVB.US"],
            "ingestion_run_id": ["20260816_103942", "20260819_220746"],
        }
    )

    assert (
        _resolve_active_resolution_vintage_id(
            base_manifest={"run_id": "20260819_220746"},
            fresh_yahoo=fresh_yahoo,
        )
        == "20260819_220746"
    )


def test_builder_rejects_a_fresh_vintage_without_current_run_observation() -> None:
    fresh_yahoo = pl.DataFrame(
        {
            "ticker": ["AVB.US"],
            "ingestion_run_id": ["20260816_103942"],
        }
    )

    with pytest.raises(RuntimeError, match="no observation"):
        _resolve_active_resolution_vintage_id(
            base_manifest={"run_id": "20260819_220746"},
            fresh_yahoo=fresh_yahoo,
        )
