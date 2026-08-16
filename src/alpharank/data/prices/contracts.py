from __future__ import annotations

from dataclasses import asdict, dataclass


PRICE_VALUE_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "ticker",
)

PRICE_LINEAGE_COLUMNS = (
    *PRICE_VALUE_COLUMNS,
    "source",
    "dataset",
    "ingestion_run_id",
    "ingested_at",
    "source_vintage_id",
    "return_source_vintage_id",
    "adjustment_policy_version",
    "adjustment_bridge_factor",
    "eodhd_seed_sha256",
    "correction_overlay_id",
)

ADJUSTMENT_POLICY_VERSION = "hybrid_price_adjustment_v1"
EODHD_SOURCE = "eodhd_frozen_history"
EODHD_DATASET = "prices_eodhd_frozen_seed"


@dataclass(frozen=True)
class PriceGatePolicy:
    """Fail-closed thresholds for a candidate canonical price package."""

    historical_return_revision_threshold: float = 0.0001
    transition_factor_jump_threshold: float = 0.0001
    recent_mutable_calendar_days: int = 7
    minimum_bridge_overlap_rows: int = 5
    bridge_factor_relative_tolerance: float = 0.001
    maximum_bridge_gap_calendar_days: int = 10
    allow_historical_price_revisions: bool = False
    allow_historical_price_key_removals: bool = False

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


PRODUCTION_PRICE_GATE_POLICY = PriceGatePolicy()
