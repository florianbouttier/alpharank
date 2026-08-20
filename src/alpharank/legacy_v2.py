"""Compatibility facade for Legacy causal replay contracts."""

from __future__ import annotations

from alpharank.replay.legacy import (
    HOLDING_MONTH_MEMBERSHIP_POLICY_ID,
    LEGACY_V2_TOLERANCE,
    require_holding_month_membership,
    validate_legacy_v2_replay,
)

__all__ = [
    "LEGACY_V2_TOLERANCE",
    "HOLDING_MONTH_MEMBERSHIP_POLICY_ID",
    "require_holding_month_membership",
    "validate_legacy_v2_replay",
]
