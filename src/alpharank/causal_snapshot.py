"""Compatibility facade for causal replay snapshot contracts."""

from __future__ import annotations

from alpharank.replay.causal_snapshot import (
    CAUSAL_SNAPSHOT_CONTRACT_VERSION,
    CAUSAL_SNAPSHOT_MANIFEST_NAME,
    CAUSAL_SNAPSHOT_SEAL_NAME,
    REQUIRED_CRITICAL_FILES,
    REQUIRED_POLICY_FILES,
    CausalSnapshotValidationError,
    seal_causal_v2_snapshot,
    validate_causal_v2_snapshot,
)

__all__ = [
    "CAUSAL_SNAPSHOT_CONTRACT_VERSION",
    "CAUSAL_SNAPSHOT_MANIFEST_NAME",
    "CAUSAL_SNAPSHOT_SEAL_NAME",
    "REQUIRED_POLICY_FILES",
    "REQUIRED_CRITICAL_FILES",
    "CausalSnapshotValidationError",
    "seal_causal_v2_snapshot",
    "validate_causal_v2_snapshot",
]
