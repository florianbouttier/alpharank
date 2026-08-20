"""Compatibility facade for sealed recomputable replay packages."""

from __future__ import annotations

from alpharank.replay.validation import (
    REPLAY_CONTRACT_VERSION,
    REPLAY_MANIFEST_NAME,
    REPLAY_SEAL_NAME,
    ReplayArtifact,
    ReplayValidationError,
    create_recomputable_replay_package,
    default_replay_code_paths,
    validate_and_recompute_replay_package,
)

__all__ = [
    "REPLAY_CONTRACT_VERSION",
    "REPLAY_MANIFEST_NAME",
    "REPLAY_SEAL_NAME",
    "ReplayValidationError",
    "ReplayArtifact",
    "default_replay_code_paths",
    "create_recomputable_replay_package",
    "validate_and_recompute_replay_package",
]
