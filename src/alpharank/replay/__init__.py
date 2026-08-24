"""Public contracts for causal, comparable, and recomputable replays."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CausalSnapshotValidationError",
    "CommonStrategyReplayConfig",
    "ReplayValidationError",
    "build_common_strategy_replay",
    "build_common_v2_comparison",
    "build_v1_v2_reconciliation",
    "create_recomputable_replay_package",
    "seal_causal_v2_snapshot",
    "validate_and_recompute_replay_package",
    "validate_boosting_v2_replay",
    "validate_causal_v2_snapshot",
    "validate_common_v2_replay",
    "validate_legacy_v2_replay",
    "validate_v1_v2_reconciliation",
]

_PUBLIC_OWNERS = {
    "CausalSnapshotValidationError": "alpharank.replay.causal_snapshot",
    "CommonStrategyReplayConfig": "alpharank.replay.common_strategy",
    "ReplayValidationError": "alpharank.replay.validation",
    "build_common_strategy_replay": "alpharank.replay.common_strategy",
    "build_common_v2_comparison": "alpharank.replay.common",
    "build_v1_v2_reconciliation": "alpharank.replay.reconciliation",
    "create_recomputable_replay_package": "alpharank.replay.validation",
    "seal_causal_v2_snapshot": "alpharank.replay.causal_snapshot",
    "validate_and_recompute_replay_package": "alpharank.replay.validation",
    "validate_boosting_v2_replay": "alpharank.replay.boosting",
    "validate_causal_v2_snapshot": "alpharank.replay.causal_snapshot",
    "validate_common_v2_replay": "alpharank.replay.common",
    "validate_legacy_v2_replay": "alpharank.replay.legacy",
    "validate_v1_v2_reconciliation": "alpharank.replay.reconciliation",
}


def __getattr__(name: str) -> Any:
    owner = _PUBLIC_OWNERS.get(name)
    if owner is None:
        raise AttributeError(f"module 'alpharank.replay' has no attribute {name!r}")
    return getattr(import_module(owner), name)
