"""Compatibility facade for the common causal replay contract."""

from __future__ import annotations

from alpharank.replay.common import (
    COMMON_V2_TOLERANCE,
    build_common_v2_comparison,
    gate_boosting_predictions_for_execution_open,
    gate_boosting_predictions_for_holding_membership,
    gate_boosting_predictions_for_pre_execution_blocks,
    standard_v2_cost_model,
    validate_common_v2_replay,
)

__all__ = [
    "COMMON_V2_TOLERANCE",
    "standard_v2_cost_model",
    "gate_boosting_predictions_for_holding_membership",
    "gate_boosting_predictions_for_execution_open",
    "gate_boosting_predictions_for_pre_execution_blocks",
    "build_common_v2_comparison",
    "validate_common_v2_replay",
]
