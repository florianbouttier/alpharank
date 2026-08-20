from __future__ import annotations

import importlib

import pytest

ROOT_REPLAY_APIS = {
    "alpharank.boosting_v2": ("validate_boosting_v2_replay",),
    "alpharank.causal_snapshot": (
        "CAUSAL_SNAPSHOT_CONTRACT_VERSION",
        "CAUSAL_SNAPSHOT_MANIFEST_NAME",
        "CAUSAL_SNAPSHOT_SEAL_NAME",
        "REQUIRED_POLICY_FILES",
        "REQUIRED_CRITICAL_FILES",
        "CausalSnapshotValidationError",
        "seal_causal_v2_snapshot",
        "validate_causal_v2_snapshot",
    ),
    "alpharank.common_v2": (
        "COMMON_V2_TOLERANCE",
        "standard_v2_cost_model",
        "gate_boosting_predictions_for_holding_membership",
        "gate_boosting_predictions_for_execution_open",
        "gate_boosting_predictions_for_pre_execution_blocks",
        "build_common_v2_comparison",
        "validate_common_v2_replay",
    ),
    "alpharank.legacy_v2": (
        "LEGACY_V2_TOLERANCE",
        "HOLDING_MONTH_MEMBERSHIP_POLICY_ID",
        "require_holding_month_membership",
        "validate_legacy_v2_replay",
    ),
    "alpharank.reconciliation_v2": (
        "RECONCILIATION_TOLERANCE",
        "build_v1_v2_reconciliation",
        "reconcile_economic_frames",
        "validate_v1_v2_reconciliation",
    ),
    "alpharank.replay_validation": (
        "REPLAY_CONTRACT_VERSION",
        "REPLAY_MANIFEST_NAME",
        "REPLAY_SEAL_NAME",
        "ReplayValidationError",
        "ReplayArtifact",
        "default_replay_code_paths",
        "create_recomputable_replay_package",
        "validate_and_recompute_replay_package",
    ),
}

OWNER_MODULES = {
    "alpharank.boosting_v2": "alpharank.replay.boosting",
    "alpharank.causal_snapshot": "alpharank.replay.causal_snapshot",
    "alpharank.common_v2": "alpharank.replay.common",
    "alpharank.legacy_v2": "alpharank.replay.legacy",
    "alpharank.reconciliation_v2": "alpharank.replay.reconciliation",
    "alpharank.replay_validation": "alpharank.replay.validation",
}


@pytest.mark.parametrize(("module_name", "public_names"), ROOT_REPLAY_APIS.items())
def test_historical_replay_modules_expose_characterized_api(
    module_name: str,
    public_names: tuple[str, ...],
) -> None:
    module = importlib.import_module(module_name)

    assert all(hasattr(module, name) for name in public_names)
    assert tuple(module.__all__) == public_names
    owner = importlib.import_module(OWNER_MODULES[module_name])
    assert all(getattr(module, name) is getattr(owner, name) for name in public_names)
