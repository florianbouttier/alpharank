"""Fail-closed verdict for refresh attribution scenarios."""

from __future__ import annotations

from typing import Mapping, Sequence


def resolve_attribution_status(
    audit: Mapping[str, object],
    legacy: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    legacy_sec_to_full: Mapping[str, object],
    legacy_price_to_full: Mapping[str, object],
    boosting_sec_to_full: Mapping[str, object],
    boosting_price_to_full: Mapping[str, object],
    common_effects_additive: bool | None,
) -> dict[str, object]:
    """Upgrade a raw drift only when the controlled effects are exhaustive."""

    legacy_by_name = {str(row["scenario"]): row for row in legacy}
    predictions_by_name = {str(row["scenario"]): row for row in predictions}
    legacy_price_neutral = _legacy_is_identical(legacy_by_name["price_only"])
    legacy_sec_neutral = _legacy_is_identical(legacy_by_name["sec_only"])
    boosting_price_neutral = _prediction_is_identical(predictions_by_name["price_only"])
    boosting_sec_neutral = _prediction_is_identical(predictions_by_name["sec_only"])
    checks = {
        "legacy_family_effect_isolated": (
            (legacy_sec_neutral and _legacy_is_identical(legacy_price_to_full))
            or (legacy_price_neutral and _legacy_is_identical(legacy_sec_to_full))
        ),
        "boosting_family_effect_isolated": (
            (boosting_sec_neutral and _prediction_is_identical(boosting_price_to_full))
            or (boosting_price_neutral and _prediction_is_identical(boosting_sec_to_full))
        ),
        "common_effects_are_additive": common_effects_additive is True,
    }
    exhaustively_attributed = all(checks.values())
    audit_status = str(audit["status"])
    status = (
        "explained_data_drift"
        if audit_status == "unexplained_portfolio_drift" and exhaustively_attributed
        else audit_status
    )
    return {
        "status": status,
        "audit_status_before_ablations": audit_status,
        "exhaustively_attributed": exhaustively_attributed,
        "checks": checks,
    }


def _prediction_is_identical(comparison: Mapping[str, object]) -> bool:
    return all(
        _as_int(comparison[key]) == 0 for key in ("added_rows", "removed_rows", "any_changed_rows")
    )


def _legacy_is_identical(comparison: Mapping[str, object]) -> bool:
    return _as_int(comparison["total_position_events"]) == 0


def _as_int(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected integer-compatible value, got {type(value).__name__}")
    return int(value)
