"""Strict filing-to-feature availability boundaries."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import polars as pl

from alpharank.data.contracts.point_in_time import join_point_in_time_attributes


def load_filing_availability_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    validate_filing_availability_policy(policy)
    return policy


def validate_filing_availability_policy(policy: Mapping[str, Any]) -> None:
    if not str(policy.get("policy_id") or "").strip():
        raise ValueError("Filing availability policy requires policy_id.")
    ZoneInfo(str(policy.get("timezone") or ""))
    time.fromisoformat(str(policy.get("date_only_assumption") or ""))
    delay = float(policy.get("operational_delay_hours", -1))
    if delay < 0:
        raise ValueError("operational_delay_hours must be non-negative.")
    if policy.get("require_filing_version_id") is not True:
        raise ValueError("Filing version identifiers are mandatory.")


def materialize_feature_availability(
    feature_rows: pl.DataFrame,
    *,
    policy: Mapping[str, Any],
) -> pl.DataFrame:
    """Attach an auditable UTC ``available_at`` to long-form filing features."""

    validate_filing_availability_policy(policy)
    required = {"ticker", "feature_name", "value", "filing_date", "filing_version_id"}
    missing = sorted(required - set(feature_rows.columns))
    if missing:
        raise ValueError("Filing features are missing: " + ", ".join(missing))
    accepted_column = "accepted_at" if "accepted_at" in feature_rows.columns else None
    zone = ZoneInfo(str(policy["timezone"]))
    date_only_time = time.fromisoformat(str(policy["date_only_assumption"]))
    delay = timedelta(hours=float(policy["operational_delay_hours"]))
    rows: list[dict[str, Any]] = []
    for row in feature_rows.to_dicts():
        version = str(row.get("filing_version_id") or "").strip()
        if not version:
            raise ValueError("Every feature value requires filing_version_id.")
        accepted_raw = row.get(accepted_column) if accepted_column else None
        if accepted_raw:
            accepted = _parse_datetime(accepted_raw, default_zone=zone)
            availability_basis = "sec_acceptance_timestamp"
        else:
            filing_day = _parse_date(row.get("filing_date"))
            accepted = datetime.combine(filing_day, date_only_time, tzinfo=zone)
            availability_basis = "filing_date_end_of_day_fallback"
        output = dict(row)
        output["accepted_at_normalized"] = accepted.astimezone(timezone.utc)
        output["available_at"] = (accepted + delay).astimezone(timezone.utc)
        output["availability_basis"] = availability_basis
        output["availability_policy_id"] = str(policy["policy_id"])
        rows.append(output)
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["ticker", "feature_name", "available_at", "filing_version_id"]
    )


def select_features_at_decisions(
    decisions: pl.DataFrame,
    available_features: pl.DataFrame,
) -> pl.DataFrame:
    """Select only the latest filing version available at each feature decision."""

    required_decisions = {"ticker", "feature_name", "decision_at"}
    missing = sorted(required_decisions - set(decisions.columns))
    if missing:
        raise ValueError("Feature decisions are missing: " + ", ".join(missing))
    required_features = {
        "ticker",
        "feature_name",
        "value",
        "available_at",
        "filing_version_id",
        "availability_basis",
        "availability_policy_id",
    }
    missing = sorted(required_features - set(available_features.columns))
    if missing:
        raise ValueError("Available features are missing: " + ", ".join(missing))

    entity = "_ticker_feature_key"
    left = decisions.with_columns(
        pl.concat_str([pl.col("ticker"), pl.col("feature_name")], separator="\u001f").alias(entity)
    )
    right = available_features.with_columns(
        pl.concat_str([pl.col("ticker"), pl.col("feature_name")], separator="\u001f").alias(entity)
    )
    selected = join_point_in_time_attributes(
        left,
        right,
        entity_column=entity,
        decision_time_column="decision_at",
        effective_time_column="available_at",
        attribute_columns=(
            "value",
            "filing_version_id",
            "accepted_at_normalized",
            "availability_basis",
            "availability_policy_id",
        ),
    ).drop(entity)
    leaked = selected.filter(
        pl.col("available_at_selected").is_not_null()
        & (pl.col("available_at_selected") > pl.col("decision_at"))
    )
    if not leaked.is_empty():
        raise RuntimeError("A future filing feature reached a past decision.")
    return selected.sort(["decision_at", "ticker", "feature_name"])


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _parse_datetime(value: Any, *, default_zone: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_zone)
    return parsed
