"""Published economic-prefix comparison contract."""

from __future__ import annotations

import hashlib
from datetime import datetime
from io import BytesIO
from typing import Any

import polars as pl

from alpharank.governance_contracts.contracts import (
    APPROVED_NUMERIC_TOLERANCE,
    ECONOMIC_PREFIX_CONTRACT_VERSION,
    HOLDINGS_PREFIX_EXACT_COLUMNS,
    HOLDINGS_PREFIX_KEYS,
    HOLDINGS_PREFIX_NUMERIC_COLUMNS,
    MONTHLY_PREFIX_EXACT_COLUMNS,
    MONTHLY_PREFIX_KEYS,
    MONTHLY_PREFIX_NUMERIC_COLUMNS,
    EconomicPrefixError,
)


def compare_economic_prefix(
    *,
    reference_holdings: pl.DataFrame,
    candidate_holdings: pl.DataFrame,
    reference_monthly: pl.DataFrame,
    candidate_monthly: pl.DataFrame,
    through_holding_month: str | None = None,
    numeric_tolerance: float = APPROVED_NUMERIC_TOLERANCE,
    tolerance_justification: str | None = (
        "owner-approved floating serialization tolerance; structural decisions remain exact"
    ),
) -> dict[str, Any]:
    """Compare the already-published economic prefix of two portfolio packages.

    The reference calendar defines the prefix unless an earlier explicit cutoff
    is supplied. Candidate rows after that cutoff are ignored. Keys and
    decision-like fields are exact; approved numeric fields use one documented
    absolute tolerance.
    """

    tolerance = float(numeric_tolerance)
    if tolerance < 0.0:
        raise ValueError("numeric_tolerance must be non-negative.")
    if tolerance > 0.0 and not str(tolerance_justification or "").strip():
        raise ValueError("A positive numeric tolerance requires a justification.")

    reference_holdings_normalized = _normalize_economic_frame(
        reference_holdings,
        keys=HOLDINGS_PREFIX_KEYS,
        numeric_columns=HOLDINGS_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=HOLDINGS_PREFIX_EXACT_COLUMNS,
        label="reference holdings",
    )
    candidate_holdings_normalized = _normalize_economic_frame(
        candidate_holdings,
        keys=HOLDINGS_PREFIX_KEYS,
        numeric_columns=HOLDINGS_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=HOLDINGS_PREFIX_EXACT_COLUMNS,
        label="candidate holdings",
    )
    reference_monthly_normalized = _normalize_economic_frame(
        reference_monthly,
        keys=MONTHLY_PREFIX_KEYS,
        numeric_columns=MONTHLY_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=MONTHLY_PREFIX_EXACT_COLUMNS,
        label="reference monthly returns",
    )
    candidate_monthly_normalized = _normalize_economic_frame(
        candidate_monthly,
        keys=MONTHLY_PREFIX_KEYS,
        numeric_columns=MONTHLY_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=MONTHLY_PREFIX_EXACT_COLUMNS,
        label="candidate monthly returns",
    )
    reference_end = reference_monthly_normalized.get_column("holding_month").max()
    if reference_end is None:
        raise ValueError("Reference monthly returns are empty.")
    cutoff = (
        datetime.fromisoformat(through_holding_month).date()
        if through_holding_month is not None
        else reference_end
    )
    if cutoff > reference_end:
        raise ValueError("through_holding_month cannot extend beyond the reference prefix.")

    frames = {
        "holdings": (
            reference_holdings_normalized.filter(pl.col("holding_month") <= cutoff),
            candidate_holdings_normalized.filter(pl.col("holding_month") <= cutoff),
            HOLDINGS_PREFIX_KEYS,
            HOLDINGS_PREFIX_NUMERIC_COLUMNS,
            HOLDINGS_PREFIX_EXACT_COLUMNS,
        ),
        "monthly": (
            reference_monthly_normalized.filter(pl.col("holding_month") <= cutoff),
            candidate_monthly_normalized.filter(pl.col("holding_month") <= cutoff),
            MONTHLY_PREFIX_KEYS,
            MONTHLY_PREFIX_NUMERIC_COLUMNS,
            MONTHLY_PREFIX_EXACT_COLUMNS,
        ),
    }
    frame_reports: dict[str, Any] = {}
    for label, (reference, candidate, keys, numeric, exact) in frames.items():
        frame_reports[label] = _compare_economic_frame(
            reference=reference,
            candidate=candidate,
            keys=keys,
            numeric_columns=numeric,
            exact_candidates=exact,
            tolerance=tolerance,
        )

    passed = all(report["passed"] for report in frame_reports.values())
    return {
        "economic_prefix_contract_version": ECONOMIC_PREFIX_CONTRACT_VERSION,
        "through_holding_month": str(cutoff),
        "numeric_tolerance": tolerance,
        "tolerance_justification": tolerance_justification,
        "structural_comparison": "exact",
        "frames": frame_reports,
        "passed": passed,
    }


def require_stable_economic_prefix(**kwargs: Any) -> dict[str, Any]:
    """Return the comparison report or fail a neutral migration closed."""

    report = compare_economic_prefix(**kwargs)
    if not report["passed"]:
        failed = [label for label, frame in report["frames"].items() if not frame["passed"]]
        raise EconomicPrefixError(
            "Published economic prefix changed in supposedly neutral migration: "
            + ", ".join(failed)
        )
    return report


def _normalize_economic_frame(
    frame: pl.DataFrame,
    *,
    keys: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    exact_candidates: tuple[str, ...],
    label: str,
) -> pl.DataFrame:
    required = set(keys) | set(numeric_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    expressions: list[pl.Expr] = []
    for key in keys:
        if key.endswith("_month"):
            expressions.append(pl.col(key).cast(pl.Date, strict=False).alias(key))
        else:
            expressions.append(pl.col(key).cast(pl.String).alias(key))
    expressions.extend(
        pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric_columns
    )
    exact_columns = tuple(column for column in exact_candidates if column in frame.columns)
    expressions.extend(pl.col(column).alias(column) for column in exact_columns)
    normalized = frame.select(expressions).sort(keys)
    duplicate_count = normalized.height - normalized.select(pl.struct(keys).n_unique()).item()
    if duplicate_count:
        raise ValueError(f"{label} has {duplicate_count} duplicate economic keys.")
    if (
        normalized.select(pl.any_horizontal([pl.col(key).is_null() for key in keys]))
        .to_series()
        .any()
    ):
        raise ValueError(f"{label} contains null economic keys.")
    return normalized


def _compare_economic_frame(
    *,
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    keys: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    exact_candidates: tuple[str, ...],
    tolerance: float,
) -> dict[str, Any]:
    reference_keys = reference.select(keys)
    candidate_keys = candidate.select(keys)
    missing_keys = reference_keys.join(candidate_keys, on=keys, how="anti")
    unexpected_keys = candidate_keys.join(reference_keys, on=keys, how="anti")
    exact_columns = tuple(
        column
        for column in exact_candidates
        if column in reference.columns and column in candidate.columns
    )
    missing_exact_columns = sorted(
        (set(exact_candidates) & set(reference.columns)) - set(candidate.columns)
    )
    joined = reference.join(candidate, on=keys, how="inner", suffix="_candidate")

    numeric_report: dict[str, Any] = {}
    for column in numeric_columns:
        left = pl.col(column)
        right = pl.col(f"{column}_candidate")
        null_mismatch = joined.filter(left.is_null() != right.is_null()).height
        finite_pairs = joined.filter(left.is_finite() & right.is_finite())
        nonfinite_mismatch = joined.filter(
            left.is_not_null()
            & right.is_not_null()
            & (~left.is_finite() | ~right.is_finite())
            & left.eq_missing(right).not_()
        ).height
        maximum = (
            finite_pairs.select((left - right).abs().max()).item()
            if not finite_pairs.is_empty()
            else 0.0
        )
        maximum = float(maximum or 0.0)
        numeric_report[column] = {
            "maximum_absolute_difference": maximum,
            "null_mismatches": null_mismatch,
            "nonfinite_mismatches": nonfinite_mismatch,
            "passed": null_mismatch == 0 and nonfinite_mismatch == 0 and maximum <= tolerance,
        }

    exact_report: dict[str, Any] = {}
    for column in exact_columns:
        mismatches = joined.filter(pl.col(column).eq_missing(pl.col(f"{column}_candidate")).not_())
        exact_report[column] = {
            "mismatches": mismatches.height,
            "passed": mismatches.is_empty(),
        }
    for column in missing_exact_columns:
        exact_report[column] = {"mismatches": None, "passed": False, "missing": True}

    passed = (
        missing_keys.is_empty()
        and unexpected_keys.is_empty()
        and all(result["passed"] for result in numeric_report.values())
        and all(result["passed"] for result in exact_report.values())
    )
    return {
        "reference_rows": reference.height,
        "candidate_rows": candidate.height,
        "reference_sha256": _dataframe_sha256(reference),
        "candidate_sha256": _dataframe_sha256(candidate),
        "missing_keys": missing_keys.head(20).to_dicts(),
        "missing_key_count": missing_keys.height,
        "unexpected_keys": unexpected_keys.head(20).to_dicts(),
        "unexpected_key_count": unexpected_keys.height,
        "numeric_columns": numeric_report,
        "exact_columns": exact_report,
        "passed": passed,
    }


def _dataframe_sha256(frame: pl.DataFrame) -> str:
    buffer = BytesIO()
    frame.write_ipc(buffer, compression="uncompressed")
    return hashlib.sha256(buffer.getvalue()).hexdigest()
