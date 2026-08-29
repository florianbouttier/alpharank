"""Source-lined return overlays for one security observed under two tickers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    PRICE_LINEAGE_COLUMNS,
    PRICE_VALUE_COLUMNS,
)

PRICE_TICKER_TRANSITION_POLICY_ID = "price_ticker_transition_return_overlay_v1"
_DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "data_quality"
    / "price_ticker_transition_policy_v1.json"
)
_REGISTRY_COLUMNS = (
    "transition_id",
    "security_id",
    "issuer_cik",
    "provider_ticker",
    "target_ticker",
    "validated_anchor_date",
    "copy_from",
    "copy_through",
    "model_ticker_effective_from",
    "official_ticker_effective_from",
    "conversion_ratio",
    "required_overlap_rows",
    "maximum_overlap_return_delta",
    "maximum_anchor_relative_delta",
    "evidence_known_at",
    "evidence_url",
    "evidence_statement",
)


@dataclass(frozen=True, slots=True)
class PriceTickerTransitionResult:
    """Canonical lineage plus the exact rows derived from ticker aliases."""

    prices: pl.DataFrame
    lineage: pl.DataFrame
    audit: pl.DataFrame
    report: dict[str, object]


def load_price_ticker_transition_registry(path: Path | None = None) -> pl.DataFrame:
    """Load the strict versioned registry used by the price publication path."""

    registry_path = (path or _DEFAULT_REGISTRY).expanduser().resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_root = {"policy_id", "schema_version", "description", "transitions"}
    if set(payload) != expected_root:
        raise ValueError("Price ticker transition policy has unknown or missing root keys")
    if payload["policy_id"] != PRICE_TICKER_TRANSITION_POLICY_ID:
        raise ValueError(f"Unexpected price ticker transition policy: {payload['policy_id']}")
    if payload["schema_version"] != 1 or not isinstance(payload["transitions"], list):
        raise ValueError("Unsupported price ticker transition policy schema")
    frame = pl.DataFrame(payload["transitions"])
    prepared = _prepare_registry(frame).with_columns(
        pl.lit(str(registry_path)).alias("registry_path"),
        pl.lit(_sha256(registry_path)).alias("registry_sha256"),
    )
    _validate_registry(prepared)
    return prepared


def apply_price_ticker_transition_overlay(
    lineage: pl.DataFrame,
    *,
    registry: pl.DataFrame | None = None,
) -> PriceTickerTransitionResult:
    """Append missing target-ticker dates from a validated alias return ledger."""

    canonical = _normalize_lineage(lineage)
    transitions = _prepare_registry(
        registry if registry is not None else load_price_ticker_transition_registry()
    )
    _validate_registry(transitions)
    extensions: list[pl.DataFrame] = []
    audits: list[pl.DataFrame] = []
    transition_reports: list[dict[str, object]] = []
    for transition in transitions.to_dicts():
        extension, audit, report = _build_transition_extension(canonical, transition)
        if extension.height:
            extensions.append(extension)
            audits.append(audit)
        transition_reports.append(report)
    selected = _combine_lineage(canonical, extensions)
    _assert_additive_overlay(previous=canonical, selected=selected)
    audit = _combine_audits(audits)
    registry_paths = _unique_non_null(transitions, "registry_path")
    registry_hashes = _unique_non_null(transitions, "registry_sha256")
    report = {
        "policy_id": PRICE_TICKER_TRANSITION_POLICY_ID,
        "registry_path": registry_paths[0] if len(registry_paths) == 1 else None,
        "registry_sha256": registry_hashes[0] if len(registry_hashes) == 1 else None,
        "transition_count": transitions.height,
        "applied_transition_count": sum(item["added_rows"] > 0 for item in transition_reports),
        "added_rows": audit.height,
        "previous_rows_changed": 0,
        "manual_price_values": 0,
        "selection_rule": "validated_target_anchor_plus_same_security_provider_daily_returns",
        "transitions": transition_reports,
        "passed": True,
    }
    return PriceTickerTransitionResult(
        prices=selected.select(PRICE_VALUE_COLUMNS),
        lineage=selected,
        audit=audit,
        report=report,
    )


def _build_transition_extension(
    lineage: pl.DataFrame,
    transition: dict[str, object],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    provider_ticker = str(transition["provider_ticker"])
    target_ticker = str(transition["target_ticker"])
    anchor_date = str(transition["validated_anchor_date"])
    target = lineage.filter(pl.col("ticker") == target_ticker).sort("date")
    if target.is_empty() or target.select(pl.col("date").max()).item() < anchor_date:
        return lineage.clear(), _empty_audit(), _not_applicable_report(transition)
    provider = lineage.filter(pl.col("ticker") == provider_ticker).sort("date")
    _validate_transition_inputs(target=target, provider=provider, transition=transition)
    overlap_report = _validate_overlap(target=target, provider=provider, transition=transition)
    candidates = _derive_candidates(target=target, provider=provider, transition=transition)
    existing = target.filter(
        pl.col("date").is_between(
            pl.lit(str(transition["copy_from"])),
            pl.lit(str(transition["copy_through"])),
        )
    )
    _validate_existing_target_rows(
        existing=existing,
        candidates=candidates,
        transition=transition,
    )
    extension = candidates.join(existing.select("date"), on="date", how="anti")
    audit = _build_audit(extension, transition)
    report = {
        "transition_id": transition["transition_id"],
        "security_id": transition["security_id"],
        "provider_ticker": provider_ticker,
        "target_ticker": target_ticker,
        "validated_anchor_date": anchor_date,
        "copy_from": transition["copy_from"],
        "copy_through": transition["copy_through"],
        "existing_target_rows": existing.height,
        "provider_candidate_rows": candidates.height,
        "added_rows": extension.height,
        "status": "applied" if extension.height else "already_present",
        **overlap_report,
        "evidence_known_at": transition["evidence_known_at"],
        "evidence_url": transition["evidence_url"],
    }
    return extension.select(PRICE_LINEAGE_COLUMNS), audit, report


def _validate_transition_inputs(
    *,
    target: pl.DataFrame,
    provider: pl.DataFrame,
    transition: dict[str, object],
) -> None:
    anchor_date = str(transition["validated_anchor_date"])
    provider_ticker = str(transition["provider_ticker"])
    target_ticker = str(transition["target_ticker"])
    if provider.is_empty():
        raise RuntimeError(f"Ticker transition provider is missing: {provider_ticker}")
    for label, frame in (("target", target), ("provider", provider)):
        if frame.filter(pl.col("date") == anchor_date).height != 1:
            ticker = target_ticker if label == "target" else provider_ticker
            raise RuntimeError(
                f"Ticker transition {label} anchor is not unique: "
                f"ticker={ticker}, date={anchor_date}"
            )
    source_tail = provider.filter(
        pl.col("date").is_between(
            pl.lit(str(transition["copy_from"])),
            pl.lit(str(transition["copy_through"])),
        )
    )
    if source_tail.is_empty():
        raise RuntimeError(
            f"Ticker transition provider has no rows in the copy interval: {provider_ticker}"
        )


def _validate_overlap(
    *,
    target: pl.DataFrame,
    provider: pl.DataFrame,
    transition: dict[str, object],
) -> dict[str, object]:
    anchor_date = str(transition["validated_anchor_date"])
    overlap = (
        _returns(target, "target_return")
        .join(_returns(provider, "provider_return"), on="date", how="inner")
        .filter((pl.col("date") <= anchor_date) & pl.col("target_return").is_not_null())
        .tail(int(transition["required_overlap_rows"]))
        .with_columns(
            (pl.col("target_return") - pl.col("provider_return"))
            .abs()
            .alias("absolute_return_delta")
        )
    )
    required = int(transition["required_overlap_rows"])
    if overlap.height != required:
        raise RuntimeError(
            "Ticker transition lacks required overlapping returns: "
            f"transition={transition['transition_id']}, "
            f"observed={overlap.height}, required={required}"
        )
    max_return_delta = float(overlap.select(pl.col("absolute_return_delta").max()).item())
    if max_return_delta > float(transition["maximum_overlap_return_delta"]):
        raise RuntimeError(
            "Ticker transition overlap returns disagree: "
            f"transition={transition['transition_id']}, maximum_delta={max_return_delta}"
        )
    target_anchor = float(
        target.filter(pl.col("date") == anchor_date).select("adjusted_close").item()
    )
    provider_anchor = float(
        provider.filter(pl.col("date") == anchor_date).select("adjusted_close").item()
    )
    anchor_relative_delta = abs(target_anchor / provider_anchor - 1.0)
    if anchor_relative_delta > float(transition["maximum_anchor_relative_delta"]):
        raise RuntimeError(
            "Ticker transition anchors disagree: "
            f"transition={transition['transition_id']}, relative_delta={anchor_relative_delta}"
        )
    return {
        "validated_overlap_rows": overlap.height,
        "maximum_overlap_return_delta_observed": max_return_delta,
        "anchor_relative_delta_observed": anchor_relative_delta,
    }


def _derive_candidates(
    *,
    target: pl.DataFrame,
    provider: pl.DataFrame,
    transition: dict[str, object],
) -> pl.DataFrame:
    anchor_date = str(transition["validated_anchor_date"])
    anchor = target.filter(pl.col("date") == anchor_date).row(0, named=True)
    with_returns = provider.with_columns(
        pl.col("adjusted_close").pct_change().alias("provider_daily_return")
    )
    tail = with_returns.filter(
        (pl.col("date") > anchor_date) & (pl.col("date") <= str(transition["copy_through"]))
    )
    invalid = tail.filter(
        pl.col("provider_daily_return").is_null()
        | ~pl.col("provider_daily_return").is_finite()
        | (pl.col("provider_daily_return") <= -1.0)
    )
    if invalid.height:
        raise RuntimeError(
            f"Ticker transition provider return is unusable: {transition['transition_id']}"
        )
    anchor_adjusted = float(anchor["adjusted_close"])
    anchor_close = float(anchor["close"])
    anchor_factor = anchor_adjusted / anchor_close if anchor_close else 1.0
    scaled = tail.with_columns(
        pl.col("adjusted_close").alias("provider_adjusted_close"),
        (pl.lit(anchor_adjusted) * (pl.col("provider_daily_return") + 1.0).cum_prod()).alias(
            "selected_adjusted_close"
        ),
    ).with_columns((pl.col("selected_adjusted_close") / anchor_factor).alias("selected_close"))
    for column in ("open", "high", "low"):
        scaled = scaled.with_columns(
            pl.when(pl.col("close").is_not_null() & (pl.col("close") != 0.0))
            .then(pl.col(column) / pl.col("close") * pl.col("selected_close"))
            .otherwise(None)
            .alias(column)
        )
    return scaled.with_columns(
        pl.col("selected_close").alias("close"),
        pl.col("selected_adjusted_close").alias("adjusted_close"),
        pl.lit(transition["target_ticker"]).alias("ticker"),
        pl.lit("ticker_transition_return_ledger").alias("source"),
        pl.lit("prices_ticker_transition_return_overlay").alias("dataset"),
        pl.lit(transition["transition_id"]).alias("source_vintage_id"),
        pl.lit(ADJUSTMENT_POLICY_VERSION).alias("adjustment_policy_version"),
        (pl.col("selected_adjusted_close") / pl.col("provider_adjusted_close")).alias(
            "adjustment_bridge_factor"
        ),
        pl.lit(anchor.get("eodhd_seed_sha256")).cast(pl.String).alias("eodhd_seed_sha256"),
        pl.lit(transition["transition_id"]).alias("correction_overlay_id"),
    ).filter(pl.col("date") >= str(transition["copy_from"]))


def _validate_existing_target_rows(
    *,
    existing: pl.DataFrame,
    candidates: pl.DataFrame,
    transition: dict[str, object],
) -> None:
    comparison = existing.select("date", "adjusted_close").join(
        candidates.select(
            "date", pl.col("selected_adjusted_close").alias("candidate_adjusted_close")
        ),
        on="date",
        how="inner",
    )
    if comparison.is_empty():
        return
    max_delta = comparison.select(
        ((pl.col("adjusted_close") / pl.col("candidate_adjusted_close")) - 1.0).abs().max()
    ).item()
    if float(max_delta) > float(transition["maximum_anchor_relative_delta"]):
        raise RuntimeError(
            f"Existing target rows disagree with ticker transition: {transition['transition_id']}"
        )


def _build_audit(
    extension: pl.DataFrame,
    transition: dict[str, object],
) -> pl.DataFrame:
    if extension.is_empty():
        return _empty_audit()
    return extension.select(
        pl.lit(transition["transition_id"]).alias("transition_id"),
        pl.lit(transition["security_id"]).alias("security_id"),
        pl.lit(transition["issuer_cik"]).alias("issuer_cik"),
        pl.lit(transition["provider_ticker"]).alias("provider_ticker"),
        pl.lit(transition["target_ticker"]).alias("target_ticker"),
        pl.lit(transition["validated_anchor_date"]).alias("validated_anchor_date"),
        "date",
        "provider_daily_return",
        "provider_adjusted_close",
        pl.col("adjusted_close").alias("selected_adjusted_close"),
        "return_source_vintage_id",
        "adjustment_bridge_factor",
        pl.lit(transition["evidence_known_at"]).alias("evidence_known_at"),
        pl.lit(transition["evidence_url"]).alias("evidence_url"),
    )


def _prepare_registry(frame: pl.DataFrame) -> pl.DataFrame:
    missing = set(_REGISTRY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Price ticker transition registry is missing: {sorted(missing)}")
    prepared = frame.with_columns(
        pl.col("transition_id").cast(pl.String),
        pl.col("security_id").cast(pl.String),
        pl.col("issuer_cik").cast(pl.String).str.extract(r"(\d+)").str.zfill(10),
        pl.col("provider_ticker").cast(pl.String).str.to_uppercase(),
        pl.col("target_ticker").cast(pl.String).str.to_uppercase(),
        pl.col("validated_anchor_date").cast(pl.String),
        pl.col("copy_from").cast(pl.String),
        pl.col("copy_through").cast(pl.String),
        pl.col("model_ticker_effective_from").cast(pl.String),
        pl.col("official_ticker_effective_from").cast(pl.String),
        pl.col("conversion_ratio").cast(pl.Float64),
        pl.col("required_overlap_rows").cast(pl.Int64),
        pl.col("maximum_overlap_return_delta").cast(pl.Float64),
        pl.col("maximum_anchor_relative_delta").cast(pl.Float64),
        pl.col("evidence_known_at").cast(pl.String),
        pl.col("evidence_url").cast(pl.String),
        pl.col("evidence_statement").cast(pl.String),
    )
    for column in ("registry_path", "registry_sha256"):
        if column not in prepared.columns:
            prepared = prepared.with_columns(pl.lit(None).cast(pl.String).alias(column))
    return prepared


def _validate_registry(registry: pl.DataFrame) -> None:
    if registry.is_empty():
        return
    if registry.height != registry.select(pl.col("transition_id").n_unique()).item():
        raise ValueError("Price ticker transition ids must be unique")
    invalid = registry.filter(
        (pl.col("provider_ticker") == pl.col("target_ticker"))
        | (pl.col("conversion_ratio") != 1.0)
        | (pl.col("required_overlap_rows") < 2)
        | (pl.col("validated_anchor_date") >= pl.col("copy_from"))
        | (pl.col("copy_from") > pl.col("copy_through"))
        | (pl.col("copy_through") >= pl.col("model_ticker_effective_from"))
        | (pl.col("model_ticker_effective_from") > pl.col("official_ticker_effective_from"))
    )
    if invalid.height:
        raise ValueError("Price ticker transition registry contains an invalid interval")
    overlaps = (
        registry.sort("target_ticker", "copy_from")
        .with_columns(
            pl.col("copy_through").shift(1).over("target_ticker").alias("previous_through")
        )
        .filter(
            pl.col("previous_through").is_not_null()
            & (pl.col("copy_from") <= pl.col("previous_through"))
        )
    )
    if overlaps.height:
        raise ValueError("Price ticker transition target intervals overlap")


def _normalize_lineage(frame: pl.DataFrame) -> pl.DataFrame:
    missing = set(PRICE_LINEAGE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Price ticker transition lineage is missing: {sorted(missing)}")
    normalized = frame.select(PRICE_LINEAGE_COLUMNS).with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.String),
    )
    duplicate_count = (
        normalized.height - normalized.select(pl.struct(["ticker", "date"]).n_unique()).item()
    )
    if duplicate_count:
        raise ValueError(f"Price ticker transition lineage has {duplicate_count} duplicate keys")
    return normalized.sort(["ticker", "date"])


def _combine_lineage(
    previous: pl.DataFrame,
    extensions: list[pl.DataFrame],
) -> pl.DataFrame:
    if not extensions:
        return previous
    return pl.concat([previous, *extensions], how="diagonal_relaxed").sort(["ticker", "date"])


def _assert_additive_overlay(*, previous: pl.DataFrame, selected: pl.DataFrame) -> None:
    observed = selected.join(
        previous.select("ticker", "date"), on=["ticker", "date"], how="inner"
    ).select(PRICE_LINEAGE_COLUMNS)
    if observed.height != previous.height or not observed.equals(previous, null_equal=True):
        raise RuntimeError("Ticker transition overlay changed a previously validated price row")


def _returns(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        frame.sort("date")
        .with_columns(pl.col("adjusted_close").pct_change().alias(column))
        .select("date", column)
    )


def _combine_audits(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(frames, how="diagonal_relaxed") if frames else _empty_audit()


def _empty_audit() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "transition_id": pl.String,
            "security_id": pl.String,
            "issuer_cik": pl.String,
            "provider_ticker": pl.String,
            "target_ticker": pl.String,
            "validated_anchor_date": pl.String,
            "date": pl.String,
            "provider_daily_return": pl.Float64,
            "provider_adjusted_close": pl.Float64,
            "selected_adjusted_close": pl.Float64,
            "return_source_vintage_id": pl.String,
            "adjustment_bridge_factor": pl.Float64,
            "evidence_known_at": pl.String,
            "evidence_url": pl.String,
        }
    )


def _not_applicable_report(transition: dict[str, object]) -> dict[str, object]:
    return {
        "transition_id": transition["transition_id"],
        "security_id": transition["security_id"],
        "provider_ticker": transition["provider_ticker"],
        "target_ticker": transition["target_ticker"],
        "validated_anchor_date": transition["validated_anchor_date"],
        "copy_from": transition["copy_from"],
        "copy_through": transition["copy_through"],
        "existing_target_rows": 0,
        "provider_candidate_rows": 0,
        "added_rows": 0,
        "status": "not_applicable_before_anchor",
        "evidence_known_at": transition["evidence_known_at"],
        "evidence_url": transition["evidence_url"],
    }


def _unique_non_null(frame: pl.DataFrame, column: str) -> list[str]:
    return [str(value) for value in frame.get_column(column).drop_nulls().unique().to_list()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
