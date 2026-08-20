"""Point-in-time security identities for symbols reused by different issuers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

SECURITY_IDENTITY_POLICY_ID = "security_identity_intervals_v1"
_REGISTRY_FILE = "security_identity_registry.csv"
_REQUIRED_COLUMNS = {
    "source_ticker",
    "canonical_ticker",
    "security_id",
    "issuer_cik",
    "valid_from",
    "valid_to",
    "identity_status",
    "evidence",
}


@dataclass(frozen=True)
class SecurityIdentityApplication:
    """A canonicalized frame plus rows rejected outside every identity window."""

    frame: pl.DataFrame
    rejected: pl.DataFrame
    report: dict[str, object]


def load_security_identity_registry(path: Path | None = None) -> pl.DataFrame:
    """Load and validate the versioned symbol-reuse registry."""

    registry_path = (
        path.expanduser().resolve()
        if path is not None
        else Path(__file__).with_name("open_source") / "reference" / _REGISTRY_FILE
    )
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing security identity registry: {registry_path}")
    frame = pl.read_csv(registry_path, infer_schema_length=0)
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Security identity registry is missing columns: {sorted(missing)}")
    prepared = _prepare_registry(frame).with_columns(
        pl.lit(str(registry_path)).alias("registry_path")
    )
    _validate_registry(prepared)
    return prepared


def apply_security_identity_policy(
    frame: pl.DataFrame,
    *,
    ticker_column: str,
    date_column: str,
    registry: pl.DataFrame | None = None,
) -> SecurityIdentityApplication:
    """Split reused symbols by interval and reject rows in an identity gap."""

    if ticker_column not in frame.columns or date_column not in frame.columns:
        missing = [column for column in (ticker_column, date_column) if column not in frame.columns]
        raise ValueError(f"Security identity input is missing columns: {missing}")
    identities = _prepare_registry(
        registry if registry is not None else load_security_identity_registry()
    )
    _validate_registry(identities)
    if frame.is_empty() or identities.is_empty():
        return SecurityIdentityApplication(
            frame=frame,
            rejected=frame.clear(),
            report={
                "policy_id": SECURITY_IDENTITY_POLICY_ID,
                "targeted_rows": 0,
                "accepted_rows": 0,
                "rejected_rows": 0,
                "security_identity_count": 0,
                "canonical_tickers": [],
            },
        )

    original_columns = frame.columns
    working = frame.with_row_index("_identity_row_number").with_columns(
        pl.col(ticker_column)
        .cast(pl.String)
        .str.to_uppercase()
        .str.replace(r"\.US$", "")
        .alias("_source_ticker_root"),
        pl.col(ticker_column)
        .cast(pl.String)
        .str.to_uppercase()
        .str.ends_with(".US")
        .alias("_ticker_has_us_suffix"),
        pl.col(date_column).cast(pl.String).str.to_date(strict=False).alias("_identity_row_date"),
    )
    source_tickers = identities.get_column("source_ticker").unique().to_list()
    targeted = working.filter(pl.col("_source_ticker_root").is_in(source_tickers))
    untouched = working.filter(~pl.col("_source_ticker_root").is_in(source_tickers))
    windows = identities.select(
        "source_ticker",
        "canonical_ticker",
        "security_id",
        "issuer_cik",
        pl.col("valid_from").str.to_date(strict=False).alias("_identity_valid_from"),
        pl.col("valid_to").str.to_date(strict=False).alias("_identity_valid_to"),
    )
    matches = (
        targeted.join(
            windows,
            left_on="_source_ticker_root",
            right_on="source_ticker",
            how="inner",
        )
        .filter(
            pl.col("_identity_row_date").is_not_null()
            & (pl.col("_identity_row_date") >= pl.col("_identity_valid_from"))
            & (
                pl.col("_identity_valid_to").is_null()
                | (pl.col("_identity_row_date") <= pl.col("_identity_valid_to"))
            )
        )
        .with_columns(
            pl.when(pl.col("_ticker_has_us_suffix"))
            .then(pl.col("canonical_ticker") + pl.lit(".US"))
            .otherwise(pl.col("canonical_ticker"))
            .alias(ticker_column)
        )
    )
    accepted_target_rows = matches.get_column("_identity_row_number").unique().to_list()
    rejected = (
        targeted.filter(~pl.col("_identity_row_number").is_in(accepted_target_rows))
        .sort("_identity_row_number")
        .select(original_columns)
    )
    accepted = (
        pl.concat(
            [
                untouched.select(["_identity_row_number", *original_columns]),
                matches.select(["_identity_row_number", *original_columns]),
            ],
            how="vertical_relaxed",
        )
        .sort("_identity_row_number")
        .select(original_columns)
    )
    return SecurityIdentityApplication(
        frame=accepted,
        rejected=rejected,
        report={
            "policy_id": SECURITY_IDENTITY_POLICY_ID,
            "targeted_rows": targeted.height,
            "accepted_rows": matches.height,
            "rejected_rows": rejected.height,
            "security_identity_count": matches.select(pl.col("security_id").n_unique()).item()
            if matches.height
            else 0,
            "canonical_tickers": sorted(matches.get_column("canonical_ticker").unique().to_list())
            if matches.height
            else [],
            "rejected_examples": rejected.head(20).to_dicts(),
        },
    )


def assert_security_identity_compliance(
    frame: pl.DataFrame,
    *,
    ticker_column: str,
    date_column: str,
    registry: pl.DataFrame | None = None,
) -> None:
    """Fail when a dated row is not already on its canonical identity key."""

    identities = _prepare_registry(
        registry if registry is not None else load_security_identity_registry()
    )
    _validate_registry(identities)
    result = apply_security_identity_policy(
        frame,
        ticker_column=ticker_column,
        date_column=date_column,
        registry=identities,
    )
    canonical = frame.with_columns(
        pl.col(ticker_column)
        .cast(pl.String)
        .str.to_uppercase()
        .str.replace(r"\.US$", "")
        .alias("_canonical_ticker_root"),
        pl.col(date_column).cast(pl.String).str.to_date(strict=False).alias("_identity_row_date"),
    ).join(
        identities.select(
            "canonical_ticker",
            pl.col("valid_from").str.to_date(strict=False).alias("_valid_from"),
            pl.col("valid_to").str.to_date(strict=False).alias("_valid_to"),
        ),
        left_on="_canonical_ticker_root",
        right_on="canonical_ticker",
        how="inner",
    )
    canonical_interval_violations = canonical.filter(
        pl.col("_identity_row_date").is_null()
        | (pl.col("_identity_row_date") < pl.col("_valid_from"))
        | (pl.col("_valid_to").is_not_null() & (pl.col("_identity_row_date") > pl.col("_valid_to")))
    )
    observed = frame.select(ticker_column, date_column).with_row_index("_row")
    canonicalized = result.frame.select(ticker_column, date_column).with_row_index("_row")
    changed = not observed.equals(canonicalized, null_equal=True)
    if result.rejected.height or changed or canonical_interval_violations.height:
        raise RuntimeError(
            "Rows violate the canonical security identity interval; "
            f"rejected={result.rejected.height}, canonicalization_required={changed}, "
            f"canonical_interval_violations={canonical_interval_violations.height}"
        )


def apply_security_identity_reference_policy(
    frame: pl.DataFrame,
    *,
    ticker_column: str,
    cik_columns: tuple[str, ...] = ("cik", "sec_cik"),
    registry: pl.DataFrame | None = None,
) -> SecurityIdentityApplication:
    """Canonicalize undated reference rows by the issuer CIK."""

    if ticker_column not in frame.columns:
        raise ValueError(f"Security identity reference is missing ticker column: {ticker_column}")
    available_cik_columns = [column for column in cik_columns if column in frame.columns]
    if not available_cik_columns:
        raise ValueError("Security identity reference has no CIK column")
    identities = _prepare_registry(
        registry if registry is not None else load_security_identity_registry()
    )
    _validate_registry(identities)
    if frame.is_empty() or identities.is_empty():
        return SecurityIdentityApplication(
            frame=frame,
            rejected=frame.clear(),
            report=_empty_report(),
        )

    original_columns = frame.columns
    normalized_ciks = [
        pl.col(column).cast(pl.String).str.extract(r"(\d+)").str.zfill(10)
        for column in available_cik_columns
    ]
    working = frame.with_row_index("_identity_row_number").with_columns(
        pl.col(ticker_column)
        .cast(pl.String)
        .str.to_uppercase()
        .str.replace(r"\.US$", "")
        .alias("_source_ticker_root"),
        pl.col(ticker_column)
        .cast(pl.String)
        .str.to_uppercase()
        .str.ends_with(".US")
        .alias("_ticker_has_us_suffix"),
        pl.coalesce(normalized_ciks).alias("_reference_cik"),
    )
    source_tickers = identities.get_column("source_ticker").unique().to_list()
    targeted = working.filter(pl.col("_source_ticker_root").is_in(source_tickers))
    untouched = working.filter(~pl.col("_source_ticker_root").is_in(source_tickers))
    matches = targeted.join(
        identities.select(
            "source_ticker",
            "canonical_ticker",
            "security_id",
            "issuer_cik",
        ),
        left_on=["_source_ticker_root", "_reference_cik"],
        right_on=["source_ticker", "issuer_cik"],
        how="inner",
    ).with_columns(
        pl.when(pl.col("_ticker_has_us_suffix"))
        .then(pl.col("canonical_ticker") + pl.lit(".US"))
        .otherwise(pl.col("canonical_ticker"))
        .alias(ticker_column)
    )
    accepted_target_rows = matches.get_column("_identity_row_number").unique().to_list()
    rejected = (
        targeted.filter(~pl.col("_identity_row_number").is_in(accepted_target_rows))
        .sort("_identity_row_number")
        .select(original_columns)
    )
    accepted = (
        pl.concat(
            [
                untouched.select(["_identity_row_number", *original_columns]),
                matches.select(["_identity_row_number", *original_columns]),
            ],
            how="vertical_relaxed",
        )
        .sort("_identity_row_number")
        .select(original_columns)
    )
    return SecurityIdentityApplication(
        frame=accepted,
        rejected=rejected,
        report={
            "policy_id": SECURITY_IDENTITY_POLICY_ID,
            "targeted_rows": targeted.height,
            "accepted_rows": matches.height,
            "rejected_rows": rejected.height,
            "security_identity_count": matches.select(pl.col("security_id").n_unique()).item()
            if matches.height
            else 0,
            "canonical_tickers": sorted(matches.get_column("canonical_ticker").unique().to_list())
            if matches.height
            else [],
            "rejected_examples": rejected.head(20).to_dicts(),
        },
    )


def assert_security_identity_reference_compliance(
    frame: pl.DataFrame,
    *,
    ticker_column: str,
    cik_columns: tuple[str, ...] = ("cik", "sec_cik"),
    registry: pl.DataFrame | None = None,
) -> None:
    """Fail when a reference row does not use the canonical key for its CIK."""

    identities = _prepare_registry(
        registry if registry is not None else load_security_identity_registry()
    )
    _validate_registry(identities)
    result = apply_security_identity_reference_policy(
        frame,
        ticker_column=ticker_column,
        cik_columns=cik_columns,
        registry=identities,
    )
    available_cik_columns = [column for column in cik_columns if column in frame.columns]
    normalized_ciks = [
        pl.col(column).cast(pl.String).str.extract(r"(\d+)").str.zfill(10)
        for column in available_cik_columns
    ]
    canonical_cik_violations = (
        frame.with_columns(
            pl.col(ticker_column)
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace(r"\.US$", "")
            .alias("_canonical_ticker_root"),
            pl.coalesce(normalized_ciks).alias("_reference_cik"),
        )
        .join(
            identities.select("canonical_ticker", "issuer_cik"),
            left_on="_canonical_ticker_root",
            right_on="canonical_ticker",
            how="inner",
        )
        .filter(pl.col("_reference_cik") != pl.col("issuer_cik"))
    )
    observed = frame.select(ticker_column, *available_cik_columns).with_row_index("_row")
    canonical = result.frame.select(ticker_column, *available_cik_columns).with_row_index("_row")
    changed = not observed.equals(canonical, null_equal=True)
    if result.rejected.height or changed or canonical_cik_violations.height:
        raise RuntimeError(
            "Reference rows violate the canonical security identity CIK; "
            f"rejected={result.rejected.height}, canonicalization_required={changed}, "
            f"canonical_cik_violations={canonical_cik_violations.height}"
        )


def _prepare_registry(frame: pl.DataFrame) -> pl.DataFrame:
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Security identity registry is missing columns: {sorted(missing)}")
    result = frame.with_columns(
        pl.col("source_ticker").cast(pl.String).str.to_uppercase().str.replace(r"\.US$", ""),
        pl.col("canonical_ticker").cast(pl.String).str.to_uppercase().str.replace(r"\.US$", ""),
        pl.col("security_id").cast(pl.String),
        pl.col("issuer_cik").cast(pl.String).str.extract(r"(\d+)").str.zfill(10),
        pl.col("valid_from").cast(pl.String),
        pl.col("valid_to").cast(pl.String),
        pl.col("identity_status").cast(pl.String),
        pl.col("evidence").cast(pl.String),
    )
    if "registry_path" not in result.columns:
        result = result.with_columns(pl.lit(None).cast(pl.String).alias("registry_path"))
    return result


def _empty_report() -> dict[str, object]:
    return {
        "policy_id": SECURITY_IDENTITY_POLICY_ID,
        "targeted_rows": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "security_identity_count": 0,
        "canonical_tickers": [],
    }


def _validate_registry(registry: pl.DataFrame) -> None:
    if registry.is_empty():
        return
    invalid = registry.filter(
        pl.any_horizontal(
            pl.col("source_ticker").is_null(),
            pl.col("canonical_ticker").is_null(),
            pl.col("security_id").is_null(),
            pl.col("issuer_cik").is_null(),
            pl.col("valid_from").str.to_date(strict=False).is_null(),
        )
    )
    if invalid.height:
        raise ValueError("Security identity registry contains incomplete rows")
    duplicate_ids = registry.group_by("security_id").len().filter(pl.col("len") > 1)
    if duplicate_ids.height:
        raise ValueError("Security identity registry contains duplicate security ids")
    duplicate_canonical = registry.group_by("canonical_ticker").len().filter(pl.col("len") > 1)
    if duplicate_canonical.height:
        raise ValueError("Security identity registry contains duplicate canonical tickers")

    dated = registry.with_columns(
        pl.col("valid_from").str.to_date(strict=False).alias("_from"),
        pl.col("valid_to").str.to_date(strict=False).alias("_to"),
    ).sort("source_ticker", "_from")
    overlap = dated.with_columns(
        pl.col("_to").shift(1).over("source_ticker").alias("_previous_to"),
        pl.col("source_ticker").shift(1).alias("_previous_source_ticker"),
    ).filter(
        (pl.col("source_ticker") == pl.col("_previous_source_ticker"))
        & (pl.col("_previous_to").is_null() | (pl.col("_previous_to") >= pl.col("_from")))
    )
    if overlap.height:
        raise ValueError("Security identity registry contains overlapping intervals")
