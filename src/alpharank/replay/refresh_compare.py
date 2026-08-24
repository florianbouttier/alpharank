"""Typed table comparisons for refresh-to-portfolio replay audits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One table grain and the temporal column used by the historical cutoff."""

    name: str
    relative_path: str
    keys: tuple[str, ...]
    temporal_column: str | None = None
    ticker_column: str | None = None
    csv: bool = False


@dataclass(frozen=True, slots=True)
class FrameDiff:
    """Summary and exact changed keys for one baseline/candidate table pair."""

    summary: dict[str, object]
    added_keys: pl.DataFrame
    removed_keys: pl.DataFrame
    changed_keys: pl.DataFrame

    @property
    def has_historical_drift(self) -> bool:
        return any(
            int(self.summary[name]) > 0
            for name in ("added_rows", "removed_rows", "changed_common_rows")
        )


SNAPSHOT_TABLES = (
    TableSpec("final_price", "US_Finalprice.parquet", ("ticker", "date"), "date", "ticker"),
    TableSpec("sp500_price", "SP500Price.parquet", ("ticker", "date"), "date", "ticker"),
    TableSpec("general", "US_General.parquet", ("Code",), ticker_column="Code"),
    TableSpec(
        "income_statement",
        "US_Income_statement.parquet",
        ("ticker", "date", "filing_date"),
        "filing_date",
        "ticker",
    ),
    TableSpec(
        "balance_sheet",
        "US_Balance_sheet.parquet",
        ("ticker", "date", "filing_date"),
        "filing_date",
        "ticker",
    ),
    TableSpec(
        "cash_flow",
        "US_Cash_flow.parquet",
        ("ticker", "date", "filing_date"),
        "filing_date",
        "ticker",
    ),
    TableSpec(
        "earnings",
        "US_Earnings.parquet",
        ("ticker", "date", "reportDate"),
        "reportDate",
        "ticker",
    ),
    TableSpec(
        "sp500_constituents",
        "SP500_Constituents.csv",
        ("Date", "Ticker", "Name"),
        "Date",
        "Ticker",
        csv=True,
    ),
)


def read_table(root: Path, spec: TableSpec) -> pl.DataFrame:
    """Read one declared table and reject a missing replay input."""

    path = root / spec.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {spec.name} replay table: {path}")
    frame = pl.read_csv(path) if spec.csv else pl.read_parquet(path)
    missing = set(spec.keys) - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.name} is missing natural-key columns: {sorted(missing)}")
    if spec.name == "sp500_constituents":
        required = [pl.col(key).is_not_null() for key in spec.keys]
        frame = frame.filter(pl.all_horizontal(required)).unique(maintain_order=True)
    return frame


def compare_frames(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    spec: TableSpec,
    historical_cutoff: date | None,
    materiality_tolerance: float,
) -> FrameDiff:
    """Compare schemas, natural keys and values through one causal cutoff."""

    baseline = _through_cutoff(baseline, spec.temporal_column, historical_cutoff)
    candidate = _through_cutoff(candidate, spec.temporal_column, historical_cutoff)
    _require_unique(baseline, spec.keys, label=f"baseline {spec.name}")
    _require_unique(candidate, spec.keys, label=f"candidate {spec.name}")
    removed_columns = [name for name in baseline.columns if name not in candidate.columns]
    added_columns = [name for name in candidate.columns if name not in baseline.columns]
    type_changes = {
        name: {"baseline": str(baseline.schema[name]), "candidate": str(candidate.schema[name])}
        for name in baseline.columns
        if name in candidate.schema and baseline.schema[name] != candidate.schema[name]
    }
    if removed_columns or type_changes:
        raise ValueError(
            f"{spec.name} schema drift: removed={removed_columns}, type_changes={type_changes}"
        )
    compared_columns = [name for name in baseline.columns if name in candidate.columns]
    value_columns = [name for name in compared_columns if name not in spec.keys]
    baseline_keys = baseline.select(spec.keys)
    candidate_keys = candidate.select(spec.keys)
    added = candidate_keys.join(baseline_keys, on=list(spec.keys), how="anti")
    removed = baseline_keys.join(candidate_keys, on=list(spec.keys), how="anti")
    paired = _paired_values(baseline, candidate, spec.keys, value_columns)
    changed, maximum_difference = _changed_keys(
        paired,
        keys=spec.keys,
        value_columns=value_columns,
        schemas=(baseline.schema, candidate.schema),
        tolerance=materiality_tolerance,
    )
    summary: dict[str, object] = {
        "table": spec.name,
        "natural_key": list(spec.keys),
        "historical_cutoff": historical_cutoff.isoformat() if historical_cutoff else None,
        "baseline_rows": baseline.height,
        "candidate_rows": candidate.height,
        "added_rows": added.height,
        "removed_rows": removed.height,
        "changed_common_rows": changed.height,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
        "materiality_tolerance": materiality_tolerance,
        "maximum_numeric_absolute_difference": maximum_difference,
    }
    return FrameDiff(summary, added, removed, changed)


def changed_key_events(diff: FrameDiff) -> pl.DataFrame:
    """Return every changed natural key with an explicit change type."""

    frames = []
    for name, frame in (
        ("added", diff.added_keys),
        ("removed", diff.removed_keys),
        ("changed", diff.changed_keys),
    ):
        if not frame.is_empty():
            frames.append(frame.with_columns(pl.lit(name).alias("change_type")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def write_frame_diff(output_dir: Path, name: str, diff: FrameDiff) -> None:
    """Write exact key evidence without embedding large examples in JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, frame in (
        ("added_keys", diff.added_keys),
        ("removed_keys", diff.removed_keys),
        ("changed_keys", diff.changed_keys),
    ):
        if not frame.is_empty():
            frame.write_parquet(output_dir / f"{name}_{suffix}.parquet")


def _through_cutoff(
    frame: pl.DataFrame,
    temporal_column: str | None,
    cutoff: date | None,
) -> pl.DataFrame:
    if temporal_column is None or cutoff is None:
        return frame
    expression = pl.col(temporal_column)
    if frame.schema[temporal_column] == pl.String:
        expression = expression.str.to_date(strict=False)
    else:
        expression = expression.cast(pl.Date, strict=False)
    return frame.filter(expression.is_not_null() & (expression <= pl.lit(cutoff)))


def _require_unique(frame: pl.DataFrame, keys: tuple[str, ...], *, label: str) -> None:
    unique_count = frame.select(pl.struct(keys).n_unique()).item()
    if unique_count != frame.height:
        raise ValueError(f"{label} contains {frame.height - unique_count} duplicate keys: {keys}")


def _paired_values(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    keys: tuple[str, ...],
    values: list[str],
) -> pl.DataFrame:
    return baseline.select(
        *keys,
        *(pl.col(name).alias(f"baseline__{name}") for name in values),
    ).join(
        candidate.select(
            *keys,
            *(pl.col(name).alias(f"candidate__{name}") for name in values),
        ),
        on=list(keys),
        how="inner",
    )


def _changed_keys(
    paired: pl.DataFrame,
    *,
    keys: tuple[str, ...],
    value_columns: list[str],
    schemas: tuple[pl.Schema, pl.Schema],
    tolerance: float,
) -> tuple[pl.DataFrame, float]:
    if not value_columns:
        return paired.select(keys).clear(), 0.0
    material_masks = []
    numeric_differences = []
    changed_names = []
    for name in value_columns:
        baseline = pl.col(f"baseline__{name}")
        candidate = pl.col(f"candidate__{name}")
        exact = baseline.eq_missing(candidate).not_()
        changed_names.append(
            pl.when(exact).then(pl.lit(name)).otherwise(pl.lit(None, dtype=pl.String))
        )
        if schemas[0][name].is_numeric() and schemas[1][name].is_numeric():
            difference = (baseline.cast(pl.Float64) - candidate.cast(pl.Float64)).abs()
            numeric_differences.append(difference)
            material_masks.append(
                pl.when(baseline.is_null() | candidate.is_null())
                .then(exact)
                .otherwise(difference.fill_nan(float("inf")) > tolerance)
            )
        else:
            material_masks.append(exact)
    changed = paired.filter(pl.any_horizontal(material_masks)).select(
        *keys,
        pl.concat_list(changed_names).list.drop_nulls().alias("changed_columns"),
    )
    maximum = 0.0
    if numeric_differences and not paired.is_empty():
        observed = paired.select(pl.max_horizontal(numeric_differences).max()).item()
        if observed is not None and observed != float("inf"):
            maximum = float(observed)
    return changed, maximum
