#!/usr/bin/env python3
"""Audit exact data revisions and downstream replay impact between two snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


DATASETS = {
    "final_price": ("US_Finalprice.parquet", ("ticker", "date")),
    "sp500_price": ("SP500Price.parquet", ("ticker", "date")),
    "general": ("US_General.parquet", ("Code",)),
    "income_statement": (
        "US_Income_statement.parquet",
        ("ticker", "date", "filing_date"),
    ),
    "balance_sheet": (
        "US_Balance_sheet.parquet",
        ("ticker", "date", "filing_date"),
    ),
    "cash_flow": ("US_Cash_flow.parquet", ("ticker", "date", "filing_date")),
    "earnings": ("US_Earnings.parquet", ("ticker", "date", "reportDate")),
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _require_unique(frame: pl.DataFrame, keys: tuple[str, ...], *, label: str) -> None:
    unique = frame.select(pl.struct(keys).n_unique()).item()
    if unique != frame.height:
        raise ValueError(f"{label} has {frame.height - unique} duplicate natural keys: {keys}")


def _compare_frames(
    previous: pl.DataFrame,
    current: pl.DataFrame,
    *,
    keys: tuple[str, ...],
    example_limit: int = 100,
    materiality_tolerance: float = 0.0,
) -> dict[str, Any]:
    if previous.columns != current.columns:
        raise ValueError(
            f"Schema drift: previous={previous.columns}, current={current.columns}"
        )
    _require_unique(previous, keys, label="previous frame")
    _require_unique(current, keys, label="current frame")
    value_columns = [column for column in previous.columns if column not in keys]
    previous_keys = previous.select(keys)
    current_keys = current.select(keys)
    added = current_keys.join(previous_keys, on=list(keys), how="anti")
    removed = previous_keys.join(current_keys, on=list(keys), how="anti")
    paired = previous.select(
        *keys,
        *(pl.col(column).alias(f"previous__{column}") for column in value_columns),
    ).join(
        current.select(
            *keys,
            *(pl.col(column).alias(f"current__{column}") for column in value_columns),
        ),
        on=list(keys),
        how="inner",
    )
    exact_change_exprs: list[pl.Expr] = []
    material_change_exprs: list[pl.Expr] = []
    numeric_difference_exprs: list[pl.Expr] = []
    for column in value_columns:
        previous_column = pl.col(f"previous__{column}")
        current_column = pl.col(f"current__{column}")
        exact_change = previous_column.eq_missing(current_column).not_()
        exact_change_exprs.append(exact_change)
        if previous.schema[column].is_numeric():
            difference = (
                previous_column.cast(pl.Float64) - current_column.cast(pl.Float64)
            ).abs()
            numeric_difference_exprs.append(difference.fill_nan(float("inf")))
            material_change_exprs.append(
                pl.when(previous_column.is_null() | current_column.is_null())
                .then(exact_change)
                .otherwise(difference.fill_nan(float("inf")) > materiality_tolerance)
            )
        else:
            material_change_exprs.append(exact_change)

    exact_changed = pl.any_horizontal(exact_change_exprs) if exact_change_exprs else pl.lit(False)
    material_changed = (
        pl.any_horizontal(material_change_exprs) if material_change_exprs else pl.lit(False)
    )
    changed_pairs = paired.filter(exact_changed).sort(list(keys))
    materially_changed_rows = paired.filter(material_changed).height
    maximum_numeric_absolute_difference = 0.0
    if numeric_difference_exprs and not paired.is_empty():
        maximum = paired.select(
            pl.max_horizontal(numeric_difference_exprs).max().alias("maximum")
        ).item()
        if maximum is not None and math.isfinite(float(maximum)):
            maximum_numeric_absolute_difference = float(maximum)

    changed_examples: list[dict[str, Any]] = []
    for row in changed_pairs.head(example_limit).iter_rows(named=True):
        changes: dict[str, Any] = {}
        for column in value_columns:
            previous_value = row[f"previous__{column}"]
            current_value = row[f"current__{column}"]
            if previous_value == current_value or (
                previous_value is None and current_value is None
            ):
                continue
            difference = None
            is_material = True
            if (
                isinstance(previous_value, (int, float))
                and not isinstance(previous_value, bool)
                and isinstance(current_value, (int, float))
                and not isinstance(current_value, bool)
            ):
                difference = abs(float(previous_value) - float(current_value))
                if math.isfinite(difference):
                    is_material = difference > materiality_tolerance
            changes[column] = {
                "previous": _clean(previous_value),
                "current": _clean(current_value),
                "absolute_difference": difference,
                "material": is_material,
            }
        changed_examples.append(
            {
                "key": _clean({key: row[key] for key in keys}),
                "changed_values": changes,
            }
        )
    return {
        "previous_rows": previous.height,
        "current_rows": current.height,
        "added_rows": added.height,
        "removed_rows": removed.height,
        "changed_common_rows": changed_pairs.height,
        "materiality_tolerance": materiality_tolerance,
        "materially_changed_common_rows": materially_changed_rows,
        "maximum_numeric_absolute_difference": maximum_numeric_absolute_difference,
        "added_key_examples": _clean(
            added.sort(list(keys)).head(example_limit).to_dicts()
        ),
        "removed_key_examples": _clean(
            removed.sort(list(keys)).head(example_limit).to_dicts()
        ),
        "changed_row_examples": changed_examples,
    }


def _audit_dataset(previous_dir: Path, current_dir: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for name, (filename, keys) in DATASETS.items():
        previous_path = previous_dir / filename
        current_path = current_dir / filename
        previous_hash = _hash(previous_path)
        current_hash = _hash(current_path)
        result = {
            "file": filename,
            "natural_key": keys,
            "previous_sha256": previous_hash,
            "current_sha256": current_hash,
            "identical_file": previous_hash == current_hash,
        }
        if previous_hash == current_hash:
            rows = pl.scan_parquet(previous_path).select(pl.len()).collect().item()
            result.update(
                {
                    "previous_rows": rows,
                    "current_rows": rows,
                    "added_rows": 0,
                    "removed_rows": 0,
                    "changed_common_rows": 0,
                    "added_key_examples": [],
                    "removed_key_examples": [],
                    "changed_row_examples": [],
                }
            )
        else:
            result.update(
                _compare_frames(
                    pl.read_parquet(previous_path),
                    pl.read_parquet(current_path),
                    keys=keys,
                )
            )
        datasets[name] = result
    constituents = "SP500_Constituents.csv"
    previous_constituents = previous_dir / constituents
    current_constituents = current_dir / constituents
    datasets["sp500_constituents"] = {
        "file": constituents,
        "previous_sha256": _hash(previous_constituents),
        "current_sha256": _hash(current_constituents),
        "identical_file": _hash(previous_constituents) == _hash(current_constituents),
        "previous_rows": pl.scan_csv(previous_constituents).select(pl.len()).collect().item(),
        "current_rows": pl.scan_csv(current_constituents).select(pl.len()).collect().item(),
    }
    return datasets


def _audit_replays(
    *,
    previous_legacy_run: Path,
    current_legacy_run: Path,
    previous_boosting_run: Path,
    current_boosting_run: Path,
) -> dict[str, Any]:
    legacy_holdings = _compare_frames(
        pl.read_parquet(previous_legacy_run / "legacy_common_holdings.parquet"),
        pl.read_parquet(current_legacy_run / "legacy_common_holdings.parquet"),
        keys=("strategy", "decision_month", "holding_month", "ticker"),
        materiality_tolerance=1e-12,
    )
    legacy_monthly = _compare_frames(
        pl.read_parquet(previous_legacy_run / "legacy_common_monthly.parquet"),
        pl.read_parquet(current_legacy_run / "legacy_common_monthly.parquet"),
        keys=("strategy", "decision_month", "holding_month"),
        materiality_tolerance=1e-12,
    )
    prediction_path = Path("classification_h06/predictions.parquet")
    boosting_predictions = _compare_frames(
        pl.read_parquet(previous_boosting_run / prediction_path),
        pl.read_parquet(current_boosting_run / prediction_path),
        keys=("decision_month", "ticker", "fold", "method", "horizon"),
        materiality_tolerance=1e-12,
    )
    return {
        "legacy_holdings": legacy_holdings,
        "legacy_monthly_returns": legacy_monthly,
        "boosting_predictions": boosting_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-snapshot", type=Path, required=True)
    parser.add_argument("--current-snapshot", type=Path, required=True)
    parser.add_argument("--previous-legacy-run", type=Path, required=True)
    parser.add_argument("--current-legacy-run", type=Path, required=True)
    parser.add_argument("--previous-boosting-run", type=Path, required=True)
    parser.add_argument("--current-boosting-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = _audit_dataset(args.previous_snapshot, args.current_snapshot)
    replay_impact = _audit_replays(
        previous_legacy_run=args.previous_legacy_run,
        current_legacy_run=args.current_legacy_run,
        previous_boosting_run=args.previous_boosting_run,
        current_boosting_run=args.current_boosting_run,
    )
    payload = {
        "status": "audited_snapshot_revision",
        "previous_snapshot": str(args.previous_snapshot.resolve()),
        "current_snapshot": str(args.current_snapshot.resolve()),
        "datasets": datasets,
        "replay_impact": replay_impact,
        "conclusion": {
            "legacy_identical": all(
                replay_impact[name]["added_rows"] == 0
                and replay_impact[name]["removed_rows"] == 0
                and replay_impact[name]["materially_changed_common_rows"] == 0
                for name in ("legacy_holdings", "legacy_monthly_returns")
            ),
            "boosting_predictions_identical": (
                replay_impact["boosting_predictions"]["added_rows"] == 0
                and replay_impact["boosting_predictions"]["removed_rows"] == 0
                and replay_impact["boosting_predictions"][
                    "materially_changed_common_rows"
                ]
                == 0
            ),
            "historical_source_revision_detected": any(
                dataset.get("changed_common_rows", 0) > 0
                for dataset in datasets.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_clean(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
