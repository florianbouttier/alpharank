"""Sealed economic reconciliation of v1-audited-biased and v2-causal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.common_v2 import validate_common_v2_replay
from alpharank.governance import reserve_run_directory, validate_baseline_package
from alpharank.portfolio.performance import legacy_report_statistics


RECONCILIATION_TOLERANCE = 1e-12


def build_v1_v2_reconciliation(
    *,
    baseline_dir: Path,
    common_v2_dir: Path,
    output_dir: Path,
    expected_composition_id: str,
) -> dict[str, Any]:
    """Write a month-complete, cause-labelled, immutable reconciliation."""

    baseline = validate_baseline_package(baseline_dir)
    common = validate_common_v2_replay(
        common_v2_dir, expected_composition_id=expected_composition_id
    )
    destination = reserve_run_directory(output_dir)
    v1_root = baseline_dir / "payload" / "common_replay"
    v1_holdings_path = v1_root / "comparison_common_holdings.parquet"
    v1_monthly_path = v1_root / "comparison_common_monthly.parquet"
    v2_manifest = _read_json(common_v2_dir / "manifest.json")
    v2_holdings_path = Path(v2_manifest["artifacts"]["holdings"]["path"])
    v2_monthly_path = Path(v2_manifest["artifacts"]["monthly_parquet"]["path"])
    v1_holdings = pl.read_parquet(v1_holdings_path)
    v1_monthly = pl.read_parquet(v1_monthly_path)
    v2_holdings = pl.read_parquet(v2_holdings_path)
    v2_monthly = pl.read_parquet(v2_monthly_path)
    frames = reconcile_economic_frames(
        v1_holdings=v1_holdings,
        v1_monthly=v1_monthly,
        v2_holdings=v2_holdings,
        v2_monthly=v2_monthly,
    )
    artifact_paths = {
        "selection_reconciliation": destination / "selection_reconciliation.parquet",
        "monthly_reconciliation": destination / "monthly_reconciliation.parquet",
        "monthly_reconciliation_csv": destination / "monthly_reconciliation.csv",
        "metrics_reconciliation": destination / "metrics_reconciliation.csv",
        "report": destination / "reconciliation_report.md",
    }
    frames["selection"].write_parquet(artifact_paths["selection_reconciliation"])
    frames["monthly"].write_parquet(artifact_paths["monthly_reconciliation"])
    frames["monthly"].write_csv(artifact_paths["monthly_reconciliation_csv"])
    frames["metrics"].write_csv(artifact_paths["metrics_reconciliation"])
    artifact_paths["report"].write_text(
        _render_report(frames), encoding="utf-8"
    )
    manifest = {
        "contract_version": 1,
        "scope": "alpharank_v1_v2_economic_reconciliation",
        "status": "explanatory_not_promoted",
        "baseline_id": baseline["baseline_id"],
        "v2_composition_id": expected_composition_id,
        "numeric_tolerance": RECONCILIATION_TOLERANCE,
        "cause_taxonomy": {
            "signal_and_universe": [
                "UNI-001/002/003/004",
                "FND-001/002/003/004",
                "BST-001/002/003",
                "LEG-001",
            ],
            "return_and_benchmark": ["SIM-001", "LEG-002", "LEG-003"],
            "allocation_and_costs": ["SIM-002", "SIM-003"],
        },
        "source_validation": {"baseline": {"passed": True}, "common_v2": common},
        "sources": {
            "v1_holdings": _file_record(v1_holdings_path),
            "v1_monthly": _file_record(v1_monthly_path),
            "v2_holdings": _file_record(v2_holdings_path),
            "v2_monthly": _file_record(v2_monthly_path),
        },
        "artifacts": {
            label: _file_record(path) for label, path in artifact_paths.items()
        },
        "summary": {
            "selection_rows": frames["selection"].height,
            "selection_changes": frames["selection"].filter(
                pl.col("status") != "unchanged"
            ).height,
            "monthly_rows": frames["monthly"].height,
            "divergent_month_rows": frames["monthly"].filter(
                pl.col("status") != "unchanged"
            ).height,
            "strategy_count": frames["monthly"]["strategy"].n_unique(),
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (destination / "manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    _remove_write_bits(destination)
    return validate_v1_v2_reconciliation(destination)


def reconcile_economic_frames(
    *,
    v1_holdings: pl.DataFrame,
    v1_monthly: pl.DataFrame,
    v2_holdings: pl.DataFrame,
    v2_monthly: pl.DataFrame,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> dict[str, pl.DataFrame]:
    """Return selection, month and recalculated metric reconciliation frames."""

    selection = _selection_reconciliation(v1_holdings, v2_holdings, tolerance)
    selection_changes = {
        (str(row["strategy"]), row["holding_month"]): int(row["changed"])
        for row in selection.group_by("strategy", "holding_month").agg(
            (pl.col("status") != "unchanged").any().cast(pl.Int64).alias("changed")
        ).to_dicts()
    }
    v1_rows = {
        (str(row["strategy"]), row["holding_month"]): row
        for row in v1_monthly.to_dicts()
    }
    v2_rows = {
        (str(row["strategy"]), row["holding_month"]): row
        for row in v2_monthly.to_dicts()
    }
    monthly_rows: list[dict[str, Any]] = []
    numeric = (
        "gross_return",
        "turnover",
        "transaction_cost",
        "net_return",
        "benchmark_return",
    )
    for key in sorted(set(v1_rows) | set(v2_rows), key=lambda item: (item[0], item[1])):
        left = v1_rows.get(key)
        right = v2_rows.get(key)
        row: dict[str, Any] = {
            "strategy": key[0],
            "holding_month": key[1],
            "decision_month_v1": left.get("decision_month") if left else None,
            "decision_month_v2": right.get("decision_month") if right else None,
        }
        for column in numeric:
            before = float(left[column]) if left is not None else None
            after = float(right[column]) if right is not None else None
            row[f"v1_{column}"] = before
            row[f"v2_{column}"] = after
            row[f"delta_{column}"] = (
                after - before if before is not None and after is not None else None
            )
        causes: list[str] = []
        if left is None:
            status = "v2_only"
            causes.append("calendar_extension")
        elif right is None:
            status = "v1_only"
            causes.append("calendar_reduction")
        else:
            if selection_changes.get(key, 0):
                causes.append(
                    "UNI-001/002/003/004+FND-001/002/003/004+"
                    "BST-001/002/003+LEG-001"
                )
            if (
                abs(row["delta_gross_return"]) > tolerance
                or abs(row["delta_benchmark_return"]) > tolerance
            ):
                causes.append("SIM-001+LEG-002+LEG-003")
            if (
                abs(row["delta_turnover"]) > tolerance
                or abs(row["delta_transaction_cost"]) > tolerance
            ):
                causes.append("SIM-002+SIM-003")
            status = (
                "unchanged"
                if all(abs(row[f"delta_{column}"]) <= tolerance for column in numeric)
                and not selection_changes.get(key, 0)
                else "divergent_explained"
            )
        row["status"] = status
        row["cause_codes"] = ";".join(causes) if causes else "none"
        monthly_rows.append(row)
    monthly = pl.DataFrame(monthly_rows)
    divergent_without_cause = monthly.filter(
        (pl.col("status") != "unchanged") & (pl.col("cause_codes") == "none")
    )
    if not divergent_without_cause.is_empty():
        raise RuntimeError("A divergent economic month has no attributed cause")
    metrics = _metrics_reconciliation(v1_monthly, v2_monthly)
    return {"selection": selection, "monthly": monthly, "metrics": metrics}


def validate_v1_v2_reconciliation(output_dir: Path) -> dict[str, Any]:
    root = output_dir.resolve()
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    expected_manifest_sha = seal_path.read_text(encoding="utf-8").split()[0]
    if _sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError("Reconciliation manifest seal mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("scope") != "alpharank_v1_v2_economic_reconciliation":
        raise RuntimeError("Invalid v1/v2 reconciliation scope")
    for label, record in manifest["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Reconciliation artifact hash mismatch: {label}")
    monthly = pl.read_parquet(
        Path(manifest["artifacts"]["monthly_reconciliation"]["path"])
    )
    if monthly.filter(
        (pl.col("status") != "unchanged") & (pl.col("cause_codes") == "none")
    ).height:
        raise RuntimeError("A divergent reconciliation month has no cause")
    stored_metrics = pl.read_csv(
        Path(manifest["artifacts"]["metrics_reconciliation"]["path"])
    )
    recalculated = _metrics_reconciliation(
        pl.read_parquet(Path(manifest["sources"]["v1_monthly"]["path"])),
        pl.read_parquet(Path(manifest["sources"]["v2_monthly"]["path"])),
    )
    numeric = [
        column
        for column in stored_metrics.columns
        if column.startswith(("v1_", "v2_", "delta_"))
    ]
    joined = stored_metrics.join(
        recalculated.select(
            "strategy",
            *[pl.col(column).alias(f"recalculated_{column}") for column in numeric],
        ),
        on="strategy",
        how="inner",
        validate="1:1",
    )
    maximum_error = max(
        float(
            joined.select(
                (pl.col(column) - pl.col(f"recalculated_{column}")).abs().max()
            ).item()
        )
        for column in numeric
    )
    if maximum_error > RECONCILIATION_TOLERANCE:
        raise RuntimeError("Reconciliation metrics do not recalculate exactly")
    return {
        "passed": True,
        "status": manifest["status"],
        "baseline_id": manifest["baseline_id"],
        "v2_composition_id": manifest["v2_composition_id"],
        "monthly_rows": monthly.height,
        "divergent_month_rows": monthly.filter(
            pl.col("status") != "unchanged"
        ).height,
        "maximum_absolute_metric_recalculation_error": maximum_error,
        "manifest_sha256": _sha256(manifest_path),
        "output_dir": str(root),
    }


def _selection_reconciliation(
    v1: pl.DataFrame, v2: pl.DataFrame, tolerance: float
) -> pl.DataFrame:
    keys = ("strategy", "decision_month", "holding_month", "ticker")
    before = {
        tuple(row[key] for key in keys): float(row["target_weight"])
        for row in v1.to_dicts()
    }
    after = {
        tuple(row[key] for key in keys): float(row["target_weight"])
        for row in v2.to_dicts()
    }
    rows = []
    for key in sorted(set(before) | set(after), key=lambda item: tuple(map(str, item))):
        left = before.get(key)
        right = after.get(key)
        if left is None:
            status, cause = "added_v2", "UNI/FND/BST/LEG signal corrections"
        elif right is None:
            status, cause = "removed_v2", "UNI/FND/BST/LEG signal corrections"
        elif abs(right - left) > tolerance:
            status, cause = "weight_changed", "SIM-002 allocation correction"
        else:
            status, cause = "unchanged", "none"
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "v1_target_weight": left,
                "v2_target_weight": right,
                "delta_target_weight": (
                    right - left if left is not None and right is not None else None
                ),
                "status": status,
                "cause_codes": cause,
            }
        )
    return pl.DataFrame(rows)


def _metrics_reconciliation(v1: pl.DataFrame, v2: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for strategy in sorted(set(v1["strategy"].to_list()) | set(v2["strategy"].to_list())):
        row: dict[str, Any] = {"strategy": strategy}
        for version, frame in (("v1", v1), ("v2", v2)):
            selected = frame.filter(pl.col("strategy") == strategy).sort("holding_month")
            stats = legacy_report_statistics(
                selected["net_return"].to_numpy(),
                holding_months=selected["holding_month"].to_list(),
            )
            row.update(
                {
                    f"{version}_cagr": float(stats["cagr"]),
                    f"{version}_sharpe": float(stats["sharpe"]),
                    f"{version}_max_drawdown": float(stats["max_drawdown"]),
                    f"{version}_average_turnover": float(selected["turnover"].mean()),
                    f"{version}_total_transaction_cost": float(
                        selected["transaction_cost"].sum()
                    ),
                }
            )
        for metric in (
            "cagr",
            "sharpe",
            "max_drawdown",
            "average_turnover",
            "total_transaction_cost",
        ):
            row[f"delta_{metric}"] = row[f"v2_{metric}"] - row[f"v1_{metric}"]
        rows.append(row)
    return pl.DataFrame(rows)


def _render_report(frames: dict[str, pl.DataFrame]) -> str:
    monthly = frames["monthly"]
    metrics = frames["metrics"]
    lines = [
        "# Rapprochement économique v1-audited-biased / v2-causal",
        "",
        "Ce rapport explique les ruptures ; il ne promeut pas automatiquement v2.",
        "",
        f"- Lignes stratégie-mois : {monthly.height}",
        f"- Lignes divergentes expliquées : {monthly.filter(pl.col('status') != 'unchanged').height}",
        f"- Changements de sélection/poids : {frames['selection'].filter(pl.col('status') != 'unchanged').height}",
        "",
        "## Métriques recalculées",
        "",
        "| Stratégie | CAGR v1 | CAGR v2 | Delta CAGR | Sharpe v1 | Sharpe v2 | Drawdown v1 | Drawdown v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort("strategy").to_dicts():
        lines.append(
            f"| {row['strategy']} | {row['v1_cagr']:.4%} | {row['v2_cagr']:.4%} | "
            f"{row['delta_cagr']:.4%} | {row['v1_sharpe']:.4f} | "
            f"{row['v2_sharpe']:.4f} | {row['v1_max_drawdown']:.4%} | "
            f"{row['v2_max_drawdown']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Taxonomie des causes",
            "",
            "- UNI/FND/BST/LEG : univers, disponibilité fondamentale, sélection et secteurs causaux.",
            "- SIM-001 + LEG-002/003 : rendement terminal, benchmark total return et ouverture suivante.",
            "- SIM-002/003 : turnover dérivé et coûts de transaction décomposés.",
            "",
        ]
    )
    return "\n".join(lines)


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_write_bits(root: Path) -> None:
    paths = [path for path in root.rglob("*") if path.is_file()]
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in [*paths, *directories, root]:
        path.chmod(path.stat().st_mode & ~0o222)
