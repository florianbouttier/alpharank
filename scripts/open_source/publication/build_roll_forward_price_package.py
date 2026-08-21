#!/usr/bin/env python3
"""Build a canonical price package from a validated base plus one fresh vintage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.data.prices import (
    audit_price_candidate,
    build_persistent_price_history_registry,
    load_eodhd_seed,
    persistent_history_summary,
    resolve_previous_validated_price_lineage,
    roll_forward_validated_price_history,
    validate_price_candidate,
)
from alpharank.data.prices.contracts import PRODUCTION_PRICE_GATE_POLICY
from alpharank.data.security_identity import (
    SECURITY_IDENTITY_POLICY_ID,
    apply_security_identity_policy,
    load_security_identity_registry,
)


def main() -> None:
    args = _parse_args()
    base_dir = args.base_package_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "lineage").mkdir()
    (output_dir / "audit").mkdir()

    previous_source = None
    if args.previous_validated_lineage is not None:
        previous_lineage_path = args.previous_validated_lineage.resolve()
    else:
        previous_source = resolve_previous_validated_price_lineage(
            args.latest_composed_manifest.resolve()
        )
        previous_lineage_path = previous_source.lineage_path
    previous_lineage = pl.read_parquet(previous_lineage_path)
    fresh_yahoo = pl.read_parquet(args.fresh_yahoo_vintage.resolve())
    security_identities = load_security_identity_registry()
    base_manifest = _read_json(base_dir / "lineage" / "manifest.json")
    active_resolution_vintage_id = _resolve_active_resolution_vintage_id(
        base_manifest=base_manifest,
        fresh_yahoo=fresh_yahoo,
    )
    active_tickers = _latest_constituents(base_dir / "SP500_Constituents.csv")
    terminal_tickers = _validated_terminal_tickers(
        requested=args.preserve_terminal_tickers,
        registry_path=args.constituent_registry.resolve(),
        expected_through=args.expected_through,
    )
    terminal_set = set(terminal_tickers)
    refreshable_active_tickers = tuple(
        ticker
        for ticker in active_tickers
        if f"{ticker.upper().removesuffix('.US')}.US" not in terminal_set
    )
    result = roll_forward_validated_price_history(
        previous_validated_lineage=previous_lineage,
        active_yahoo_vintage=fresh_yahoo,
        active_tickers=active_tickers,
        preserved_terminal_tickers=terminal_tickers,
        active_resolution_vintage_id=active_resolution_vintage_id,
        security_identity_registry=security_identities,
    )
    seed = load_eodhd_seed(args.eodhd_seed.resolve(), start_date=args.start_date)
    gate = audit_price_candidate(
        previous_prices=previous_lineage,
        candidate_prices=result.prices,
        candidate_lineage=result.lineage,
        active_tickers=refreshable_active_tickers,
        expected_eodhd_keys=seed.frame.select("ticker", "date"),
        expected_through=args.expected_through,
        policy=PRODUCTION_PRICE_GATE_POLICY,
        active_resolution_vintage_id=active_resolution_vintage_id,
    )
    history_registry = build_persistent_price_history_registry(
        result.lineage,
        active_tickers=active_tickers,
        preserved_terminal_tickers=terminal_tickers,
    )
    history_summary = persistent_history_summary(history_registry)

    result.prices.write_parquet(output_dir / "US_Finalprice.parquet")
    result.lineage.write_parquet(output_dir / "lineage" / "prices_open_source_lineage.parquet")
    history_registry.write_parquet(
        output_dir / "lineage" / "persistent_price_history_registry.parquet"
    )
    shutil.copy2(base_dir / "SP500Price.parquet", output_dir / "SP500Price.parquet")
    constituents_identity = apply_security_identity_policy(
        pl.read_csv(base_dir / "SP500_Constituents.csv", infer_schema_length=0),
        ticker_column="Ticker",
        date_column="Date",
        registry=security_identities,
    )
    constituents_identity.frame.write_csv(output_dir / "SP500_Constituents.csv")
    gate.daily_return_revisions.write_parquet(
        output_dir / "audit" / "price_daily_return_revisions.parquet"
    )
    gate.transition_factor_findings.write_parquet(
        output_dir / "audit" / "price_transition_factor_findings.parquet"
    )
    gate.historical_key_removals.write_parquet(
        output_dir / "audit" / "price_historical_key_removals.parquet"
    )
    _write_json(output_dir / "audit" / "price_revision_guard.json", gate.report)
    _write_json(output_dir / "audit" / "price_composition.json", result.composition_report)

    source_contract = dict(base_manifest["source_refresh_contract"])
    source_contract["contract_version"] = 2
    source_contract["price_composition"] = result.composition_report
    source_contract["price_revision_guard"] = gate.report
    source_contract["previous_validated_price_lineage"] = {
        **_file_record(previous_lineage_path),
        "resolution": (
            "explicit_cli_path" if previous_source is None else "latest_composed_model_snapshot"
        ),
        "composition_id": (previous_source.composition_id if previous_source is not None else None),
    }
    source_contract["fresh_yahoo_vintage"] = _file_record(args.fresh_yahoo_vintage.resolve())
    source_contract["eodhd_price_seed"] = seed.manifest()
    source_contract["persistent_price_history"] = {
        **history_summary,
        "semantics": (
            "Every ticker/date published by the preceding validated lineage is "
            "retained when the ticker leaves the active refresh universe, "
            "including histories first acquired from Yahoo and absent from EODHD."
        ),
        "routine_deletion_allowed": False,
    }
    source_contract["security_identity"] = {
        "policy_id": SECURITY_IDENTITY_POLICY_ID,
        "registry": _file_record(
            Path(security_identities.get_column("registry_path").drop_nulls().unique().item())
        ),
        "price_lineage": result.composition_report["security_identity"],
        "constituents": constituents_identity.report,
    }
    source_contract["policy"] = {
        **source_contract["policy"],
        "allow_historical_price_revisions": False,
        "allow_historical_price_key_removals": False,
    }
    manifest = {
        "contract_version": 2,
        "scope": "canonical_price_package",
        "run_id": base_manifest["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_full_ingestion_package": str(base_dir),
        "source_refresh_contract": source_contract,
        "data_freshness": base_manifest["data_freshness"],
        "output_sha256": {
            name: _sha256(output_dir / name)
            for name in (
                "US_Finalprice.parquet",
                "SP500Price.parquet",
                "SP500_Constituents.csv",
            )
        },
        "validation": {
            "inactive_history_byte_preserved": True,
            "all_previous_validated_inactive_history_preserved": True,
            "open_source_only_inactive_history_persisted": True,
            "active_history_single_fresh_yahoo_vintage": (
                result.composition_report["audited_carried_active_rows"] == 0
            ),
            "active_history_audited_resolution_run": True,
            "active_history_audited_carried_rows": result.composition_report[
                "audited_carried_active_rows"
            ],
            "price_revision_guard_passed": gate.report["passed"],
            "security_identity_policy_applied": True,
        },
        "artifacts": {
            "price_lineage": _file_record(
                output_dir / "lineage" / "prices_open_source_lineage.parquet"
            ),
            "persistent_price_history_registry": _file_record(
                output_dir / "lineage" / "persistent_price_history_registry.parquet"
            ),
        },
    }
    _write_json(output_dir / "lineage" / "manifest.json", manifest)
    validate_price_candidate(gate)
    print(json.dumps({"output_dir": str(output_dir), "manifest": manifest}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package-dir", type=Path, required=True)
    parser.add_argument(
        "--previous-validated-lineage",
        type=Path,
        help=(
            "Explicit prior lineage. If omitted, resolve it from the latest "
            "validated composed model snapshot."
        ),
    )
    parser.add_argument(
        "--latest-composed-manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "data"
            / "model_inputs"
            / "manifests"
            / "latest.json"
        ),
    )
    parser.add_argument("--fresh-yahoo-vintage", type=Path, required=True)
    parser.add_argument("--eodhd-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-through", default=date.today().isoformat())
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument(
        "--preserve-terminal-tickers",
        nargs="*",
        default=(),
        help="Carry forward only tickers with a confirmed removal event in the registry.",
    )
    parser.add_argument(
        "--constituent-registry",
        type=Path,
        default=Path("configs/data_quality/sp500_constituent_changes_2026.json"),
    )
    return parser.parse_args()


def _latest_constituents(path: Path) -> tuple[str, ...]:
    frame = pl.read_csv(path, infer_schema_length=0)
    date_col = "Date" if "Date" in frame.columns else "date"
    ticker_col = "Ticker" if "Ticker" in frame.columns else "ticker"
    latest = frame.select(pl.col(date_col).max()).item()
    return tuple(
        frame.filter(pl.col(date_col) == latest)
        .get_column(ticker_col)
        .drop_nulls()
        .cast(pl.String)
        .str.to_uppercase()
        .unique()
        .sort()
        .to_list()
    )


def _resolve_active_resolution_vintage_id(
    *,
    base_manifest: dict[str, object],
    fresh_yahoo: pl.DataFrame,
) -> str:
    """Bind audited carried rows to the full-ingestion run that selected them."""

    run_id = str(base_manifest.get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("Full-ingestion manifest does not declare a run_id")
    vintage_column = (
        "source_vintage_id"
        if "source_vintage_id" in fresh_yahoo.columns
        else "ingestion_run_id"
        if "ingestion_run_id" in fresh_yahoo.columns
        else None
    )
    if vintage_column is None:
        raise RuntimeError("Fresh Yahoo vintage does not carry a source or ingestion run id")
    observed_vintages = {
        str(value)
        for value in fresh_yahoo.get_column(vintage_column).drop_nulls().unique().to_list()
    }
    if run_id not in observed_vintages:
        raise RuntimeError(
            f"Fresh Yahoo vintage has no observation from the full-ingestion run; run_id={run_id}"
        )
    return run_id


def _validated_terminal_tickers(
    *,
    requested: tuple[str, ...] | list[str],
    registry_path: Path,
    expected_through: str,
) -> tuple[str, ...]:
    requested_normalized = {str(ticker).upper().removesuffix(".US") for ticker in requested}
    if not requested_normalized:
        return ()
    registry = _read_json(registry_path)
    confirmed = {
        str(operation["ticker"]).upper().removesuffix(".US")
        for event in registry.get("events", [])
        if str(event.get("effective_date", "")) <= expected_through
        for operation in event.get("operations", [])
        if operation.get("action") == "remove"
    }
    unconfirmed = sorted(requested_normalized - confirmed)
    if unconfirmed:
        raise RuntimeError(
            f"Terminal price preservation lacks a confirmed removal event: {unconfirmed}"
        )
    return tuple(sorted(f"{ticker}.US" for ticker in requested_normalized))


def _file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
