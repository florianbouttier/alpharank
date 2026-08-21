#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.data.open_source.legacy_export import export_legacy_compatible_fundamental_outputs
from alpharank.data.quality.revision_guard import audit_historical_revisions
from alpharank.data.open_source.sec_mapping import load_sec_historical_ticker_bridge
from alpharank.data.sources.sec_only import (
    build_sec_only_earnings,
    build_sec_only_financials,
    build_sec_only_general_reference_from_raw_lineage,
)
from alpharank.data.ingestion.storage import utc_now_iso, write_json
from alpharank.data.lineage.output_history import snapshot_output_directory
from alpharank.data.security_identity import (
    SECURITY_IDENTITY_POLICY_ID,
    apply_security_identity_policy,
    apply_security_identity_reference_policy,
    load_security_identity_registry,
)

RAW_FILE_NAMES = (
    "financials_sec_companyfacts.parquet",
    "financials_sec_filing.parquet",
    "earnings_sec_calendar.parquet",
    "earnings_sec_actuals.parquet",
    "general_reference_lineage.parquet",
)
ALLOWED_FINANCIAL_SOURCES = frozenset({"sec_companyfacts", "sec_filing"})
ALLOWED_EARNINGS_SOURCES = frozenset(
    {"sec_companyfacts", "sec_filing", "sec_derived_eps", "sec_submissions"}
)


def main(
    *,
    raw_source_dir: str | Path | None = None,
    reference_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    previous_output_dir: str | Path | None = None,
    expected_through: str | None = None,
    allow_historical_revisions: bool = False,
    revision_review_note: str | None = None,
    identity_remediation_only: bool = False,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    resolved_raw_source_dir = (
        Path(raw_source_dir).expanduser().resolve()
        if raw_source_dir
        else project_root / "data" / "open_source" / "official" / "raw"
    )
    resolved_reference_data_dir = (
        Path(reference_data_dir).expanduser().resolve()
        if reference_data_dir
        else project_root / "data"
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else project_root / "data" / "sec" / "output"
    )
    history_root = resolved_output_dir.parent / "history" / "output"
    resolved_previous_output_dir = (
        Path(previous_output_dir).expanduser().resolve()
        if previous_output_dir
        else project_root / "data" / "sec" / "output"
    )
    expected_through = expected_through or date.today().isoformat()
    if allow_historical_revisions and not revision_review_note:
        raise ValueError("--revision-review-note is required when historical revisions are allowed")

    sec_companyfacts = pl.read_parquet(
        resolved_raw_source_dir / "financials_sec_companyfacts.parquet"
    )
    sec_filing = pl.read_parquet(resolved_raw_source_dir / "financials_sec_filing.parquet")
    sec_calendar = pl.read_parquet(resolved_raw_source_dir / "earnings_sec_calendar.parquet")
    sec_actuals = pl.read_parquet(resolved_raw_source_dir / "earnings_sec_actuals.parquet")
    general_reference_lineage_raw = pl.read_parquet(
        resolved_raw_source_dir / "general_reference_lineage.parquet"
    )
    historical_bridge = load_sec_historical_ticker_bridge(resolved_reference_data_dir)
    security_identities = load_security_identity_registry()
    historical_bridge_without_reused_symbols = historical_bridge.filter(
        ~pl.col("ticker").is_in(security_identities.get_column("source_ticker").unique().to_list())
    )

    sec_companyfacts = _filter_frame_by_historical_windows(
        sec_companyfacts,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        date_col="date",
    )
    sec_companyfacts = _filter_frame_by_bridge_cik(
        sec_companyfacts,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    sec_filing = _filter_frame_by_historical_windows(
        sec_filing,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        date_col="date",
    )
    sec_filing = _filter_frame_by_bridge_cik(
        sec_filing,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    sec_calendar = _filter_frame_by_historical_windows(
        sec_calendar,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        date_col="period_end",
    )
    sec_calendar = _filter_frame_by_bridge_cik(
        sec_calendar,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    sec_actuals = _filter_frame_by_historical_windows(
        sec_actuals,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        date_col="period_end",
    )
    sec_actuals = _filter_frame_by_bridge_cik(
        sec_actuals,
        bridge=historical_bridge_without_reused_symbols,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    dated_identity_results = {
        "financials_sec_companyfacts": apply_security_identity_policy(
            sec_companyfacts,
            ticker_column="ticker",
            date_column="date",
            registry=security_identities,
        ),
        "financials_sec_filing": apply_security_identity_policy(
            sec_filing,
            ticker_column="ticker",
            date_column="date",
            registry=security_identities,
        ),
        "earnings_sec_calendar": apply_security_identity_policy(
            sec_calendar,
            ticker_column="ticker",
            date_column="period_end",
            registry=security_identities,
        ),
        "earnings_sec_actuals": apply_security_identity_policy(
            sec_actuals,
            ticker_column="ticker",
            date_column="period_end",
            registry=security_identities,
        ),
    }
    sec_companyfacts = dated_identity_results["financials_sec_companyfacts"].frame
    sec_filing = dated_identity_results["financials_sec_filing"].frame
    sec_calendar = dated_identity_results["earnings_sec_calendar"].frame
    sec_actuals = dated_identity_results["earnings_sec_actuals"].frame
    general_reference_lineage_raw = _override_general_reference_lineage_from_bridge(
        general_reference_lineage_raw,
        bridge=historical_bridge_without_reused_symbols,
    )
    general_identity_result = apply_security_identity_reference_policy(
        general_reference_lineage_raw,
        ticker_column="ticker",
        registry=security_identities,
    )
    if general_identity_result.rejected.height:
        raise RuntimeError(
            "SEC general reference contains an unregistered reused-symbol CIK: "
            f"{general_identity_result.rejected.select('ticker').head(20).to_dicts()}"
        )
    general_reference_lineage_raw = general_identity_result.frame
    security_identity_report = {
        "policy_id": SECURITY_IDENTITY_POLICY_ID,
        "registry": _file_record(
            Path(security_identities.get_column("registry_path").drop_nulls().unique().item())
        ),
        "datasets": {name: result.report for name, result in dated_identity_results.items()},
        "general_reference": general_identity_result.report,
    }

    consolidated_financials, consolidated_lineage, source_summary = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_filing,
    )
    earnings_consolidated, earnings_lineage, earnings_long = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=consolidated_financials,
    )
    general_reference, general_reference_lineage = (
        build_sec_only_general_reference_from_raw_lineage(general_reference_lineage_raw)
    )
    identity_overlay_report: dict[str, object] | None = None
    if identity_remediation_only:
        (
            general_reference,
            general_reference_lineage,
            consolidated_financials,
            consolidated_lineage,
            earnings_consolidated,
            earnings_lineage,
            earnings_long,
            identity_overlay_report,
        ) = _overlay_identity_remediation(
            previous_output_dir=resolved_previous_output_dir,
            registry=security_identities,
            general_reference=general_reference,
            general_reference_lineage=general_reference_lineage,
            consolidated_financials=consolidated_financials,
            consolidated_lineage=consolidated_lineage,
            earnings_consolidated=earnings_consolidated,
            earnings_lineage=earnings_lineage,
            earnings_long=earnings_long,
        )
    _validate_sec_only_lineage(
        consolidated_lineage=consolidated_lineage,
        earnings_lineage=earnings_lineage,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    staging_dir = resolved_output_dir.parent / f"._staging_sec_output_package_{timestamp}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    legacy_paths = export_legacy_compatible_fundamental_outputs(
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=earnings_consolidated,
        reference_data_dir=resolved_reference_data_dir,
        output_dir=staging_dir,
        align_shares_with_earnings_semantics=False,
    )
    if identity_remediation_only:
        if identity_overlay_report is None:
            raise RuntimeError("Identity overlay report was not initialized")
        identity_overlay_report["legacy_outputs"] = _overlay_legacy_identity_outputs(
            previous_output_dir=resolved_previous_output_dir,
            candidate_paths=legacy_paths,
            registry=security_identities,
        )

    raw_records = {name: _file_record(resolved_raw_source_dir / name) for name in RAW_FILE_NAMES}
    source_run_ids = sorted(
        {
            str(value)
            for frame in (
                sec_companyfacts,
                sec_filing,
                sec_calendar,
                sec_actuals,
                general_reference_lineage_raw,
            )
            if "ingestion_run_id" in frame.columns
            for value in frame.get_column("ingestion_run_id").drop_nulls().unique().to_list()
        }
    )
    chronological_run_ids = [
        value for value in source_run_ids if re.fullmatch(r"\d{8}_\d{6}", value)
    ]
    run_id = max(chronological_run_ids) if chronological_run_ids else timestamp[:15]
    historical_revision_guard = audit_historical_revisions(
        previous_output_dir=resolved_previous_output_dir,
        candidate_paths=legacy_paths,
        expected_through=expected_through,
        guard_days=730,
    )
    historical_revision_guard.update(
        {
            "override_enabled": allow_historical_revisions,
            "revision_review_note": revision_review_note,
            "previous_output_dir": str(resolved_previous_output_dir),
            "candidate_dir": str(staging_dir),
        }
    )
    write_json(
        staging_dir / "lineage" / "historical_revision_guard.json",
        historical_revision_guard,
    )
    write_json(
        staging_dir / "lineage" / "security_identity_report.json",
        security_identity_report,
    )

    manifest = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "output_dir": str(resolved_output_dir),
        "raw_source_dir": str(resolved_raw_source_dir),
        "reference_data_dir": str(resolved_reference_data_dir),
        "scope": "sec_only_fundamentals",
        "source_run_ids": source_run_ids,
        "raw_sources": raw_records,
        "historical_revision_guard": historical_revision_guard,
        "security_identity": security_identity_report,
        "identity_overlay": identity_overlay_report,
        "revision_review": {
            "required": historical_revision_guard["historical_revisions_detected"],
            "approved": allow_historical_revisions,
            "note": revision_review_note,
        },
        "dataset_policy": {
            "financials": "sec_companyfacts -> sec_filing",
            "earnings_calendar": "sec_submissions",
            "earnings_actual": "sec_companyfacts -> sec_filing -> sec_derived_eps(net_income / outstanding_shares)",
            "earnings_estimate": None,
            "earnings_surprise_percent": None,
            "general_reference": "sec_mapping + sec_sic",
            "prices": None,
        },
        "semantics": {
            "reported_metrics": [
                "revenue",
                "gross_profit",
                "operating_income",
                "net_income",
                "total_assets",
                "total_liabilities",
                "stockholders_equity",
                "cash_and_equivalents",
                "operating_cash_flow",
                "capital_expenditures",
                "outstanding_shares",
                "epsActual",
            ],
            "derived_from_sec": [
                "free_cash_flow",
                "epsActual (fallback only when SEC published EPS is missing)",
            ],
            "missing_by_design": [
                "epsEstimate",
                "surprisePercent",
                "US_Finalprice.parquet",
                "SP500Price.parquet",
            ],
        },
        "summary": {
            "general_rows": general_reference.height,
            "financial_rows": consolidated_financials.height,
            "earnings_rows": earnings_consolidated.height,
            "financial_tickers": consolidated_financials.get_column("ticker").n_unique()
            if not consolidated_financials.is_empty()
            else 0,
            "earnings_tickers": earnings_consolidated.get_column("ticker").n_unique()
            if not earnings_consolidated.is_empty()
            else 0,
            "financial_date_min": consolidated_financials.get_column("date").min()
            if not consolidated_financials.is_empty()
            else None,
            "financial_date_max": consolidated_financials.get_column("date").max()
            if not consolidated_financials.is_empty()
            else None,
            "earnings_period_end_min": earnings_consolidated.get_column("period_end").min()
            if not earnings_consolidated.is_empty()
            else None,
            "earnings_period_end_max": earnings_consolidated.get_column("period_end").max()
            if not earnings_consolidated.is_empty()
            else None,
        },
    }
    if (
        historical_revision_guard["historical_revisions_detected"]
        and not allow_historical_revisions
    ):
        write_json(staging_dir / "lineage" / "candidate_manifest.json", manifest)
        raise RuntimeError(
            "SEC-only historical revisions require review before publication; "
            f"blocked_datasets={historical_revision_guard['blocked_datasets']}; "
            f"candidate={staging_dir}"
        )

    published = publish_sec_output_package(
        output_dir=resolved_output_dir,
        history_root=history_root,
        legacy_paths=legacy_paths,
        constituents_source_path=resolved_reference_data_dir / "SP500_Constituents.csv",
        general_reference=general_reference,
        general_reference_lineage=general_reference_lineage,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_consolidated=earnings_consolidated,
        earnings_lineage=earnings_lineage,
        earnings_long=earnings_long,
        manifest=manifest,
    )

    print(f"SEC-only output package written to: {resolved_output_dir}")
    print("Exact-name outputs:")
    for path in sorted(resolved_output_dir.glob("*")):
        if path.is_file():
            print(f"  - {path.name}")
    print(f"Lineage directory: {resolved_output_dir / 'lineage'}")
    if published is not None:
        print(f"Previous output snapshot: {published}")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build and publish the strict SEC-only fundamentals package."
    )
    parser.add_argument(
        "--raw-source-dir",
        type=Path,
        default=project_root / "data" / "open_source" / "official" / "raw",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=project_root / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "sec" / "output",
    )
    parser.add_argument(
        "--previous-output-dir",
        type=Path,
        default=project_root / "data" / "sec" / "output",
    )
    parser.add_argument("--expected-through", default=date.today().isoformat())
    parser.add_argument(
        "--allow-historical-revisions",
        action="store_true",
        help="Publish only after the generated SEC-only revision report was reviewed.",
    )
    parser.add_argument(
        "--revision-review-note",
        help="Required audit note explaining reviewed historical revisions.",
    )
    parser.add_argument(
        "--identity-remediation-only",
        action="store_true",
        help=(
            "Freeze all non-target rows from --previous-output-dir and replace only "
            "symbols declared in the security identity registry."
        ),
    )
    return parser.parse_args()


def _overlay_identity_remediation(
    *,
    previous_output_dir: Path,
    registry: pl.DataFrame,
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    earnings_consolidated: pl.DataFrame,
    earnings_lineage: pl.DataFrame,
    earnings_long: pl.DataFrame,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, object],
]:
    """Replace registered reused symbols without revising unrelated SEC history."""
    lineage_dir = previous_output_dir / "lineage"
    frame_specs = {
        "general_reference": (
            general_reference,
            lineage_dir / "general_reference.parquet",
        ),
        "general_reference_lineage": (
            general_reference_lineage,
            lineage_dir / "general_reference_lineage.parquet",
        ),
        "financials_sec_consolidated": (
            consolidated_financials,
            lineage_dir / "financials_sec_consolidated.parquet",
        ),
        "financials_sec_lineage": (
            consolidated_lineage,
            lineage_dir / "financials_sec_lineage.parquet",
        ),
        "earnings_sec_consolidated": (
            earnings_consolidated,
            lineage_dir / "earnings_sec_consolidated.parquet",
        ),
        "earnings_sec_lineage": (
            earnings_lineage,
            lineage_dir / "earnings_sec_lineage.parquet",
        ),
        "earnings_sec_long": (
            earnings_long,
            lineage_dir / "earnings_sec_long.parquet",
        ),
    }
    missing = [str(path) for _, path in frame_specs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Identity remediation requires a complete previous SEC lineage package: "
            + ", ".join(missing)
        )

    source_tickers, output_tickers, targeted = _identity_target_tickers(registry)
    overlaid: dict[str, pl.DataFrame] = {}
    datasets: dict[str, object] = {}
    for name, (candidate, previous_path) in frame_specs.items():
        previous = pl.read_parquet(previous_path)
        if "ticker" not in previous.columns or "ticker" not in candidate.columns:
            raise ValueError(f"Identity overlay dataset {name} must contain ticker")
        frozen = previous.filter(~pl.col("ticker").is_in(targeted))
        replacement = candidate.filter(pl.col("ticker").is_in(targeted))
        merged = pl.concat([frozen, replacement], how="diagonal_relaxed")
        identity_validation: dict[str, object] | None = None
        if "date" in merged.columns:
            merged, identity_validation = _filter_canonical_identity_intervals(
                merged,
                ticker_column="ticker",
                date_column="date",
                registry=registry,
            )
        sort_columns = [
            column
            for column in ("ticker", "date", "period_end", "filing_date", "metric")
            if column in merged.columns
        ]
        if sort_columns:
            merged = merged.sort(sort_columns)
        overlaid[name] = merged
        datasets[name] = {
            "previous_rows": previous.height,
            "frozen_non_target_rows": frozen.height,
            "candidate_target_rows": replacement.height,
            "published_rows": merged.height,
            "previous_sha256": _sha256(previous_path),
            "identity_validation": identity_validation,
        }

    report = {
        "mode": "registered_security_identity_only",
        "previous_output_dir": str(previous_output_dir),
        "source_tickers": source_tickers,
        "published_tickers": output_tickers,
        "datasets": datasets,
    }
    return (
        overlaid["general_reference"],
        overlaid["general_reference_lineage"],
        overlaid["financials_sec_consolidated"],
        overlaid["financials_sec_lineage"],
        overlaid["earnings_sec_consolidated"],
        overlaid["earnings_sec_lineage"],
        overlaid["earnings_sec_long"],
        report,
    )


def _overlay_legacy_identity_outputs(
    *,
    previous_output_dir: Path,
    candidate_paths: dict[str, Path],
    registry: pl.DataFrame,
) -> dict[str, object]:
    """Keep published non-target legacy rows byte-equivalent at the data level."""
    _, _, targeted = _identity_target_tickers(registry)
    report: dict[str, object] = {}
    for file_name, candidate_path in candidate_paths.items():
        previous_path = previous_output_dir / file_name
        if not previous_path.is_file():
            raise FileNotFoundError(
                f"Identity remediation requires previous legacy output {previous_path}"
            )
        previous = pl.read_parquet(previous_path)
        candidate = pl.read_parquet(candidate_path)
        ticker_column = "Ticker" if "Ticker" in previous.columns else "ticker"
        if ticker_column not in candidate.columns:
            raise ValueError(f"Identity overlay output {file_name} must contain {ticker_column}")
        frozen = previous.filter(~pl.col(ticker_column).is_in(targeted))
        replacement = candidate.filter(pl.col(ticker_column).is_in(targeted))
        merged = pl.concat([frozen, replacement], how="diagonal_relaxed")
        identity_validation: dict[str, object] | None = None
        if "date" in merged.columns:
            merged, identity_validation = _filter_canonical_identity_intervals(
                merged,
                ticker_column=ticker_column,
                date_column="date",
                registry=registry,
            )
        sort_columns = [
            column
            for column in (ticker_column, "date", "reportDate", "filing_date")
            if column in merged.columns
        ]
        if sort_columns:
            merged = merged.sort(sort_columns)
        merged.write_parquet(candidate_path)
        report[file_name] = {
            "previous_rows": previous.height,
            "frozen_non_target_rows": frozen.height,
            "candidate_target_rows": replacement.height,
            "published_rows": merged.height,
            "previous_sha256": _sha256(previous_path),
            "identity_validation": identity_validation,
        }
    return report


def _identity_target_tickers(
    registry: pl.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    source_tickers = registry.get_column("source_ticker").unique().to_list()
    output_tickers = sorted(
        {
            f"{value}.US"
            for value in registry.get_column("canonical_ticker").drop_nulls().unique().to_list()
        }
    )
    source_roots = {f"{value}.US" for value in source_tickers}
    targeted = sorted(source_roots | set(output_tickers))
    return source_tickers, output_tickers, targeted


def _filter_canonical_identity_intervals(
    frame: pl.DataFrame,
    *,
    ticker_column: str,
    date_column: str,
    registry: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Drop canonical identity rows that fall outside their own validity interval."""
    original_columns = frame.columns
    windows = registry.select(
        (pl.col("canonical_ticker") + pl.lit(".US")).alias("_identity_ticker"),
        pl.col("valid_from").cast(pl.String).str.to_date(strict=False).alias("_valid_from"),
        pl.col("valid_to").cast(pl.String).str.to_date(strict=False).alias("_valid_to"),
    )
    canonical_tickers = windows.get_column("_identity_ticker").to_list()
    working = frame.with_row_index("_identity_row").with_columns(
        pl.col(ticker_column).cast(pl.String).str.to_uppercase().alias("_identity_ticker"),
        pl.col(date_column).cast(pl.String).str.to_date(strict=False).alias("_identity_date"),
    )
    targeted = working.filter(pl.col("_identity_ticker").is_in(canonical_tickers))
    accepted = targeted.join(windows, on="_identity_ticker", how="inner").filter(
        pl.col("_identity_date").is_not_null()
        & (pl.col("_identity_date") >= pl.col("_valid_from"))
        & (pl.col("_valid_to").is_null() | (pl.col("_identity_date") <= pl.col("_valid_to")))
    )
    accepted_rows = accepted.get_column("_identity_row").to_list()
    rejected = targeted.filter(~pl.col("_identity_row").is_in(accepted_rows))
    published = working.filter(
        ~pl.col("_identity_ticker").is_in(canonical_tickers)
        | pl.col("_identity_row").is_in(accepted_rows)
    ).select(original_columns)
    return published, {
        "policy_id": SECURITY_IDENTITY_POLICY_ID,
        "targeted_rows": targeted.height,
        "accepted_rows": len(accepted_rows),
        "rejected_rows": rejected.height,
        "canonical_tickers": canonical_tickers,
    }


def publish_sec_output_package(
    *,
    output_dir: Path,
    history_root: Path,
    legacy_paths: dict[str, Path],
    constituents_source_path: Path,
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    source_summary: pl.DataFrame,
    earnings_consolidated: pl.DataFrame,
    earnings_lineage: pl.DataFrame,
    earnings_long: pl.DataFrame,
    manifest: dict[str, object],
) -> Path | None:
    candidate_dirs = {path.parent.resolve() for path in legacy_paths.values()}
    if len(candidate_dirs) != 1:
        raise RuntimeError("SEC legacy outputs must share one staging directory")
    candidate_dir = next(iter(candidate_dirs))
    lineage_dir = candidate_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    allowed_output_files = set(legacy_paths) | {"SP500_Constituents.csv", "README.md"}
    for existing in candidate_dir.iterdir():
        if existing.name == "lineage" or existing.name in allowed_output_files:
            continue
        if existing.is_file():
            existing.unlink()

    for file_name, source_path in legacy_paths.items():
        if source_path.resolve() != (candidate_dir / file_name).resolve():
            shutil.copy2(source_path, candidate_dir / file_name)
    shutil.copy2(constituents_source_path, candidate_dir / "SP500_Constituents.csv")
    _write_readme(candidate_dir / "README.md")

    lineage_outputs = {
        "general_reference.parquet": general_reference,
        "general_reference_lineage.parquet": general_reference_lineage,
        "financials_sec_consolidated.parquet": consolidated_financials,
        "financials_sec_lineage.parquet": consolidated_lineage,
        "financials_sec_source_summary.parquet": source_summary,
        "earnings_sec_consolidated.parquet": earnings_consolidated,
        "earnings_sec_lineage.parquet": earnings_lineage,
        "earnings_sec_long.parquet": earnings_long,
    }
    allowed_lineage_files = set(lineage_outputs) | {
        "historical_revision_guard.json",
        "manifest.json",
        "security_identity_report.json",
    }
    for existing in lineage_dir.iterdir():
        if existing.name in allowed_lineage_files:
            continue
        if existing.is_file():
            existing.unlink()
    for file_name, frame in lineage_outputs.items():
        frame.write_parquet(lineage_dir / file_name)
    write_json(lineage_dir / "manifest.json", manifest)
    snapshot_dir = snapshot_output_directory(
        candidate_dir,
        history_root=history_root,
        snapshot_prefix="sec_output",
        metadata=manifest,
    )
    if snapshot_dir is None:
        raise RuntimeError("SEC candidate snapshot was not created")
    _replace_directory_atomically(source_dir=candidate_dir, output_dir=output_dir)
    return snapshot_dir


def _validate_sec_only_lineage(
    *,
    consolidated_lineage: pl.DataFrame,
    earnings_lineage: pl.DataFrame,
) -> None:
    financial_source_col = (
        "selected_source" if "selected_source" in consolidated_lineage.columns else "source"
    )
    financial_sources = set(
        consolidated_lineage.get_column(financial_source_col)
        .drop_nulls()
        .cast(pl.String)
        .unique()
        .to_list()
    )
    forbidden_financial = sorted(financial_sources - ALLOWED_FINANCIAL_SOURCES)
    if forbidden_financial:
        raise RuntimeError(
            f"SEC-only financial lineage contains forbidden sources: {forbidden_financial}"
        )
    for column in ("calendar_source", "actual_source"):
        if column not in earnings_lineage.columns:
            continue
        sources = set(
            earnings_lineage.get_column(column).drop_nulls().cast(pl.String).unique().to_list()
        )
        forbidden = sorted(sources - ALLOWED_EARNINGS_SOURCES)
        if forbidden:
            raise RuntimeError(
                f"SEC-only earnings lineage contains forbidden {column}: {forbidden}"
            )


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_directory_atomically(*, source_dir: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir.parent / f".{output_dir.name}.previous"
    shutil.rmtree(backup_dir, ignore_errors=True)
    if output_dir.exists():
        os.replace(output_dir, backup_dir)
    try:
        os.replace(source_dir, output_dir)
    except (OSError, RuntimeError):
        if backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _filter_frame_by_historical_windows(
    frame: pl.DataFrame,
    *,
    bridge: pl.DataFrame,
    ticker_col: str,
    date_col: str,
) -> pl.DataFrame:
    if (
        frame.is_empty()
        or bridge.is_empty()
        or ticker_col not in frame.columns
        or date_col not in frame.columns
    ):
        return frame
    windows = (
        bridge.select(
            [
                (pl.col("ticker") + pl.lit(".US")).alias(ticker_col),
                pl.col("start_date").cast(pl.Utf8),
                pl.col("end_date").cast(pl.Utf8),
            ]
        )
        .with_columns(
            [
                pl.col("start_date").str.strptime(pl.Date, strict=False).alias("_start_dt"),
                pl.col("end_date").str.strptime(pl.Date, strict=False).alias("_end_dt"),
            ]
        )
        .drop(["start_date", "end_date"])
    )
    return (
        frame.with_columns(pl.col(date_col).str.strptime(pl.Date, strict=False).alias("_row_dt"))
        .join(windows, on=ticker_col, how="left")
        .filter(
            pl.col("_row_dt").is_null()
            | (
                (pl.col("_start_dt").is_null() | (pl.col("_row_dt") >= pl.col("_start_dt")))
                & (pl.col("_end_dt").is_null() | (pl.col("_row_dt") <= pl.col("_end_dt")))
            )
        )
        .drop(["_row_dt", "_start_dt", "_end_dt"])
    )


def _override_general_reference_lineage_from_bridge(
    raw_lineage: pl.DataFrame,
    *,
    bridge: pl.DataFrame,
) -> pl.DataFrame:
    if raw_lineage.is_empty() or bridge.is_empty() or "ticker" not in raw_lineage.columns:
        return raw_lineage
    overrides = bridge.select(
        [
            (pl.col("ticker") + pl.lit(".US")).alias("ticker"),
            pl.col("name").alias("_bridge_name"),
            pl.col("exchange").alias("_bridge_exchange"),
            pl.col("cik").alias("_bridge_cik"),
        ]
    )
    result = raw_lineage.join(overrides, on="ticker", how="left")
    updates: list[pl.Expr] = []
    if "name" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_name"), pl.col("name")]).alias("name"))
    if "exchange" in result.columns:
        updates.append(
            pl.coalesce([pl.col("_bridge_exchange"), pl.col("exchange")]).alias("exchange")
        )
    if "cik" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_cik"), pl.col("cik")]).alias("cik"))
    if "sec_name" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_name"), pl.col("sec_name")]).alias("sec_name"))
    if "sec_exchange" in result.columns:
        updates.append(
            pl.coalesce([pl.col("_bridge_exchange"), pl.col("sec_exchange")]).alias("sec_exchange")
        )
    if "sec_cik" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_cik"), pl.col("sec_cik")]).alias("sec_cik"))
    if updates:
        result = result.with_columns(updates)
    return result.drop(
        [
            column
            for column in ["_bridge_name", "_bridge_exchange", "_bridge_cik"]
            if column in result.columns
        ]
    )


def _filter_frame_by_bridge_cik(
    frame: pl.DataFrame,
    *,
    bridge: pl.DataFrame,
    ticker_col: str,
    accession_col: str,
) -> pl.DataFrame:
    if (
        frame.is_empty()
        or bridge.is_empty()
        or ticker_col not in frame.columns
        or accession_col not in frame.columns
    ):
        return frame
    expected = bridge.select(
        [
            (pl.col("ticker") + pl.lit(".US")).alias(ticker_col),
            pl.col("cik").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10).alias("_bridge_cik"),
        ]
    ).unique(subset=[ticker_col], keep="first")
    return (
        frame.join(expected, on=ticker_col, how="left")
        .with_columns(
            pl.col(accession_col).cast(pl.Utf8).str.extract(r"(\d{10})").alias("_accession_cik")
        )
        .filter(
            pl.col("_bridge_cik").is_null()
            | pl.col("_accession_cik").is_null()
            | (pl.col("_bridge_cik") == pl.col("_accession_cik"))
        )
        .drop(["_bridge_cik", "_accession_cik"])
    )


def _write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# SEC Output Package",
                "",
                "Official strict SEC-only fundamentals package.",
                "",
                "- Exact-name legacy files are published only for fundamentals and earnings.",
                "- No price files are published here because SEC is not a market price source.",
                "- `lineage/manifest.json` is the official policy and provenance entrypoint.",
                "- `free_cash_flow` is derived from SEC cash-flow components when available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = _parse_args()
    main(
        raw_source_dir=args.raw_source_dir,
        reference_data_dir=args.reference_data_dir,
        output_dir=args.output_dir,
        previous_output_dir=args.previous_output_dir,
        expected_through=args.expected_through,
        allow_historical_revisions=args.allow_historical_revisions,
        revision_review_note=args.revision_review_note,
        identity_remediation_only=args.identity_remediation_only,
    )
