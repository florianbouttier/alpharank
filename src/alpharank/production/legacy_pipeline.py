"""Testable implementation of the canonical monthly Legacy pipeline."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import polars as pl

from alpharank.data.contracts.ticker_integrity import (
    DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
    exclude_tickers_from_frame,
    load_ticker_exclusion_registry,
    normalize_tickers,
)
from alpharank.data.lineage.snapshots import load_latest_manifest, write_manifest
from alpharank.data.price_eligibility import (
    STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY,
    build_monthly_price_eligibility,
    monthly_price_eligibility_policy,
)
from alpharank.data.processing import IndexDataManager, PricesDataPreprocessor
from alpharank.data.publishing.snapshot_publication import (
    SNAPSHOT_POINTER_CONTRACT,
    validate_snapshot_publication,
)
from alpharank.data.publishing.snapshot_storage import copy_snapshot_file
from alpharank.data.warehouse.mart import MartInputResolution, resolve_mart_model_input
from alpharank.features.indicators import TechnicalIndicators
from alpharank.governance import capture_runtime_provenance, reserve_run_directory
from alpharank.observability import get_run_logger, set_run_log_context
from alpharank.portfolio.adapters.legacy import legacy_detailed_to_holdings
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.benchmark import (
    SPY_TOTAL_RETURN,
    completed_through_month,
    monthly_benchmark_returns,
)
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.costs import TransactionCostModel
from alpharank.portfolio.execution import (
    ALPHARANK_REFERENCE_CLOSE,
    LEGACY_NEXT_SESSION_OPEN,
    apply_next_session_open_holding_returns,
    build_execution_sensitivity_report,
    build_monthly_execution_orders,
    write_execution_sensitivity_report,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.terminal_event_registry import load_terminal_event_registry
from alpharank.replay import validate_causal_v2_snapshot
from alpharank.replay.legacy import (
    HOLDING_MONTH_MEMBERSHIP_POLICY_ID,
    require_holding_month_membership,
)
from alpharank.strategy.legacy import ModelEvaluator, StrategyLearner
from alpharank.strategy.legacy_valuation import (
    LEGACY_PE_MARKET_CAP_POLICY_ID,
    build_legacy_selection_universe,
)
from alpharank.strategy.search_protocol import write_legacy_search_audit
from alpharank.utils.frame_backend import (
    normalize_year_month_to_period,
    normalize_year_month_to_timestamp,
    to_pandas,
    to_polars,
)
from alpharank.visualization.plotting import PortfolioVisualizer

INPUT_PACKAGE_FILENAMES: Dict[str, str] = {
    "final_price": "US_Finalprice.parquet",
    "general": "US_General.parquet",
    "income_statement": "US_Income_statement.parquet",
    "balance_sheet": "US_Balance_sheet.parquet",
    "cash_flow": "US_Cash_flow.parquet",
    "earnings": "US_Earnings.parquet",
    "sp500_constituents": "SP500_Constituents.csv",
    "sp500_price": "SP500Price.parquet",
}

LEGACY_V2_COST_SCENARIOS = (
    TransactionCostModel("zero"),
    TransactionCostModel(
        "standard_10bps",
        spread_bps=3.0,
        slippage_bps=2.0,
        impact_bps=2.0,
        commission_bps=2.0,
        fx_bps=1.0,
        fx_turnover_fraction=1.0,
    ),
    TransactionCostModel(
        "stress_30bps",
        spread_bps=8.0,
        slippage_bps=7.0,
        impact_bps=8.0,
        commission_bps=5.0,
        fx_bps=2.0,
        fx_turnover_fraction=1.0,
    ),
)

HISTORICAL_LEGACY_MISSING_RETURN_POLICY = "renormalize_available"
LOGGER = get_run_logger(__name__)


@dataclass
class PipelineOutput:
    monthly_return: pl.DataFrame
    final_price_vs_index: pl.DataFrame
    stocks_selections: pl.DataFrame
    optuna_outputs: Dict[str, Dict[str, Any]]
    combined_equal: Dict[str, Any]
    combined_frequency: Dict[str, Any]
    metrics: Any
    artifacts: Dict[str, Path]


def _load_data(
    data_dir: Path,
    *,
    final_price_path: Path | None = None,
    sp500_price_path: Path | None = None,
) -> Dict[str, pl.DataFrame]:
    LOGGER.info("Loading Legacy input data", extra={"result": "started"})
    return {
        "final_price": pl.read_parquet(final_price_path or (data_dir / "US_Finalprice.parquet")),
        "general": pl.read_parquet(data_dir / "US_General.parquet"),
        "income_statement": pl.read_parquet(data_dir / "US_Income_statement.parquet"),
        "balance_sheet": pl.read_parquet(data_dir / "US_Balance_sheet.parquet"),
        "cash_flow": pl.read_parquet(data_dir / "US_Cash_flow.parquet"),
        "earnings": pl.read_parquet(data_dir / "US_Earnings.parquet"),
        "us_historical_company": pl.read_csv(data_dir / "SP500_Constituents.csv", try_parse_dates=True),
        "sp500_price": pl.read_parquet(sp500_price_path or (data_dir / "SP500Price.parquet")),
    }


def _input_files(data_dir: Path, *, final_price_path: Path | None = None, sp500_price_path: Path | None = None) -> Dict[str, Path]:
    return {
        "final_price": final_price_path or (data_dir / "US_Finalprice.parquet"),
        "general": data_dir / "US_General.parquet",
        "income_statement": data_dir / "US_Income_statement.parquet",
        "balance_sheet": data_dir / "US_Balance_sheet.parquet",
        "cash_flow": data_dir / "US_Cash_flow.parquet",
        "earnings": data_dir / "US_Earnings.parquet",
        "sp500_constituents": data_dir / "SP500_Constituents.csv",
        "sp500_price": sp500_price_path or (data_dir / "SP500Price.parquet"),
    }


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_snapshot_file(source, destination)


def _copy_snapshot_file(source: Path | str, destination: Path | str) -> str:
    """Backward-compatible wrapper for the shared snapshot storage helper."""

    return copy_snapshot_file(source, destination)


def _snapshot_input_package(*, source_data_dir: Path, input_files: Dict[str, Path], run_day_dir: Path) -> Path:
    snapshot_dir = run_day_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    storage_modes: list[str] = []
    for name, source_path in input_files.items():
        target_name = INPUT_PACKAGE_FILENAMES[name]
        storage_modes.append(copy_snapshot_file(source_path, snapshot_dir / target_name))

    lineage_dir = source_data_dir / "lineage"
    if lineage_dir.exists():
        def copy_lineage_file(source: str, destination: str) -> str:
            storage_modes.append(copy_snapshot_file(source, destination))
            return destination

        shutil.copytree(
            lineage_dir,
            snapshot_dir / "lineage",
            dirs_exist_ok=True,
            copy_function=copy_lineage_file,
        )
    for metadata_name in ("snapshot_manifest.json", "latest_snapshot.json", "README.md"):
        source = source_data_dir / metadata_name
        if source.exists():
            storage_modes.append(copy_snapshot_file(source, snapshot_dir / metadata_name))
    storage_manifest = {
        "strategy": "copy_on_write_with_physical_copy_fallback",
        "semantics": "independent path with byte-identical content; APFS clones are copy-on-write",
        "source_data_dir": str(source_data_dir.resolve()),
        "file_count": len(storage_modes),
        "storage_mode_counts": {
            mode: storage_modes.count(mode) for mode in sorted(set(storage_modes))
        },
    }
    (snapshot_dir / "storage_manifest.json").write_text(
        json.dumps(storage_manifest, indent=2),
        encoding="utf-8",
    )
    return snapshot_dir


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_identifier(data_dir: Path) -> str | None:
    """Resolve a stable identity from composed or historical snapshot metadata."""

    for relative in (
        "snapshot_manifest.json",
        "lineage/manifest.json",
        "latest_snapshot.json",
    ):
        payload = _read_json_if_exists(data_dir / relative)
        if not isinstance(payload, dict):
            continue
        for key in ("snapshot_id", "composition_id", "run_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
    return None


def _resolve_published_output_snapshot(official_dir: str | None, published_snapshot: str | None) -> Path | None:
    if not official_dir or not published_snapshot:
        return None
    snapshot_path = Path(published_snapshot)
    if snapshot_path.is_absolute():
        return snapshot_path
    official_path = Path(official_dir)
    return official_path.parent / snapshot_path


def _package_file_hashes(data_dir: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for file_name in INPUT_PACKAGE_FILENAMES.values():
        path = data_dir / file_name
        if path.exists():
            hashes[file_name] = _sha256_path(path)
    return hashes


def _open_source_lineage_context(data_dir: Path) -> Dict[str, Any]:
    output_manifest_path = data_dir / "lineage" / "manifest.json"
    output_manifest = _read_json_if_exists(output_manifest_path)
    if output_manifest is None:
        return {}
    snapshot_manifest_path = data_dir / "snapshot_manifest.json"
    snapshot_manifest = _read_json_if_exists(snapshot_manifest_path)

    lineage_run_id = output_manifest.get("run_id")
    snapshot_run_id = snapshot_manifest.get("run_id") if snapshot_manifest else None
    output_run_id = snapshot_run_id or lineage_run_id

    context: Dict[str, Any] = {
        "open_source_output_manifest_path": str(output_manifest_path.resolve()),
        "open_source_output_run_id": output_run_id,
        "open_source_output_lineage_run_id": lineage_run_id,
        "open_source_official_dir": output_manifest.get("official_dir"),
        "open_source_target_dir": output_manifest.get("target_dir"),
        "open_source_output_dir": output_manifest.get("output_dir"),
        "open_source_legacy_dir": output_manifest.get("legacy_dir"),
    }
    output_refresh_contract = output_manifest.get("source_refresh_contract")
    if isinstance(output_refresh_contract, dict):
        context["open_source_source_refresh_contract"] = output_refresh_contract
        context["open_source_source_refresh_scope"] = output_refresh_contract.get("snapshot_scope")
    output_freshness = output_manifest.get("data_freshness")
    if isinstance(output_freshness, dict):
        context["open_source_data_freshness"] = output_freshness
    if snapshot_manifest is not None:
        context["open_source_output_snapshot_manifest_path"] = str(snapshot_manifest_path.resolve())
        context["open_source_output_snapshot_run_id"] = snapshot_run_id
    if snapshot_run_id and lineage_run_id:
        context["open_source_output_manifest_run_id_match"] = snapshot_run_id == lineage_run_id

    official_dir = output_manifest.get("official_dir")
    run_manifest_path = Path(official_dir) / "runs" / str(output_run_id) / "manifest.json" if official_dir and output_run_id else None
    run_manifest = _read_json_if_exists(run_manifest_path) if run_manifest_path else None
    latest_run_manifest_path = Path(official_dir) / "manifests" / "latest_run.json" if official_dir and run_manifest is None else None
    latest_run = _read_json_if_exists(latest_run_manifest_path) if latest_run_manifest_path else None
    ingestion_manifest = run_manifest or latest_run
    ingestion_manifest_path = run_manifest_path if run_manifest is not None else latest_run_manifest_path
    if ingestion_manifest is not None and ingestion_manifest_path is not None:
        context.update(
            {
                "open_source_ingestion_manifest_path": str(ingestion_manifest_path.resolve()),
                "open_source_ingestion_run_id": ingestion_manifest.get("run_id"),
                "open_source_ingested_at": (
                    ingestion_manifest.get("ingested_at")
                    or ingestion_manifest.get("generated_at")
                ),
                "open_source_mode": ingestion_manifest.get("mode"),
                "open_source_price_window": ingestion_manifest.get("price_window"),
                "open_source_financial_years_refreshed": ingestion_manifest.get("financial_years_refreshed"),
                "open_source_ticker_count": ingestion_manifest.get("ticker_count"),
                "open_source_sec_companyfacts_years_refreshed": ingestion_manifest.get(
                    "sec_companyfacts_years_refreshed"
                ),
            }
        )
        published_snapshot = ingestion_manifest.get("published_output_snapshot")
        published_snapshot_dir = _resolve_published_output_snapshot(official_dir, published_snapshot)
        if published_snapshot_dir is not None:
            context["open_source_ingestion_published_output_snapshot"] = str(published_snapshot_dir.resolve())
            if published_snapshot_dir.exists():
                current_hashes = _package_file_hashes(data_dir)
                published_hashes = _package_file_hashes(published_snapshot_dir)
                compared_files = sorted(set(current_hashes) | set(published_hashes))
                differing_files = [
                    file_name
                    for file_name in compared_files
                    if current_hashes.get(file_name) != published_hashes.get(file_name)
                ]
                context["open_source_output_matches_published_snapshot"] = not differing_files
                context["open_source_output_published_snapshot_differing_files"] = differing_files
            else:
                context["open_source_output_matches_published_snapshot"] = False
                context["open_source_output_published_snapshot_differing_files"] = ["<missing snapshot directory>"]

    output_run_id = context.get("open_source_output_run_id")
    ingestion_run_id = context.get("open_source_ingestion_run_id")
    if output_run_id and ingestion_run_id:
        context["open_source_run_id_match"] = output_run_id == ingestion_run_id

    return {key: value for key, value in context.items() if value is not None}


def _manifest_extra_context(
    *,
    data_dir: Path,
    latest_snapshot: Dict[str, Any] | None,
    source_data_dir: Path | None = None,
    input_snapshot_dir: Path | None = None,
    run_config: Dict[str, Any] | None = None,
    code_context: Dict[str, Any] | None = None,
    runtime_provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "consumer": "scripts.run_legacy",
    }
    if source_data_dir is not None:
        extra["source_data_dir"] = str(source_data_dir.resolve())
    if input_snapshot_dir is not None:
        extra["input_snapshot_dir"] = str(input_snapshot_dir.resolve())
    if run_config is not None:
        extra["run_config"] = run_config
    if code_context is not None:
        extra["code_context"] = code_context
    if runtime_provenance is not None:
        extra["runtime_provenance"] = runtime_provenance
    if latest_snapshot is not None:
        extra.update(
            {
                "source_snapshot_id": latest_snapshot.get("snapshot_id"),
                "source_snapshot_generated_at": latest_snapshot.get("generated_at"),
                "source_snapshot_dir": latest_snapshot.get("snapshot_dir"),
                "source_snapshot_manifest_path": latest_snapshot.get("manifest_path"),
            }
        )
    extra.update(_open_source_lineage_context(data_dir))
    return {key: value for key, value in extra.items() if value is not None}


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning(
            "Git commit identifier is unavailable",
            extra={"result": "unavailable", "error": str(exc)},
        )
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _simulate_historical_legacy_common(common_holdings: pl.DataFrame) -> pl.DataFrame:
    """Replay the frozen Legacy convention without weakening causal-v2 runs."""

    return pl.concat(
        [
            simulate_weighted_portfolio(
                strategy_holdings,
                transaction_cost_bps=0.0,
                missing_return_policy=HISTORICAL_LEGACY_MISSING_RETURN_POLICY,
                causal_timing_policy="legacy_month_only",
            )
            for strategy_holdings in common_holdings.partition_by(
                "strategy",
                maintain_order=True,
            )
        ]
    )

def _code_context(project_root: Path) -> Dict[str, Any]:
    critical_files = [
        "scripts/run_legacy.py",
        "src/alpharank/production/legacy_pipeline.py",
        "src/alpharank/data/processing.py",
        "src/alpharank/data/warehouse/mart.py",
        "src/alpharank/strategy/legacy.py",
        "src/alpharank/data/open_source/legacy_export.py",
        "src/alpharank/data/ingestion/cadrage.py",
        "src/alpharank/data/publishing/open_source_package.py",
        "src/alpharank/data/terminal_eligibility.py",
        "src/alpharank/portfolio/terminal_event_registry.py",
        "configs/data_quality/terminal_shareholder_events_v1.json",
        "configs/data_quality/terminal_shareholder_events_v2.json",
    ]
    file_hashes = {
        path: _sha256_path(project_root / path)
        for path in critical_files
        if (project_root / path).exists()
    }
    git_status = _git_output(project_root, "status", "--short")
    return {
        "git_head": _git_output(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status_short": git_status,
        "critical_file_sha256": file_hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "polars": _package_version("polars"),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "optuna": _package_version("optuna"),
        },
    }


def _open_source_output_run_id(data_dir: Path) -> str | None:
    try:
        snapshot_manifest = _read_json_if_exists(data_dir / "snapshot_manifest.json")
        if snapshot_manifest and snapshot_manifest.get("run_id"):
            return snapshot_manifest.get("run_id")
        manifest = _read_json_if_exists(data_dir / "lineage" / "manifest.json")
    except json.JSONDecodeError:
        return None
    return None if manifest is None else manifest.get("run_id")


def _resolve_open_source_output_by_run_id(project_root: Path, run_id: str) -> Path:
    current_output_dir = project_root / "data" / "open_source" / "output"

    history_root = project_root / "data" / "open_source" / "history" / "output"
    for output_dir in sorted(history_root.glob("open_source_output_*"), reverse=True):
        if _open_source_output_run_id(output_dir) == run_id:
            return output_dir

    if _open_source_output_run_id(current_output_dir) == run_id:
        return current_output_dir

    raise FileNotFoundError(
        f"Open-source output package for run_id={run_id!r} was not found under "
        f"{current_output_dir} or {history_root}."
    )


def _write_checkpoint(df: Any, checkpoints_dir: Optional[Path], name: str) -> None:
    if checkpoints_dir is None:
        return
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoints_dir / f"{name}.parquet"
    to_polars(df).write_parquet(out)


def _save_html(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def _sort_monthly_frame(df: Any) -> pl.DataFrame:
    out = to_polars(df)
    sort_cols = [c for c in ["year_month", "model", "ticker"] if c in out.columns]
    if sort_cols:
        out = out.sort(sort_cols)
    return out


def _named_frames_to_long(named_frames: Dict[str, Any], *, label_col: str = "model") -> pl.DataFrame:
    has_label_collision = any(label_col in _sort_monthly_frame(frame).columns for frame in named_frames.values())
    effective_label_col = f"portfolio_{label_col}" if has_label_collision else label_col

    frames: list[pl.DataFrame] = []
    for label, frame in named_frames.items():
        sorted_frame = _sort_monthly_frame(frame)
        if sorted_frame.is_empty():
            continue
        frames.append(
            sorted_frame.with_columns(pl.lit(label).alias(effective_label_col)).select(
                [effective_label_col, *sorted_frame.columns]
            )
        )

    if not frames:
        return pl.DataFrame(schema={effective_label_col: pl.Utf8})

    combined = pl.concat(frames, how="diagonal_relaxed")
    sort_cols = [c for c in [effective_label_col, "year_month", "ticker"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort(sort_cols)
    return combined


def _indexed_frame_to_polars(df: Any, *, index_name: str = "year_month") -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df

    if isinstance(df, pd.Series):
        value_name = df.name if df.name is not None else "value"
        out = df.rename(value_name).rename_axis(index_name).reset_index()
    elif isinstance(df, pd.DataFrame):
        out = df.reset_index(names=index_name) if df.index.name is not None else df.reset_index()
    else:
        out = pd.DataFrame(df)

    out = out.rename(columns={"index": index_name})
    return to_polars(out)


def _get_detailed_output(output: Dict[str, Any], *, label: str | None = None) -> Any:
    if "detailed" in output:
        return output["detailed"]
    if "detailled" in output:
        return output["detailled"]
    suffix = f" for {label}" if label else ""
    raise KeyError(f"Missing `detailed`/`detailled` portfolio output{suffix}.")


def _write_artifact_frame(frame: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _sort_monthly_frame(frame).write_parquet(output_path)
    return output_path


def _extract_start_year(first_date: str) -> int:
    try:
        return int(str(first_date).split("-")[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --first-date format: {first_date!r}. Expected YYYY-MM.") from exc


def _max_file_mtime(paths: Dict[str, Path]) -> str:
    latest = max(path.stat().st_mtime for path in paths.values())
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")


def _max_date_str(df: pl.DataFrame, col: str) -> str:
    if col not in df.columns or df.is_empty():
        return "n/a"
    value = df.select(pl.col(col).max().alias("max_value")).item()
    return "n/a" if value is None else str(value)


def _month_view_date(month: Any) -> str:
    try:
        period = pd.Period(str(month), freq="M")
        last_day = monthrange(period.year, period.month)[1]
        return f"{period.year:04d}-{period.month:02d}-{last_day:02d}"
    except (TypeError, ValueError) as exc:
        LOGGER.warning(
            "Month label could not be normalized for the report",
            extra={"month": str(month), "result": "fallback", "error": str(exc)},
        )
        return str(month)


def _build_report_context(
    *,
    month: Any,
    input_files: Dict[str, Path],
    run_manifest: Dict[str, Any] | None,
    final_price: pl.DataFrame,
    sp500_price: pl.DataFrame,
    us_historical_company: pl.DataFrame,
    income_statement: pl.DataFrame,
    balance_sheet: pl.DataFrame,
    cash_flow: pl.DataFrame,
    earnings: pl.DataFrame,
) -> Dict[str, str]:
    context = {
        "portfolio_month": str(month),
        "portfolio_view_date": _month_view_date(month),
        "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_snapshot_at": _max_file_mtime(input_files),
        "price_data_max_date": _max_date_str(final_price, "date"),
        "sp500_price_max_date": _max_date_str(sp500_price, "date"),
        "sp500_constituents_max_month": _max_date_str(us_historical_company, "year_month"),
        "income_statement_max_date": _max_date_str(income_statement, "date"),
        "balance_sheet_max_date": _max_date_str(balance_sheet, "date"),
        "cash_flow_max_date": _max_date_str(cash_flow, "date"),
        "earnings_max_date": _max_date_str(earnings, "date"),
    }
    if run_manifest is not None:
        context["data_snapshot_id"] = str(run_manifest.get("snapshot_id", "n/a"))
        source_generated_at = run_manifest.get("source_snapshot_generated_at")
        if source_generated_at:
            context["source_snapshot_generated_at"] = str(source_generated_at)
        for key in (
            "open_source_output_run_id",
            "open_source_ingestion_run_id",
            "open_source_ingested_at",
            "open_source_mode",
            "open_source_ticker_count",
            "open_source_run_id_match",
        ):
            value = run_manifest.get(key)
            if value is not None:
                context[key] = str(value)
        price_window = run_manifest.get("open_source_price_window")
        if isinstance(price_window, dict):
            start_date = price_window.get("start_date")
            end_date = price_window.get("end_date")
            if start_date or end_date:
                context["open_source_price_window"] = f"{start_date or 'n/a'} -> {end_date or 'n/a'}"
        financial_years = run_manifest.get("open_source_financial_years_refreshed")
        if financial_years:
            context["open_source_financial_years_refreshed"] = ", ".join(str(year) for year in financial_years)
    return context


def run_pipeline(
    *,
    n_trials: int,
    n_jobs: int,
    first_date: str,
    data_dir: Optional[Path] = None,
    open_source_run_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    final_price_path: Optional[Path] = None,
    sp500_price_path: Optional[Path] = None,
    methodology_manifest: Optional[Path] = None,
    ticker_exclusion_registry: Optional[Path] = DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
    price_eligibility_policy_id: str = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.policy_id,
    minimum_monthly_price_observations: int = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_observations,
    minimum_monthly_median_dollar_volume: float = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.minimum_median_dollar_volume,
    maximum_monthly_ohlc_violation_rate: float = STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.maximum_ohlc_violation_rate,
    fundamental_eligibility_policy_id: str = LEGACY_PE_MARKET_CAP_POLICY_ID,
) -> PipelineOutput:
    backend = "polars"
    project_root = Path(__file__).resolve().parents[3]
    mart_input: MartInputResolution | None = None
    if data_dir is not None and open_source_run_id is not None:
        raise ValueError("Use either data_dir or open_source_run_id, not both.")
    if open_source_run_id is not None:
        data_dir = _resolve_open_source_output_by_run_id(project_root, open_source_run_id)
    else:
        if data_dir is None:
            mart_pointer_path = (
                project_root
                / "data"
                / "model_inputs"
                / "manifests"
                / "latest.json"
            )
            pointer_payload = _read_json_if_exists(mart_pointer_path)
            if pointer_payload is not None and pointer_payload.get("contract") is not None:
                if pointer_payload.get("contract") != SNAPSHOT_POINTER_CONTRACT:
                    raise RuntimeError("Unsupported production snapshot pointer contract")
                validate_snapshot_publication(mart_pointer_path)
            mart_input = resolve_mart_model_input(
                mart_pointer_path,
                warehouse_root=project_root / "data" / "warehouse",
            )
            data_dir = mart_input.mart_dir
    output_dir = output_dir if output_dir is not None else project_root / "outputs"
    methodology_identity: Dict[str, Any] | None = None
    if methodology_manifest is not None:
        methodology_manifest = methodology_manifest.resolve()
        causal_package = methodology_manifest.parent
        validation = validate_causal_v2_snapshot(causal_package)
        expected_data_dir = (causal_package / "input_snapshot").resolve()
        if data_dir is None or data_dir.resolve() != expected_data_dir:
            raise ValueError(
                "A causal methodology manifest requires --data-dir to reference "
                "its sealed input_snapshot."
            )
        methodology_identity = {
            **validation,
            "methodology_version": "v2-causal",
            "methodology_manifest": str(methodology_manifest),
            "methodology_manifest_sha256": _sha256_path(methodology_manifest),
        }
    run_started_at = datetime.now()
    run_instance_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    run_date_dir = output_dir / run_started_at.strftime("%Y-%m-%d")
    run_day_dir = run_date_dir / "runs" / run_instance_id
    run_day_dir = reserve_run_directory(run_day_dir)
    set_run_log_context(run_id=run_instance_id, component=__name__, step="legacy_pipeline")

    source_data_dir = data_dir.resolve()
    source_input_files = _input_files(
        source_data_dir,
        final_price_path=final_price_path,
        sp500_price_path=sp500_price_path,
    )
    input_snapshot_dir = _snapshot_input_package(
        source_data_dir=source_data_dir,
        input_files=source_input_files,
        run_day_dir=run_day_dir,
    )
    data_dir = input_snapshot_dir.resolve()
    os.chdir(data_dir)  # Keep legacy behaviour while making the working package immutable for this run.

    input_files = _input_files(data_dir)
    payload = _load_data(data_dir)
    latest_snapshot = load_latest_manifest(data_dir)
    set_run_log_context(
        run_id=run_instance_id,
        snapshot_id=(latest_snapshot or {}).get("snapshot_id") or run_instance_id,
        component=__name__,
        step="legacy_pipeline",
    )
    audited_registry = (
        load_ticker_exclusion_registry(ticker_exclusion_registry)
        if ticker_exclusion_registry is not None
        else None
    )
    terminal_event_registry = load_terminal_event_registry()
    legacy_exclusions = ("SII.US", "CBE.US", "TIE.US")
    ticker_to_exclude = normalize_tickers(
        (
            *legacy_exclusions,
            *(audited_registry.excluded_tickers if audited_registry else ()),
        )
    )
    price_eligibility_policy = monthly_price_eligibility_policy(
        policy_id=price_eligibility_policy_id,
        minimum_observations=minimum_monthly_price_observations,
        minimum_median_dollar_volume=minimum_monthly_median_dollar_volume,
        maximum_ohlc_violation_rate=maximum_monthly_ohlc_violation_rate,
    )
    run_config = {
        "backend": backend,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "first_date": first_date,
        "open_source_run_id": open_source_run_id,
        "run_instance_id": run_instance_id,
        "run_output_dir": str(run_day_dir.resolve()),
        "source_input_files": {name: str(path.resolve()) for name, path in source_input_files.items()},
        "source_input_sha256": {name: _sha256_path(path) for name, path in source_input_files.items()},
        "mart_input": mart_input.to_manifest() if mart_input is not None else None,
        "input_snapshot_storage": _read_json_if_exists(
            input_snapshot_dir / "storage_manifest.json"
        ),
        "excluded_tickers": list(ticker_to_exclude),
        "price_eligibility_policy_id": price_eligibility_policy.policy_id,
        "minimum_monthly_price_observations": (
            price_eligibility_policy.minimum_observations
        ),
        "minimum_monthly_median_dollar_volume": (
            price_eligibility_policy.minimum_median_dollar_volume
        ),
        "maximum_monthly_ohlc_violation_rate": (
            price_eligibility_policy.maximum_ohlc_violation_rate
        ),
        "fundamental_eligibility_policy_id": fundamental_eligibility_policy_id,
        "ticker_exclusion_registry": (
            str(audited_registry.path) if audited_registry is not None else None
        ),
        "ticker_exclusion_registry_id": (
            audited_registry.registry_id if audited_registry is not None else None
        ),
        "ticker_exclusion_registry_sha256": (
            _sha256_path(audited_registry.path)
            if audited_registry is not None
            else None
        ),
        "terminal_entry_registry": str(terminal_event_registry.path),
        "terminal_entry_registry_id": terminal_event_registry.payload["registry_id"],
        "terminal_entry_registry_sha256": terminal_event_registry.sha256,
        "decision_data_completed_through_month": str(
            completed_through_month(payload["sp500_price"])
        ),
        "partial_price_month_policy": (
            "exclude the latest observed calendar month from model inputs; "
            "retain it only as snapshot freshness evidence"
        ),
        "historical_legacy_missing_return_policy": (
            HISTORICAL_LEGACY_MISSING_RETURN_POLICY
        ),
        "execution_sensitivity_require_canonical_available": (
            methodology_identity is not None
        ),
        "canonical_execution_policy": ALPHARANK_REFERENCE_CLOSE.to_manifest(),
        "mandatory_execution_sensitivities": [
            LEGACY_NEXT_SESSION_OPEN.to_manifest()
        ],
        "methodology_identity": methodology_identity,
    }
    runtime_provenance = capture_runtime_provenance(
        project_root=project_root,
        entrypoint="scripts.run_legacy.run_pipeline",
        command_argv=[sys.executable, *sys.argv],
        resolved_config=run_config,
        seeds={
            "frequency_split_1": 42,
            "frequency_split_2": 41,
            "equal_split_1": 42,
            "equal_split_2": 41,
        },
        critical_files=(
            "scripts/run_legacy.py",
            "src/alpharank/production/legacy_pipeline.py",
            "src/alpharank/causal_snapshot.py",
            "src/alpharank/data/processing.py",
            "src/alpharank/data/warehouse/mart.py",
            "src/alpharank/strategy/legacy.py",
            "src/alpharank/strategy/legacy_valuation.py",
            "src/alpharank/portfolio/simulation.py",
            "src/alpharank/portfolio/execution.py",
            "src/alpharank/portfolio/costs.py",
            "src/alpharank/data/price_eligibility.py",
            "src/alpharank/data/terminal_eligibility.py",
            "src/alpharank/portfolio/terminal_event_registry.py",
            "configs/data_quality/terminal_shareholder_events_v1.json",
            "configs/data_quality/terminal_shareholder_events_v2.json",
            "src/alpharank/governance.py",
            "src/alpharank/governance_contracts/common.py",
            "src/alpharank/governance_contracts/contracts.py",
            "src/alpharank/governance_contracts/runtime_provenance.py",
        ),
        data_identifiers={
            "input_snapshot_id": (
                (latest_snapshot or {}).get("snapshot_id")
                or _snapshot_identifier(data_dir)
                or run_instance_id
            ),
            "input_snapshot_dir": str(input_snapshot_dir.resolve()),
            "source_input_sha256": run_config["source_input_sha256"],
            "ticker_exclusion_registry_id": run_config[
                "ticker_exclusion_registry_id"
            ],
        },
        patch_path=run_day_dir / "runtime_git_patch.json",
    )
    manifest_extra = _manifest_extra_context(
        data_dir=data_dir,
        latest_snapshot=latest_snapshot,
        source_data_dir=source_data_dir,
        input_snapshot_dir=input_snapshot_dir,
        run_config=run_config,
        code_context=_code_context(project_root),
        runtime_provenance=runtime_provenance,
    )
    run_manifest = write_manifest(
        manifest_path=run_day_dir / "data_input_manifest.json",
        files=input_files,
        frames={
            "final_price": payload["final_price"],
            "general": payload["general"],
            "income_statement": payload["income_statement"],
            "balance_sheet": payload["balance_sheet"],
            "cash_flow": payload["cash_flow"],
            "earnings": payload["earnings"],
            "sp500_constituents": payload["us_historical_company"],
            "sp500_price": payload["sp500_price"],
        },
        snapshot_id=(latest_snapshot or {}).get("snapshot_id") or manifest_extra.get("open_source_output_run_id"),
        extra=manifest_extra,
    )
    final_price = payload["final_price"]
    general = payload["general"]
    income_statement = payload["income_statement"]
    balance_sheet = payload["balance_sheet"]
    cash_flow = payload["cash_flow"]
    earnings = payload["earnings"]
    us_historical_company = payload["us_historical_company"]
    sp500_price = payload["sp500_price"]

    LOGGER.info("Preprocessing Legacy inputs", extra={"backend": backend})
    LOGGER.info(
        "Full-trajectory ticker exclusions: "
        + ", ".join(ticker_to_exclude)
    )
    final_price = exclude_tickers_from_frame(final_price, ticker_to_exclude)
    general = exclude_tickers_from_frame(general, ticker_to_exclude)
    income_statement = exclude_tickers_from_frame(income_statement, ticker_to_exclude)
    balance_sheet = exclude_tickers_from_frame(balance_sheet, ticker_to_exclude)
    cash_flow = exclude_tickers_from_frame(cash_flow, ticker_to_exclude)
    earnings = exclude_tickers_from_frame(earnings, ticker_to_exclude)
    us_historical_company = exclude_tickers_from_frame(
        us_historical_company,
        ticker_to_exclude,
        ticker_column="Ticker",
    )

    decision_data_cutoff = completed_through_month(sp500_price)
    final_price = final_price.filter(
        pl.col("date").cast(pl.Date, strict=False).dt.truncate("1mo")
        <= pl.lit(decision_data_cutoff)
    )
    sp500_price = sp500_price.filter(
        pl.col("date").cast(pl.Date, strict=False).dt.truncate("1mo")
        <= pl.lit(decision_data_cutoff)
    )
    LOGGER.info(
        "Decision data completed through: "
        f"{decision_data_cutoff}; latest observed partial month excluded"
    )

    monthly_price_eligibility = build_monthly_price_eligibility(
        final_price,
        policy=price_eligibility_policy,
    )
    monthly_price_eligibility_file = run_day_dir / "monthly_price_eligibility.parquet"
    monthly_price_eligibility.write_parquet(monthly_price_eligibility_file)
    LOGGER.info(
        "Monthly price eligibility: "
        f"policy={price_eligibility_policy.policy_id}, "
        f"observations>={price_eligibility_policy.minimum_observations}, "
        "median_dollar_volume>="
        f"{price_eligibility_policy.minimum_median_dollar_volume:.0f}, "
        "ohlc_violation_rate<="
        f"{price_eligibility_policy.maximum_ohlc_violation_rate:.2%}"
    )

    final_price = final_price.with_columns(
        pl.col("date").cast(pl.Date, strict=False).dt.truncate("1mo").alias("year_month")
    )
    us_historical_company = (
        us_historical_company
        .with_columns([
            pl.col("Ticker").cast(pl.Utf8).str.replace_all(r"\\.", "-").alias("ticker"),
            pl.col("Date").cast(pl.Date, strict=False).dt.truncate("1mo").alias("year_month"),
        ])
        .with_columns((pl.col("ticker") + pl.lit(".US")).alias("ticker"))
    )
    eligible_historical_company = (
        us_historical_company.join(
            monthly_price_eligibility.select(
                pl.col("decision_month").alias("year_month"),
                "ticker",
                "price_eligible",
            ),
            on=["ticker", "year_month"],
            how="left",
        )
        .filter(pl.col("price_eligible").fill_null(False))
        .drop("price_eligible")
    )

    index_data = IndexDataManager(
        daily_prices_df=sp500_price.with_columns(
            pl.col("adjusted_close").alias("close")
        ),
        components_df=eligible_historical_company.clone(),
        backend=backend,
    )

    monthly_return = to_polars(
        PricesDataPreprocessor.calculate_monthly_returns(
            df=final_price.clone(),
            column_close="adjusted_close",
            column_date="date",
            backend=backend,
        )
    )
    _write_checkpoint(monthly_return, checkpoints_dir, f"{backend}_monthly_return")

    LOGGER.info("Calculating stock prices relative to the index")
    sp500_price = sp500_price.with_columns(
        pl.col("adjusted_close").alias("sp500_adjusted_close")
    )
    final_price_vs_index = to_polars(
        PricesDataPreprocessor.prices_vs_index(
            index=sp500_price.clone(),
            prices=final_price.clone(),
            column_close_index="sp500_adjusted_close",
            column_close_prices="adjusted_close",
            backend=backend,
        )
    )
    final_price_vs_index = to_polars(
        PricesDataPreprocessor.compute_dr(
            df=final_price_vs_index,
            column_date="date",
            column_close="adjusted_close",
            backend=backend,
        )
    )
    _write_checkpoint(
        final_price_vs_index.select(["ticker", "date", "close_vs_index", "dr_vs_index", "dr"]),
        checkpoints_dir,
        f"{backend}_final_price_vs_index",
    )

    LOGGER.info(
        "Building Legacy selection universe",
        extra={"fundamental_eligibility_policy_id": fundamental_eligibility_policy_id},
    )
    stocks_selections = build_legacy_selection_universe(
        policy_id=fundamental_eligibility_policy_id,
        monthly_return=monthly_return,
        historical_membership=eligible_historical_company,
        balance=balance_sheet,
        cash_flow=cash_flow,
        earnings=earnings,
        income=income_statement,
    )
    membership_gate_summary: dict[str, object] | None = None
    if methodology_identity is not None:
        candidates_before_gate = stocks_selections.height
        stocks_selections = require_holding_month_membership(
            stocks_selections,
            us_historical_company.select("year_month", "ticker"),
        )
        membership_gate_summary = {
            "policy_id": HOLDING_MONTH_MEMBERSHIP_POLICY_ID,
            "effective_boundary": "start_of_holding_month_before_first_execution",
            "uses_holding_prices_or_returns": False,
            "candidate_rows_before": candidates_before_gate,
            "candidate_rows_after": stocks_selections.height,
            "candidate_rows_removed": candidates_before_gate - stocks_selections.height,
        }
    _write_checkpoint(stocks_selections, checkpoints_dir, f"{backend}_stocks_selections")

    LOGGER.info("Running Legacy strategy learning", extra={"optimizer": "optuna"})
    prices_for_learning = to_pandas(final_price_vs_index)
    stocks_filter_for_learning = normalize_year_month_to_period(to_pandas(stocks_selections), col="year_month")
    sector_for_learning = to_pandas(general.select(["ticker", "Sector"]))

    optuna_output_1 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=2,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=42,
        backend=backend,
    )
    optuna_output_12 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=2,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=41,
        backend=backend,
    )
    optuna_output_21 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=1,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=42,
        backend=backend,
    )
    optuna_output_22 = StrategyLearner.learning_process_optuna_full(
        prices=prices_for_learning.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_filter_for_learning.copy(),
        sector=sector_for_learning.copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=n_trials,
        alpha=1,
        temp=10 * 12,
        n_jobs=n_jobs,
        mode="mean",
        seed=41,
        backend=backend,
    )
    optuna_outputs = {
        "11": optuna_output_1,
        "12": optuna_output_12,
        "21": optuna_output_21,
        "22": optuna_output_22,
    }
    for key, out in optuna_outputs.items():
        _write_checkpoint(out["aggregated"], checkpoints_dir, f"{backend}_optuna_output_{key}_aggregated")
        _write_checkpoint(_get_detailed_output(out, label=f"optuna_output_{key}"), checkpoints_dir, f"{backend}_optuna_output_{key}_detailed")
    write_legacy_search_audit(
        output_path=run_day_dir / "legacy_search_protocol.json",
        experiments=optuna_outputs,
        n_trials=n_trials,
        first_date=first_date,
        n_jobs=n_jobs,
    )

    combined_equal = StrategyLearner.aggregate_portfolios(
        [optuna_output_1, optuna_output_12, optuna_output_21, optuna_output_22],
        mode="equal",
        index=index_data,
        backend=backend,
    )
    combined_frequency = StrategyLearner.aggregate_portfolios(
        [optuna_output_1, optuna_output_12, optuna_output_21, optuna_output_22],
        mode="frequency",
        index=index_data,
        backend=backend,
    )
    _write_checkpoint(combined_equal["aggregated"], checkpoints_dir, f"{backend}_combined_equal")
    _write_checkpoint(combined_frequency["aggregated"], checkpoints_dir, f"{backend}_combined_frequency")
    _write_checkpoint(_get_detailed_output(combined_equal, label="combined_equal"), checkpoints_dir, f"{backend}_combined_equal_detailed")
    _write_checkpoint(_get_detailed_output(combined_frequency, label="combined_frequency"), checkpoints_dir, f"{backend}_combined_frequency_detailed")
    _write_checkpoint(index_data.monthly_returns, checkpoints_dir, f"{backend}_sp500")

    common_benchmark = monthly_benchmark_returns(
        sp500_price,
        convention=SPY_TOTAL_RETURN,
    ).filter(
        pl.col("year_month") <= decision_data_cutoff
    )
    common_holdings_parts: list[pl.DataFrame] = []
    for strategy_name, portfolio_output in (
        ("Combined_Equal", combined_equal),
        ("Combined_Frequency", combined_frequency),
    ):
        common_holdings_parts.append(
            legacy_detailed_to_holdings(
                to_polars(
                    normalize_year_month_to_timestamp(
                        to_pandas(_get_detailed_output(portfolio_output, label=strategy_name)),
                        col="year_month",
                    )
                ),
                strategy=strategy_name,
                benchmark_monthly=common_benchmark,
            ).filter(pl.col("benchmark_return").is_not_null())
        )
    common_holdings = pl.concat(common_holdings_parts, how="diagonal_relaxed")
    legacy_v2_holdings_file: Path | None = None
    legacy_v2_monthly_file: Path | None = None
    legacy_v2_manifest_file: Path | None = None
    legacy_v2_holdings: pl.DataFrame | None = None
    if methodology_identity is not None:
        price_columns = [
            "ticker",
            "date",
            "open",
            "close",
            "adjusted_close",
        ]
        benchmark_tickers = sp500_price.get_column("ticker").drop_nulls().unique()
        if benchmark_tickers.len() != 1:
            raise RuntimeError("Causal Legacy replay requires exactly one SPY ticker.")
        benchmark_seed = (
            common_holdings.select("decision_month", "holding_month")
            .unique()
            .with_columns(
                pl.lit("SPY_Total_Return").alias("strategy"),
                pl.lit(str(benchmark_tickers.item())).alias("ticker"),
                pl.lit(1.0).alias("target_weight"),
                pl.lit(0.0).alias("realized_return"),
                pl.lit(0.0).alias("benchmark_return"),
            )
        )
        benchmark_next_open = apply_next_session_open_holding_returns(
            benchmark_seed,
            sp500_price.select(price_columns),
        ).select(
            "holding_month",
            pl.col("realized_return").alias("benchmark_return_next_open"),
        )
        legacy_v2_holdings = (
            apply_next_session_open_holding_returns(
                common_holdings,
                final_price.select(price_columns),
            )
            .drop("benchmark_return")
            .join(
                benchmark_next_open,
                on="holding_month",
                how="left",
                validate="m:1",
            )
            .rename({"benchmark_return_next_open": "benchmark_return"})
        )
        scenario_monthly: list[pl.DataFrame] = []
        for scenario in LEGACY_V2_COST_SCENARIOS:
            for strategy_holdings in legacy_v2_holdings.partition_by(
                "strategy", maintain_order=True
            ):
                scenario_monthly.append(
                    simulate_weighted_portfolio(
                        strategy_holdings,
                        transaction_cost_model=scenario,
                        missing_return_policy="raise",
                        causal_timing_policy="require_explicit",
                    )
                )
        legacy_v2_monthly = pl.concat(scenario_monthly, how="diagonal_relaxed")
        legacy_v2_holdings_file = run_day_dir / "legacy_v2_holdings.parquet"
        legacy_v2_monthly_file = run_day_dir / "legacy_v2_monthly.parquet"
        legacy_v2_holdings.write_parquet(legacy_v2_holdings_file)
        legacy_v2_monthly.write_parquet(legacy_v2_monthly_file)
        legacy_v2_monthly.write_csv(run_day_dir / "legacy_v2_monthly.csv")
        legacy_v2_manifest = {
            "contract_version": 1,
            "scope": "alpharank_legacy_v2_replay",
            "methodology_identity": methodology_identity,
            "execution_policy": {
                "identifier": "next_session_open_v1",
                "return_window": "adjusted_open_to_last_adjusted_close_in_holding_month",
            },
            "missing_return_policy": "raise",
            "benchmark": {
                "ticker": str(benchmark_tickers.item()),
                "return_window": "adjusted_open_to_last_adjusted_close_in_holding_month",
            },
            "cost_scenarios": [asdict(model) for model in LEGACY_V2_COST_SCENARIOS],
            "canonical_cost_scenario_id": "standard_10bps",
            "candidate_membership_policy": membership_gate_summary,
            "holdings_rows": legacy_v2_holdings.height,
            "monthly_rows": legacy_v2_monthly.height,
            "strategies": sorted(legacy_v2_holdings["strategy"].unique().to_list()),
            "artifacts": {
                "holdings": {
                    "path": str(legacy_v2_holdings_file.resolve()),
                    "sha256": _sha256_path(legacy_v2_holdings_file),
                },
                "monthly": {
                    "path": str(legacy_v2_monthly_file.resolve()),
                    "sha256": _sha256_path(legacy_v2_monthly_file),
                },
            },
        }
        legacy_v2_manifest_file = run_day_dir / "legacy_v2_replay_manifest.json"
        legacy_v2_manifest_file.write_text(
            json.dumps(legacy_v2_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    common_monthly = _simulate_historical_legacy_common(common_holdings)
    common_monthly = pl.concat(
        [
            common_monthly,
            reference_monthly_series(
                common_monthly.filter(pl.col("strategy") == "Combined_Frequency"),
                strategy=SPY_TOTAL_RETURN.label,
                return_column="benchmark_return",
            ),
        ],
        how="diagonal_relaxed",
    )
    common_artifacts = write_common_portfolio_artifacts(
        output_dir=run_day_dir,
        holdings=common_holdings,
        monthly_returns=common_monthly,
        prefix="legacy_common",
        benchmark_metadata={
            "id": SPY_TOTAL_RETURN.identifier,
            "label": SPY_TOTAL_RETURN.label,
            "price_column": SPY_TOTAL_RETURN.price_column,
            "includes_distributions": SPY_TOTAL_RETURN.includes_distributions,
            "completed_through_month": str(decision_data_cutoff),
        },
    )

    models = {
        "Legacy_Optuna_11": to_pandas(_sort_monthly_frame(optuna_output_1["aggregated"])),
        "Legacy_Optuna_12": to_pandas(_sort_monthly_frame(optuna_output_12["aggregated"])),
        "Legacy_Optuna_21": to_pandas(_sort_monthly_frame(optuna_output_21["aggregated"])),
        "Legacy_Optuna_22": to_pandas(_sort_monthly_frame(optuna_output_22["aggregated"])),
        "Combined_Equal": to_pandas(_sort_monthly_frame(combined_equal["aggregated"])),
        "Combined_Frequency": to_pandas(_sort_monthly_frame(combined_frequency["aggregated"])),
        "SP500_Price_Return_Legacy_Signal": to_pandas(
            _sort_monthly_frame(index_data.monthly_returns)
        ),
        "SPY_Total_Return": to_pandas(
            _sort_monthly_frame(common_benchmark.select("year_month", "monthly_return"))
        ),
    }
    metrics, cumulative, correlation, worst_periods, drawdowns, annual_returns, cumulative_metrics, annual_metrics, monthly_returns = ModelEvaluator.compare_models(
        models,
        start_year=_extract_start_year(first_date),
    )
    _write_checkpoint(metrics.reset_index().rename(columns={"index": "model"}), checkpoints_dir, f"{backend}_metrics")

    detailed_outputs = {
        "Legacy_Optuna_11": _get_detailed_output(optuna_output_1, label="Legacy_Optuna_11"),
        "Legacy_Optuna_12": _get_detailed_output(optuna_output_12, label="Legacy_Optuna_12"),
        "Legacy_Optuna_21": _get_detailed_output(optuna_output_21, label="Legacy_Optuna_21"),
        "Legacy_Optuna_22": _get_detailed_output(optuna_output_22, label="Legacy_Optuna_22"),
        "Combined_Equal": _get_detailed_output(combined_equal, label="Combined_Equal"),
        "Combined_Frequency": _get_detailed_output(combined_frequency, label="Combined_Frequency"),
    }
    aggregated_outputs = {
        "Legacy_Optuna_11": optuna_output_1["aggregated"],
        "Legacy_Optuna_12": optuna_output_12["aggregated"],
        "Legacy_Optuna_21": optuna_output_21["aggregated"],
        "Legacy_Optuna_22": optuna_output_22["aggregated"],
        "Combined_Equal": combined_equal["aggregated"],
        "Combined_Frequency": combined_frequency["aggregated"],
        "SP500": index_data.monthly_returns,
    }
    comparison_curves_long = _named_frames_to_long(models)
    detailed_returns_long = _named_frames_to_long(detailed_outputs)
    aggregated_returns_long = _named_frames_to_long(aggregated_outputs)
    execution_holdings = (
        legacy_v2_holdings.select(
            pl.col("strategy").alias("portfolio_model"),
            pl.col("holding_month").alias("year_month"),
            "ticker",
        )
        if legacy_v2_holdings is not None
        else detailed_returns_long.filter(
            pl.col("portfolio_model").is_in(
                ["Combined_Equal", "Combined_Frequency"]
            )
        )
    )
    execution_price_columns = ["ticker", "date", "open", "close"]
    if "vwap" in final_price.columns:
        execution_price_columns.append("vwap")
    execution_orders = build_monthly_execution_orders(
        execution_holdings,
        final_price.select(execution_price_columns),
    )
    require_canonical_execution = methodology_identity is not None
    execution_sensitivity = build_execution_sensitivity_report(
        execution_orders,
        final_price.select(execution_price_columns),
        policy=ALPHARANK_REFERENCE_CLOSE,
        require_canonical_available=require_canonical_execution,
    )
    write_execution_sensitivity_report(
        execution_sensitivity,
        run_day_dir,
        policy=ALPHARANK_REFERENCE_CLOSE,
        require_canonical_available=require_canonical_execution,
    )
    comparison_monthly_returns_long = _named_frames_to_long(
        {
            model_name: _indexed_frame_to_polars(returns_series.rename("monthly_return"))
            for model_name, returns_series in monthly_returns.items()
        }
    )
    _write_checkpoint(comparison_curves_long, checkpoints_dir, f"{backend}_comparison_curves")
    _write_checkpoint(aggregated_returns_long, checkpoints_dir, f"{backend}_aggregated_returns")
    _write_checkpoint(detailed_returns_long, checkpoints_dir, f"{backend}_detailed_returns")
    _write_checkpoint(comparison_monthly_returns_long, checkpoints_dir, f"{backend}_comparison_monthly_returns")

    LOGGER.info("Generating Legacy reports")
    comparison_html = PortfolioVisualizer.make_comparison_report(
        metrics_df=metrics,
        cumulative_returns=cumulative,
        drawdowns_df=drawdowns,
        annual_returns_df=annual_returns,
        correlation_matrix=correlation,
        worst_periods_df=worst_periods,
        cumulative_metrics_dict=cumulative_metrics,
        annual_metrics_dict=annual_metrics,
        monthly_returns_dict=monthly_returns,
        title=f"Strategy Performance Comparison ({backend})",
    )
    is_test_run = n_trials < 30
    file_suffix = "_test" if is_test_run else datetime.now().strftime("%Y-%m-%d")
    comparison_file = run_day_dir / f"performance_of_models_{backend}{file_suffix}.html"
    _save_html(comparison_html, comparison_file)

    comparison_curves_file = _write_artifact_frame(
        comparison_curves_long,
        run_day_dir / f"legacy_comparison_curves_{backend}.parquet",
    )
    aggregated_returns_file = _write_artifact_frame(
        aggregated_returns_long,
        run_day_dir / f"legacy_aggregated_returns_{backend}.parquet",
    )
    detailed_returns_file = _write_artifact_frame(
        detailed_returns_long,
        run_day_dir / f"legacy_detailed_returns_{backend}.parquet",
    )
    monthly_returns_file = _write_artifact_frame(
        comparison_monthly_returns_long,
        run_day_dir / f"legacy_monthly_returns_{backend}.parquet",
    )
    cumulative_returns_file = _write_artifact_frame(
        _indexed_frame_to_polars(cumulative),
        run_day_dir / f"legacy_cumulative_returns_{backend}.parquet",
    )
    drawdowns_file = _write_artifact_frame(
        _indexed_frame_to_polars(drawdowns),
        run_day_dir / f"legacy_drawdowns_{backend}.parquet",
    )
    annual_returns_file = _write_artifact_frame(
        _indexed_frame_to_polars(annual_returns),
        run_day_dir / f"legacy_annual_returns_{backend}.parquet",
    )
    metrics_file = _write_artifact_frame(
        metrics.reset_index().rename(columns={"index": "model"}),
        run_day_dir / f"legacy_metrics_{backend}.parquet",
    )

    if "close" not in final_price.columns and "adjusted_close" in final_price.columns:
        final_price_long = final_price.rename({"adjusted_close": "close"})
    else:
        final_price_long = final_price

    current_portfolio_freq = StrategyLearner.get_portfolio_at_month(combined_frequency)
    current_freq_context = _build_report_context(
        month=current_portfolio_freq.attrs.get("month", "Latest"),
        input_files=input_files,
        run_manifest=run_manifest,
        final_price=final_price_long,
        sp500_price=sp500_price,
        us_historical_company=us_historical_company,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        earnings=earnings,
    )
    report_html_freq = PortfolioVisualizer.make_portfolio_report(
        portfolio=current_portfolio_freq,
        title=f"Aggregated Portfolio (Frequency Weighted) - {backend}",
        price_data=to_pandas(final_price_long),
        balance_sheet=to_pandas(balance_sheet),
        income_statement=to_pandas(income_statement),
        cash_flow=to_pandas(cash_flow),
        earnings=to_pandas(earnings),
        backend="polars",
        report_context=current_freq_context,
    )
    freq_file = run_day_dir / f"portfolio_report_frequency_{backend}{file_suffix}.html"
    _save_html(report_html_freq, freq_file)

    current_portfolio_equal = StrategyLearner.get_portfolio_at_month(combined_equal)
    current_equal_context = _build_report_context(
        month=current_portfolio_equal.attrs.get("month", "Latest"),
        input_files=input_files,
        run_manifest=run_manifest,
        final_price=final_price_long,
        sp500_price=sp500_price,
        us_historical_company=us_historical_company,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        earnings=earnings,
    )
    report_html_equal = PortfolioVisualizer.make_portfolio_report(
        portfolio=current_portfolio_equal,
        title=f"Aggregated Portfolio (Equal Weighted) - {backend}",
        price_data=to_pandas(final_price_long),
        balance_sheet=to_pandas(balance_sheet),
        income_statement=to_pandas(income_statement),
        cash_flow=to_pandas(cash_flow),
        earnings=to_pandas(earnings),
        backend="polars",
        report_context=current_equal_context,
    )
    equal_file = run_day_dir / f"portfolio_report_equal_{backend}{file_suffix}.html"
    _save_html(report_html_equal, equal_file)

    # Generate end-of-month portfolio snapshots for the last 3 months available in backtest outputs.
    available_months = sorted(to_pandas(combined_frequency["aggregated"])["year_month"].dropna().unique())
    last_three_months = available_months[-3:]
    monthly_snapshot_files: Dict[str, Path] = {}
    for month in last_three_months:
        month_label = str(month).replace("/", "-")
        month_context = _build_report_context(
            month=month,
            input_files=input_files,
            run_manifest=run_manifest,
            final_price=final_price_long,
            sp500_price=sp500_price,
            us_historical_company=us_historical_company,
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            earnings=earnings,
        )

        freq_month_portfolio = StrategyLearner.get_portfolio_at_month(combined_frequency, month=month)
        freq_month_html = PortfolioVisualizer.make_portfolio_report(
            portfolio=freq_month_portfolio,
            title=f"Aggregated Portfolio (Frequency Weighted) - {backend} - {month_label}",
            price_data=to_pandas(final_price_long),
            balance_sheet=to_pandas(balance_sheet),
            income_statement=to_pandas(income_statement),
            cash_flow=to_pandas(cash_flow),
            earnings=to_pandas(earnings),
            backend="polars",
            report_context=month_context,
        )
        freq_month_file = run_day_dir / f"portfolio_report_frequency_{backend}_{month_label}.html"
        _save_html(freq_month_html, freq_month_file)
        monthly_snapshot_files[f"portfolio_frequency_{month_label}"] = freq_month_file

        equal_month_portfolio = StrategyLearner.get_portfolio_at_month(combined_equal, month=month)
        equal_month_html = PortfolioVisualizer.make_portfolio_report(
            portfolio=equal_month_portfolio,
            title=f"Aggregated Portfolio (Equal Weighted) - {backend} - {month_label}",
            price_data=to_pandas(final_price_long),
            balance_sheet=to_pandas(balance_sheet),
            income_statement=to_pandas(income_statement),
            cash_flow=to_pandas(cash_flow),
            earnings=to_pandas(earnings),
            backend="polars",
            report_context=month_context,
        )
        equal_month_file = run_day_dir / f"portfolio_report_equal_{backend}_{month_label}.html"
        _save_html(equal_month_html, equal_month_file)
        monthly_snapshot_files[f"portfolio_equal_{month_label}"] = equal_month_file

    latest_pointer = {
        "run_instance_id": run_instance_id,
        "run_output_dir": str(run_day_dir.resolve()),
        "data_input_manifest": str((run_day_dir / "data_input_manifest.json").resolve()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    latest_pointer_file = run_date_dir / "latest_legacy_run.json"
    latest_pointer_file.write_text(json.dumps(latest_pointer, indent=2), encoding="utf-8")

    return PipelineOutput(
        monthly_return=monthly_return,
        final_price_vs_index=final_price_vs_index,
        stocks_selections=stocks_selections,
        optuna_outputs=optuna_outputs,
        combined_equal=combined_equal,
        combined_frequency=combined_frequency,
        metrics=metrics,
        artifacts={
            "comparison_html": comparison_file,
            "comparison_curves": comparison_curves_file,
            "aggregated_returns": aggregated_returns_file,
            "detailed_returns": detailed_returns_file,
            "monthly_returns": monthly_returns_file,
            "cumulative_returns": cumulative_returns_file,
            "drawdowns": drawdowns_file,
            "annual_returns": annual_returns_file,
            "metrics": metrics_file,
            **{f"common_{key}": value for key, value in common_artifacts.items()},
            "portfolio_frequency_html": freq_file,
            "portfolio_equal_html": equal_file,
            "data_input_manifest": run_day_dir / "data_input_manifest.json",
            "input_snapshot_dir": input_snapshot_dir,
            "monthly_price_eligibility": monthly_price_eligibility_file,
            **(
                {
                    "legacy_v2_holdings": legacy_v2_holdings_file,
                    "legacy_v2_monthly": legacy_v2_monthly_file,
                    "legacy_v2_manifest": legacy_v2_manifest_file,
                }
                if legacy_v2_manifest_file is not None
                else {}
            ),
            "latest_run_pointer": latest_pointer_file,
            **monthly_snapshot_files,
        },
    )
