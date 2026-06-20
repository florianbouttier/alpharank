# %%
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from calendar import monthrange
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import polars as pl

from alpharank.data.lineage import load_latest_manifest, write_manifest
from alpharank.data.processing import FundamentalProcessor, IndexDataManager, PricesDataPreprocessor
from alpharank.features.indicators import TechnicalIndicators
from alpharank.strategy.legacy import ModelEvaluator, StrategyLearner
from alpharank.utils.frame_backend import normalize_year_month_to_period, to_pandas, to_polars
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
    print("Loading data...")
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
        shutil.copy2(source, destination)


def _snapshot_input_package(*, source_data_dir: Path, input_files: Dict[str, Path], run_day_dir: Path) -> Path:
    snapshot_dir = run_day_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, source_path in input_files.items():
        target_name = INPUT_PACKAGE_FILENAMES[name]
        shutil.copy2(source_path, snapshot_dir / target_name)

    lineage_dir = source_data_dir / "lineage"
    if lineage_dir.exists():
        shutil.copytree(lineage_dir, snapshot_dir / "lineage", dirs_exist_ok=True)
    for metadata_name in ("snapshot_manifest.json", "latest_snapshot.json", "README.md"):
        _copy_if_exists(source_data_dir / metadata_name, snapshot_dir / metadata_name)
    return snapshot_dir


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
                "open_source_ingested_at": ingestion_manifest.get("ingested_at"),
                "open_source_mode": ingestion_manifest.get("mode"),
                "open_source_price_window": ingestion_manifest.get("price_window"),
                "open_source_financial_years_refreshed": ingestion_manifest.get("financial_years_refreshed"),
                "open_source_ticker_count": ingestion_manifest.get("ticker_count"),
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
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _code_context(project_root: Path) -> Dict[str, Any]:
    critical_files = [
        "scripts/run_legacy.py",
        "src/alpharank/data/processing.py",
        "src/alpharank/strategy/legacy.py",
        "src/alpharank/data/open_source/legacy_export.py",
        "src/alpharank/data/open_source/pipeline.py",
        "src/alpharank/data/open_source/publishing.py",
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
    except Exception as exc:
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
    except Exception:
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
) -> PipelineOutput:
    backend = "polars"
    project_root = Path(__file__).parent.parent
    if data_dir is not None and open_source_run_id is not None:
        raise ValueError("Use either data_dir or open_source_run_id, not both.")
    if open_source_run_id is not None:
        data_dir = _resolve_open_source_output_by_run_id(project_root, open_source_run_id)
    else:
        data_dir = data_dir if data_dir is not None else project_root / "data"
    output_dir = output_dir if output_dir is not None else project_root / "outputs"
    run_started_at = datetime.now()
    run_instance_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    run_date_dir = output_dir / run_started_at.strftime("%Y-%m-%d")
    run_day_dir = run_date_dir / "runs" / run_instance_id
    run_day_dir.mkdir(parents=True, exist_ok=True)

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
    }
    manifest_extra = _manifest_extra_context(
        data_dir=data_dir,
        latest_snapshot=latest_snapshot,
        source_data_dir=source_data_dir,
        input_snapshot_dir=input_snapshot_dir,
        run_config=run_config,
        code_context=_code_context(project_root),
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

    print(f"Preprocessing ({backend})...")
    ticker_to_exclude = ["SII.US", "CBE.US", "TIE.US"]
    final_price = final_price.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    general = general.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    income_statement = income_statement.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    balance_sheet = balance_sheet.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    cash_flow = cash_flow.filter(~pl.col("ticker").is_in(ticker_to_exclude))
    earnings = earnings.filter(~pl.col("ticker").is_in(ticker_to_exclude))

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

    index_data = IndexDataManager(
        daily_prices_df=sp500_price.clone(),
        components_df=us_historical_company.clone(),
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

    print("Calculating prices vs index...")
    sp500_price = sp500_price.rename({"close": "sp500_close"})
    final_price_vs_index = to_polars(
        PricesDataPreprocessor.prices_vs_index(
            index=sp500_price.clone(),
            prices=final_price.clone(),
            column_close_index="sp500_close",
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

    print("Calculating ratios...")
    stocks_selections = to_polars(
        FundamentalProcessor.calculate_pe_ratios(
            balance=balance_sheet,
            earnings=earnings,
            cashflow=cash_flow,
            income=income_statement,
            earning_choice="netincome_rolling",
            monthly_return=to_pandas(monthly_return),
            list_date_to_maximise=["filing_date_income", "filing_date_balance"],
            backend=backend,
        )
    )
    _ = FundamentalProcessor.calculate_all_ratios(
        balance_sheet=balance_sheet,
        income_statement=income_statement,
        cash_flow=cash_flow,
        earnings=earnings,
        monthly_return=to_pandas(monthly_return),
        backend=backend,
    )

    stocks_selections = (
        stocks_selections
        .with_columns(pl.col("year_month").cast(pl.Date, strict=False))
        .filter(
            (pl.col("pe") < 100)
            & (pl.col("pe") > 0)
            & pl.col("pe").is_not_null()
            & pl.col("market_cap").is_not_null()
        )
        .join(
            us_historical_company.select(
                pl.col("year_month").cast(pl.Date, strict=False).alias("year_month"),
                pl.col("ticker"),
            ),
            how="inner",
            on=["ticker", "year_month"],
        )
    )
    _write_checkpoint(stocks_selections, checkpoints_dir, f"{backend}_stocks_selections")

    print("Running strategy learning (Optuna)...")
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

    models = {
        "Legacy_Optuna_11": to_pandas(_sort_monthly_frame(optuna_output_1["aggregated"])),
        "Legacy_Optuna_12": to_pandas(_sort_monthly_frame(optuna_output_12["aggregated"])),
        "Legacy_Optuna_21": to_pandas(_sort_monthly_frame(optuna_output_21["aggregated"])),
        "Legacy_Optuna_22": to_pandas(_sort_monthly_frame(optuna_output_22["aggregated"])),
        "Combined_Equal": to_pandas(_sort_monthly_frame(combined_equal["aggregated"])),
        "Combined_Frequency": to_pandas(_sort_monthly_frame(combined_frequency["aggregated"])),
        "SP500": to_pandas(_sort_monthly_frame(index_data.monthly_returns)),
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

    print("Generating reports...")
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
            "portfolio_frequency_html": freq_file,
            "portfolio_equal_html": equal_file,
            "data_input_manifest": run_day_dir / "data_input_manifest.json",
            "input_snapshot_dir": input_snapshot_dir,
            "latest_run_pointer": latest_pointer_file,
            **monthly_snapshot_files,
        },
    )


def main(
    *,
    n_trials: int = 30,
    n_jobs: int = 1,
    first_date: str = "2010-01",
    data_dir: str | Path | None = None,
    open_source_run_id: str | None = None,
    output_dir: str | Path | None = None,
    checkpoints_dir: str | Path = "outputs/checkpoints",
    final_price_path: str | Path | None = None,
    sp500_price_path: str | Path | None = None,
) -> None:
    checkpoints_dir = Path(checkpoints_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    final_price_path = Path(final_price_path).expanduser().resolve() if final_price_path else None
    sp500_price_path = Path(sp500_price_path).expanduser().resolve() if sp500_price_path else None

    out = run_pipeline(
        n_trials=n_trials,
        n_jobs=n_jobs,
        first_date=first_date,
        data_dir=data_dir,
        open_source_run_id=open_source_run_id,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        final_price_path=final_price_path,
        sp500_price_path=sp500_price_path,
    )
    print("Artifacts:")
    for k, v in out.artifacts.items():
        print(f"  {k}: {v}")


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "default"


def _default_log_stem(*, data_dir: str | None, open_source_run_id: str | None) -> str:
    if open_source_run_id:
        return f"run_legacy_open_source_{_slug(open_source_run_id)}"
    if data_dir and "open_source" in data_dir:
        return "run_legacy_open_source"
    return "run_legacy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the legacy monthly portfolio pipeline.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--first-date", default="2010-01")
    parser.add_argument("--data-dir")
    parser.add_argument("--open-source-run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--final-price-path")
    parser.add_argument("--sp500-price-path")
    parser.add_argument("--log-dir", default="logs/legacy_runs")
    parser.add_argument("--no-log", action="store_true", help="Disable automatic CLI log capture.")
    return parser.parse_args()


def _run_cli() -> None:
    args = _parse_args()
    kwargs = {
        "n_trials": args.n_trials,
        "n_jobs": args.n_jobs,
        "first_date": args.first_date,
        "data_dir": args.data_dir,
        "open_source_run_id": args.open_source_run_id,
        "output_dir": args.output_dir,
        "checkpoints_dir": args.checkpoints_dir,
        "final_price_path": args.final_price_path,
        "sp500_price_path": args.sp500_price_path,
    }
    if args.no_log:
        main(**kwargs)
        return

    project_root = Path(__file__).parent.parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _default_log_stem(data_dir=args.data_dir, open_source_run_id=args.open_source_run_id)
    log_path = log_dir / f"{stem}_{timestamp}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = _Tee(sys.stdout, log_file)
        stderr = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Log file: {log_path}")
            print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Arguments: {kwargs}")
            main(**kwargs)
            print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    _run_cli()
