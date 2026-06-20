#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


DEFAULT_KPI_YEAR_PAIRS: tuple[tuple[str, int], ...] = (
    ("revenue", 2023),
    ("net_income", 2023),
    ("epsActual", 2022),
)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the reproducible SEC candidate workflow: targeted companyfacts refresh, probe rebuild, fix2 overlay, quality report, yearly KPI report."
    )
    parser.add_argument("--raw-dir", type=Path, default=project_root / "data" / "open_source" / "official" / "raw")
    parser.add_argument("--cache-dir", type=Path, default=project_root / "data" / "open_source" / "_cache" / "sec_companyfacts")
    parser.add_argument("--quality-dir", type=Path, default=project_root / "outputs" / "sec_quality_dashboard_latest")
    parser.add_argument("--fix2-sec-dir", type=Path, default=project_root / "outputs" / "sec_overlay_fix2_output")
    parser.add_argument("--output-prefix", type=Path, default=project_root / "outputs" / "sec_q4_fix2_candidate")
    parser.add_argument("--max-tickers-per-pair", type=int, default=20)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def _load_target_tickers(*, quality_dir: Path, max_tickers_per_pair: int) -> list[str]:
    holes = pl.read_parquet(quality_dir / "quarterly_holes.parquet")
    tickers: list[str] = []
    for metric, fiscal_year in DEFAULT_KPI_YEAR_PAIRS:
        top = (
            holes.filter((pl.col("metric") == metric) & (pl.col("fiscal_year") == fiscal_year))
            .group_by("ticker")
            .agg(pl.len().alias("missing_quarters"))
            .sort(["missing_quarters", "ticker"], descending=[True, False])
            .head(max_tickers_per_pair)
            .get_column("ticker")
            .to_list()
        )
        tickers.extend(top)
    return sorted(set(tickers))


def _copy_probe_raw(*, raw_dir: Path, probe_raw_dir: Path) -> None:
    if probe_raw_dir.exists():
        shutil.rmtree(probe_raw_dir)
    probe_raw_dir.mkdir(parents=True, exist_ok=True)
    required = (
        "financials_sec_companyfacts.parquet",
        "financials_sec_filing.parquet",
        "earnings_sec_calendar.parquet",
        "earnings_sec_actuals.parquet",
        "general_reference_lineage.parquet",
    )
    for name in required:
        shutil.copy2(raw_dir / name, probe_raw_dir / name)


def _run_python(*, project_root: Path, args: list[str]) -> None:
    cmd = [str(project_root / ".venv" / "bin" / "python"), *args]
    subprocess.run(cmd, cwd=project_root, check=True)


def _replace_path(*, source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output_prefix.resolve()

    probe_raw_dir = output_prefix.parent / f"{output_prefix.name}_raw_{timestamp}"
    probe_cache_dir = output_prefix.parent / f"{output_prefix.name}_cache_{timestamp}"
    probe_output_dir = output_prefix.parent / f"{output_prefix.name}_probe_output_{timestamp}"
    probe_quality_dir = output_prefix.parent / f"{output_prefix.name}_probe_quality_{timestamp}"
    probe_yearly_dir = output_prefix.parent / f"{output_prefix.name}_probe_yearly_{timestamp}"
    combo_output_dir = output_prefix.parent / f"{output_prefix.name}_combo_output_{timestamp}"
    combo_quality_dir = output_prefix.parent / f"{output_prefix.name}_combo_quality_{timestamp}"
    combo_yearly_dir = output_prefix.parent / f"{output_prefix.name}_combo_yearly_{timestamp}"
    hybrid_output_dir = output_prefix.parent / f"sec_kpi_hybrid_output_{timestamp}"
    hybrid_quality_dir = output_prefix.parent / f"sec_kpi_hybrid_quality_{timestamp}"
    hybrid_yearly_dir = output_prefix.parent / f"sec_kpi_hybrid_yearly_{timestamp}"
    manifest_path = output_prefix.parent / f"{output_prefix.name}_manifest_{timestamp}.json"

    target_tickers = _load_target_tickers(
        quality_dir=args.quality_dir.resolve(),
        max_tickers_per_pair=args.max_tickers_per_pair,
    )
    _copy_probe_raw(raw_dir=args.raw_dir.resolve(), probe_raw_dir=probe_raw_dir)

    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/refresh_sec_companyfacts_from_cache.py",
            "--raw-dir",
            str(probe_raw_dir),
            "--cache-dir",
            str(probe_cache_dir),
            "--ingestion-run-id",
            f"q4_candidate_{timestamp}",
            "--tickers",
            *target_tickers,
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_output_package.py",
            "--raw-source-dir",
            str(probe_raw_dir),
            "--output-dir",
            str(probe_output_dir),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_quality_dashboard.py",
            "--sec-output-dir",
            str(probe_output_dir),
            "--output-dir",
            str(probe_quality_dir),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_core_kpi_yearly_report.py",
            "--sec-output-dir",
            str(probe_output_dir),
            "--quality-dir",
            str(probe_quality_dir),
            "--output-dir",
            str(probe_yearly_dir),
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_overlay_package.py",
            "--primary-sec-dir",
            str(probe_output_dir),
            "--secondary-sec-dir",
            str(args.fix2_sec_dir.resolve()),
            "--output-dir",
            str(combo_output_dir),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_quality_dashboard.py",
            "--sec-output-dir",
            str(combo_output_dir),
            "--output-dir",
            str(combo_quality_dir),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_core_kpi_yearly_report.py",
            "--sec-output-dir",
            str(combo_output_dir),
            "--quality-dir",
            str(combo_quality_dir),
            "--output-dir",
            str(combo_yearly_dir),
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_metric_hybrid_package.py",
            "--base-sec-dir",
            str(combo_output_dir),
            "--eps-fallback-sec-dir",
            str(args.fix2_sec_dir.resolve()),
            "--output-dir",
            str(hybrid_output_dir),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_quality_dashboard.py",
            "--sec-output-dir",
            str(hybrid_output_dir),
            "--output-dir",
            str(hybrid_quality_dir),
        ],
    )
    _run_python(
        project_root=project_root,
        args=[
            "scripts/open_source/build_sec_core_kpi_yearly_report.py",
            "--sec-output-dir",
            str(hybrid_output_dir),
            "--quality-dir",
            str(hybrid_quality_dir),
            "--output-dir",
            str(hybrid_yearly_dir),
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
        ],
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quality_dir": str(args.quality_dir.resolve()),
        "fix2_sec_dir": str(args.fix2_sec_dir.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "probe_raw_dir": str(probe_raw_dir),
        "probe_output_dir": str(probe_output_dir),
        "probe_quality_dir": str(probe_quality_dir),
        "probe_yearly_dir": str(probe_yearly_dir),
        "combo_output_dir": str(combo_output_dir),
        "combo_quality_dir": str(combo_quality_dir),
        "combo_yearly_dir": str(combo_yearly_dir),
        "hybrid_output_dir": str(hybrid_output_dir),
        "hybrid_quality_dir": str(hybrid_quality_dir),
        "hybrid_yearly_dir": str(hybrid_yearly_dir),
        "kpi_year_pairs": [{"metric": metric, "fiscal_year": fiscal_year} for metric, fiscal_year in DEFAULT_KPI_YEAR_PAIRS],
        "target_tickers": target_tickers,
        "start_year": args.start_year,
        "end_year": args.end_year,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    latest_map = {
        probe_raw_dir: output_prefix.parent / f"{output_prefix.name}_raw_latest",
        probe_output_dir: output_prefix.parent / f"{output_prefix.name}_probe_output_latest",
        probe_quality_dir: output_prefix.parent / f"{output_prefix.name}_probe_quality_latest",
        probe_yearly_dir: output_prefix.parent / f"{output_prefix.name}_probe_yearly_latest",
        combo_output_dir: output_prefix.parent / f"{output_prefix.name}_combo_output_latest",
        combo_quality_dir: output_prefix.parent / f"{output_prefix.name}_combo_quality_latest",
        combo_yearly_dir: output_prefix.parent / f"{output_prefix.name}_combo_yearly_latest",
        hybrid_output_dir: output_prefix.parent / "sec_kpi_hybrid_output_latest",
        hybrid_quality_dir: output_prefix.parent / "sec_kpi_hybrid_quality_latest",
        hybrid_yearly_dir: output_prefix.parent / "sec_kpi_hybrid_yearly_latest",
        manifest_path: output_prefix.parent / f"{output_prefix.name}_manifest_latest.json",
    }
    for source, destination in latest_map.items():
        _replace_path(source=source, destination=destination)

    print(manifest_path)
    print(combo_yearly_dir / "worst_year_brief.md")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - operational wrapper
        print(f"Command failed with exit code {exc.returncode}: {' '.join(map(str, exc.cmd))}", file=sys.stderr)
        raise
