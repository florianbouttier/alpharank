#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def run_once(cmd: List[str], log_file: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee {log_file}")
    return elapsed


def bench_polars(
    *,
    alpharank_repo: Path,
    data_dir: Path,
    output_dir: Path,
    checkpoints_dir: Path,
    n_trials: int,
    first_date: str,
    warmups: int,
    runs: int,
    logs_dir: Path,
) -> Dict[str, float]:
    cmd = [
        sys.executable,
        "-c",
        (
            "from scripts.run_legacy import main; "
            f"main(n_trials={n_trials}, n_jobs=1, first_date={first_date!r}, "
            f"data_dir={str(data_dir)!r}, output_dir={str(output_dir)!r}, "
            f"checkpoints_dir={str(checkpoints_dir)!r})"
        ),
    ]

    for i in range(warmups):
        run_once(cmd, logs_dir / f"polars_warmup_{i+1}.log")

    samples: List[float] = []
    for i in range(runs):
        elapsed = run_once(cmd, logs_dir / f"polars_run_{i+1}.log")
        samples.append(elapsed)

    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
        "runs": runs,
    }


def main(
    *,
    alpharank_repo: str | Path | None = None,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    n_trials: int = 2,
    first_date: str = "2020-01",
    warmups: int = 1,
    runs: int = 3,
) -> None:
    benchmark_repo_root = Path(__file__).resolve().parents[1]
    alpharank_repo = (
        Path(alpharank_repo).expanduser().resolve()
        if alpharank_repo
        else benchmark_repo_root.parents[1]
    )
    data_dir = (
        Path(data_dir).expanduser().resolve()
        if data_dir
        else benchmark_repo_root / "data" / "subset_legacy_v1"
    )
    results_dir = (
        Path(results_dir).expanduser().resolve()
        if results_dir
        else benchmark_repo_root / "results" / "legacy_v1"
    )
    logs_dir = (
        Path(logs_dir).expanduser().resolve()
        if logs_dir
        else benchmark_repo_root / "logs" / "legacy_v1"
    )
    output_dir = results_dir / "outputs"
    checkpoints_dir = results_dir / "checkpoints"

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    polars_stats = bench_polars(
        alpharank_repo=alpharank_repo,
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        n_trials=n_trials,
        first_date=first_date,
        warmups=warmups,
        runs=runs,
        logs_dir=logs_dir,
    )

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alpharank_repo": str(alpharank_repo),
        "data_dir": str(data_dir),
        "config": {
            "n_trials": n_trials,
            "first_date": first_date,
            "warmups": warmups,
            "runs": runs,
        },
        "polars": polars_stats,
    }

    out_json = results_dir / "benchmark_summary.json"
    out_csv = results_dir / "benchmark_summary.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "median_s", "mean_s", "min_s", "max_s", "runs"])
        writer.writerow(["polars", polars_stats["median"], polars_stats["mean"], polars_stats["min"], polars_stats["max"], polars_stats["runs"]])

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
