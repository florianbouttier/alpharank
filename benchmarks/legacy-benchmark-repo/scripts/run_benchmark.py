#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
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
    script = alpharank_repo / "scripts" / "run_legacy.py"
    cmd = [
        "python3",
        str(script),
        "--n-trials",
        str(n_trials),
        "--n-jobs",
        "1",
        "--first-date",
        first_date,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--checkpoints-dir",
        str(checkpoints_dir),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy polars runtime on a subset dataset.")
    parser.add_argument("--alpharank-repo", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--logs-dir", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--first-date", type=str, default="2020-01")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    alpharank_repo = Path(args.alpharank_repo).resolve()
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    output_dir = results_dir / "outputs"
    checkpoints_dir = results_dir / "checkpoints"

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    polars_stats = bench_polars(
        alpharank_repo=alpharank_repo,
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        n_trials=args.n_trials,
        first_date=args.first_date,
        warmups=args.warmups,
        runs=args.runs,
        logs_dir=logs_dir,
    )

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alpharank_repo": str(alpharank_repo),
        "data_dir": str(data_dir),
        "config": {
            "n_trials": args.n_trials,
            "first_date": args.first_date,
            "warmups": args.warmups,
            "runs": args.runs,
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
