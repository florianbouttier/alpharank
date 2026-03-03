#!/usr/bin/env python3
"""Benchmark pandas vs polars runtime for scripts/run_legacy.py."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def _run_once(cmd: List[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def bench_backend(repo_root: Path, backend: str, n_trials: int, warmups: int, runs: int) -> Dict[str, float]:
    script = repo_root / "scripts" / "run_legacy.py"
    cmd = [
        "python3",
        str(script),
        "--backend",
        backend,
        "--n-trials",
        str(n_trials),
        "--n-jobs",
        "1",
    ]
    for _ in range(warmups):
        _run_once(cmd)
    samples = [_run_once(cmd) for _ in range(runs)]
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy pipeline runtime (pandas vs polars).")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--target-ratio", type=float, default=0.70, help="Expected polars/pandas median ratio.")
    parser.add_argument("--enforce-target", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    pandas_stats = bench_backend(repo_root, "pandas", args.n_trials, args.warmups, args.runs)
    polars_stats = bench_backend(repo_root, "polars", args.n_trials, args.warmups, args.runs)
    ratio = polars_stats["median"] / pandas_stats["median"]

    print("Pandas stats:", pandas_stats)
    print("Polars stats:", polars_stats)
    print(f"Median ratio (polars/pandas): {ratio:.4f}")

    if args.enforce_target and ratio > args.target_ratio:
        raise SystemExit(f"Performance target failed: ratio={ratio:.4f} > {args.target_ratio:.4f}")


if __name__ == "__main__":
    main()
