#!/usr/bin/env python3
"""Benchmark pandas vs polars runtime for scripts/run_legacy.py."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def _run_once(cmd: List[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def bench_legacy_runtime(
    repo_root: Path,
    *,
    n_trials: int,
    warmups: int,
    runs: int,
) -> Dict[str, float]:
    checkpoint_dir = repo_root / "outputs" / "benchmark_checkpoints"
    cmd = [
        sys.executable,
        "-c",
        (
            "from scripts.run_legacy import main; "
            f"main(n_trials={n_trials}, n_jobs=1, checkpoints_dir={str(checkpoint_dir)!r})"
        ),
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


def main(
    *,
    n_trials: int = 5,
    warmups: int = 3,
    runs: int = 5,
    target_median_seconds: float | None = None,
    enforce_target: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    stats = bench_legacy_runtime(
        repo_root,
        n_trials=n_trials,
        warmups=warmups,
        runs=runs,
    )

    print("Legacy runtime stats:", stats)

    if enforce_target and target_median_seconds is not None and stats["median"] > target_median_seconds:
        raise SystemExit(
            f"Performance target failed: median={stats['median']:.4f}s > {target_median_seconds:.4f}s"
        )


if __name__ == "__main__":
    main()
