from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "open_source" / "run_sec_q4_fix2_candidate.py"
SPEC = importlib.util.spec_from_file_location("run_sec_q4_fix2_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

load_target_tickers = MODULE._load_target_tickers


def test_load_target_tickers_aggregates_top_tickers_from_default_kpi_year_pairs(tmp_path: Path) -> None:
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    holes = pl.DataFrame(
        {
            "ticker": [
                "AAA.US",
                "BBB.US",
                "CCC.US",
                "DDD.US",
                "EEE.US",
                "FFF.US",
            ],
            "metric": [
                "revenue",
                "revenue",
                "net_income",
                "net_income",
                "epsActual",
                "epsActual",
            ],
            "fiscal_year": [2023, 2023, 2023, 2023, 2022, 2022],
            "present": [False, False, False, False, False, False],
        }
    )
    holes.write_parquet(quality_dir / "quarterly_holes.parquet")

    tickers = load_target_tickers(quality_dir=quality_dir, max_tickers_per_pair=1)

    assert tickers == ["AAA.US", "CCC.US", "EEE.US"]
