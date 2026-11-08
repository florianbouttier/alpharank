#!/usr/bin/env python3
"""Build an immutable point-in-time sector-history DEF release."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alpharank.data.warehouse.sector_history import (
    load_sector_observations,
    materialize_sector_history_release,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=project_root / "data" / "open_source" / "official" / "runs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "warehouse" / "def" / "sector_history",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=project_root / "configs" / "data_contracts" / "sector_history_v1.json",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = tuple(args.runs_root.glob("*/raw/general_reference_lineage.parquet"))
    observations = load_sector_observations(paths, root=project_root)
    release = materialize_sector_history_release(
        observations,
        contract_path=args.contract,
        output_root=args.output_root,
    )
    logging.info("sector_history_release=%s", release.release_dir)
    logging.info("release_id=%s", release.release_id)
    logging.info("quality=%s", release.manifest["quality"])


if __name__ == "__main__":
    main()
