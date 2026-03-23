#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil

from alpharank.data.output_history import snapshot_output_directory


EXACT_NAMES = (
    "SP500Price.parquet",
    "SP500_Constituents.csv",
    "US_Balance_sheet.parquet",
    "US_Cash_flow.parquet",
    "US_Earnings.parquet",
    "US_Finalprice.parquet",
    "US_General.parquet",
    "US_Income_statement.parquet",
    "US_share.parquet",
)


def main(*, source_dir: str | Path | None = None, output_dir: str | Path | None = None) -> None:
    project_root = Path(__file__).resolve().parent.parent
    resolved_source_dir = Path(source_dir).expanduser().resolve() if source_dir else project_root / "data"
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else project_root / "data" / "eodhd" / "output"
    history_root = resolved_output_dir.parent / "history" / "output"

    snapshot_dir = snapshot_output_directory(
        resolved_output_dir,
        history_root=history_root,
        snapshot_prefix="eodhd_output",
        metadata={"source_dir": str(resolved_source_dir), "output_dir": str(resolved_output_dir)},
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for file_name in EXACT_NAMES:
        source_path = resolved_source_dir / file_name
        destination_path = resolved_output_dir / file_name
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)

    print(f"EODHD output synced to: {resolved_output_dir}")
    if snapshot_dir is not None:
        print(f"Previous output snapshot: {snapshot_dir}")
    for path in copied:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
