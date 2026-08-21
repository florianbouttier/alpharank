from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.governance import reserve_run_directory


def test_run_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fixed-run-id"

    reserved = reserve_run_directory(run_dir)
    sentinel = reserved / "first-run.txt"
    sentinel.write_text("immutable first run", encoding="utf-8")

    with pytest.raises(FileExistsError, match="cannot be reused"):
        reserve_run_directory(run_dir)

    assert sentinel.read_text(encoding="utf-8") == "immutable first run"
    assert list(reserved.iterdir()) == [sentinel]
