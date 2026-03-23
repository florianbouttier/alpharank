from __future__ import annotations

from pathlib import Path

from alpharank.data.open_source.benchmark import resolve_eodhd_output_dir


def test_resolve_eodhd_output_dir_prefers_nested_mirror_when_root_is_not_packaged(tmp_path: Path) -> None:
    root = tmp_path / "data"
    nested = root / "eodhd" / "output"
    nested.mkdir(parents=True)
    (nested / "US_Finalprice.parquet").write_text("", encoding="utf-8")
    (nested / "SP500_Constituents.csv").write_text("", encoding="utf-8")

    assert resolve_eodhd_output_dir(root) == nested


def test_resolve_eodhd_output_dir_keeps_direct_output_dir_when_already_packaged(tmp_path: Path) -> None:
    direct = tmp_path / "output"
    direct.mkdir(parents=True)
    (direct / "US_Finalprice.parquet").write_text("", encoding="utf-8")
    (direct / "SP500_Constituents.csv").write_text("", encoding="utf-8")

    assert resolve_eodhd_output_dir(direct) == direct
