from pathlib import Path

from alpharank.data.warehouse import WarehousePaths


def test_warehouse_paths_expose_raw_stg_def_and_mart(tmp_path: Path) -> None:
    paths = WarehousePaths(tmp_path / "warehouse")
    paths.ensure()

    assert paths.raw == tmp_path / "warehouse" / "raw"
    assert paths.stg == tmp_path / "warehouse" / "stg"
    assert paths.definitive == tmp_path / "warehouse" / "def"
    assert paths.mart == tmp_path / "warehouse" / "mart"
    assert paths.manifests == tmp_path / "warehouse" / "manifests"
    assert all(path.is_dir() for path in (paths.raw, paths.stg, paths.definitive, paths.mart, paths.manifests))
