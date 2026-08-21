from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.prices.history import (
    resolve_previous_validated_price_lineage,
)


def _write_retained_snapshot(root: Path, *, valid_gate: bool = True) -> tuple[Path, Path]:
    snapshot = root / "history" / "snapshot"
    price_lineage_dir = snapshot / "lineage" / "prices"
    price_lineage_dir.mkdir(parents=True)
    lineage_path = price_lineage_dir / "prices_open_source_lineage.parquet"
    pl.DataFrame(
        {
            "ticker": ["CI.US"],
            "date": ["2026-07-31"],
            "source": ["yfinance"],
        }
    ).write_parquet(lineage_path)
    lineage_hash = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    (price_lineage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_refresh_contract": {
                    "price_revision_guard": {"passed": valid_gate}
                },
                "artifacts": {"price_lineage": {"sha256": lineage_hash}},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "lineage" / "manifest.json").write_text(
        json.dumps(
            {
                "composition_id": "composition-1",
                "validation": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    pointer = root / "manifests" / "latest.json"
    pointer.parent.mkdir()
    pointer.write_text(
        json.dumps(
            {
                "composition_id": "composition-1",
                "snapshot_dir": str(snapshot),
            }
        ),
        encoding="utf-8",
    )
    return pointer, lineage_path


def test_resolve_previous_lineage_from_latest_composed_snapshot(tmp_path: Path) -> None:
    pointer, lineage_path = _write_retained_snapshot(tmp_path)

    source = resolve_previous_validated_price_lineage(pointer)

    assert source.lineage_path == lineage_path.resolve()
    assert source.composition_id == "composition-1"


def test_resolve_previous_lineage_rejects_failed_price_gate(tmp_path: Path) -> None:
    pointer, _ = _write_retained_snapshot(tmp_path, valid_gate=False)

    with pytest.raises(RuntimeError, match="price gate"):
        resolve_previous_validated_price_lineage(pointer)
