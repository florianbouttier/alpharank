from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import alpharank.governance as governance
from alpharank.governance import (
    promote_methodology_version,
    rollback_methodology_version,
)


def test_promotion_is_atomic_and_reversible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = tmp_path / "versions" / "v1"
    v2 = tmp_path / "versions" / "v2"
    v1.mkdir(parents=True)
    v2.mkdir(parents=True)
    (v1 / "returns.parquet").write_bytes(b"v1 immutable economics")
    (v2 / "returns.parquet").write_bytes(b"v2 causal economics")
    pointer = tmp_path / "published" / "latest.json"
    changed_at = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)

    first = promote_methodology_version(
        pointer_path=pointer,
        version_dir=v1,
        version_id="v1",
        approved_by="methodology-owner",
        reason="initial audited publication",
        changed_at=changed_at,
    )
    original_pointer = pointer.read_bytes()
    original_hashes = first["active_record"]["artifact_hashes"]

    real_replace = governance.os.replace

    def interrupted_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption before atomic pointer swap")

    monkeypatch.setattr(governance.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        promote_methodology_version(
            pointer_path=pointer,
            version_dir=v2,
            version_id="v2",
            approved_by="methodology-owner",
            reason="causal promotion",
            changed_at=changed_at,
        )
    assert pointer.read_bytes() == original_pointer

    monkeypatch.setattr(governance.os, "replace", real_replace)
    promoted = promote_methodology_version(
        pointer_path=pointer,
        version_dir=v2,
        version_id="v2",
        approved_by="methodology-owner",
        reason="causal promotion",
        changed_at=changed_at,
    )
    assert promoted["active_version"] == "v2"
    assert promoted["version_records"]["v1"]["status"] == "superseded"

    rolled_back = rollback_methodology_version(
        pointer_path=pointer,
        target_version_id="v1",
        approved_by="methodology-owner",
        reason="validated rollback drill",
        changed_at=changed_at,
    )
    assert rolled_back["active_version"] == "v1"
    assert rolled_back["active_record"]["artifact_hashes"] == original_hashes
    assert v1.exists() and v2.exists()
    assert json.loads(pointer.read_text(encoding="utf-8"))["actions"][-1][
        "action"
    ] == "rollback"
