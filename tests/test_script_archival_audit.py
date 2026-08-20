from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "docs" / "architecture" / "script_archival_audit_v1.json"
INVENTORY_PATH = ROOT / "docs" / "architecture" / "code_dependency_inventory_v1.json"


def test_archived_scripts_preserve_bytes_and_have_no_active_readers() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    nodes = {row["path"]: row for row in inventory["nodes"]}

    archived = [row for row in audit["candidates"] if row["decision"] == "archived"]
    assert len(archived) == 7
    for row in archived:
        archive_path = ROOT / row["archive_path"]
        assert row["active_readers"] == []
        assert not (ROOT / row["source_path"]).exists()
        assert archive_path.is_file()
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == row["sha256_before_move"]
        assert nodes[row["archive_path"]]["lifecycle"] == "archived"


def test_candidate_with_current_readers_is_retained() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    retained = [row for row in audit["candidates"] if row["decision"] == "retained"]

    assert len(retained) == 1
    assert retained[0]["active_readers"]
    assert (ROOT / retained[0]["source_path"]).is_file()
