from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHITECTURE = ROOT / "docs" / "architecture"


def test_every_legacy_reader_has_a_validated_migration_decision() -> None:
    registry = json.loads(
        (ARCHITECTURE / "data_reader_migration_v1.json").read_text(
            encoding="utf-8"
        )
    )

    report = registry["summary"]
    decisions = registry["reader_decisions"]
    comparisons = registry["path_comparisons"]

    assert report["reader_edge_count"] == report["classified_reader_edge_count"]
    assert report["unclassified_reader_edge_count"] == 0
    assert report["different_path_count"] > 0
    assert report["default_entrypoint_count"] == 2
    assert len(decisions) == report["reader_edge_count"]
    assert len(comparisons) == report["compared_path_count"]
    assert all(row["old_sha256"] and row["new_sha256"] for row in comparisons)
    assert registry["validation"] == {
        "all_legacy_paths_compared": True,
        "all_reader_edges_classified": True,
        "passed": True,
        "silent_substitution_for_different_bytes": False,
    }
