from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from alpharank.governance import (
    RuntimeProvenanceError,
    capture_runtime_provenance,
    validate_runtime_provenance,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "audit@example.test")
    _git(repo, "config", "user.name", "Audit Test")
    (repo / "model.py").write_text("SEED = 42\n", encoding="utf-8")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_manifest_captures_complete_runtime_provenance(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    (repo / "model.py").write_text("SEED = 43\n", encoding="utf-8")
    (repo / "untracked.py").write_text("FEATURE = 'causal'\n", encoding="utf-8")
    patch_path = tmp_path / "runtime_git_patch.json"

    provenance = capture_runtime_provenance(
        project_root=repo,
        entrypoint="tests.synthetic_run",
        command_argv=["python", "run.py", "--snapshot", "v2"],
        resolved_config={
            "snapshot": "v2",
            "api_token": "must-not-leak",
        },
        seeds={"model": 42},
        critical_files=("model.py",),
        data_identifiers={"snapshot_id": "snapshot-20260817"},
        patch_path=patch_path,
    )

    validation = validate_runtime_provenance(provenance, project_root=repo)
    patch_bundle = json.loads(patch_path.read_text(encoding="utf-8"))

    assert validation["passed"] is True
    assert validation["git_dirty"] is True
    assert provenance["resolved_config"]["api_token"] == "<redacted>"
    assert provenance["git"]["tracked_diff_bytes"] > 0
    assert provenance["git"]["untracked_file_count"] == 1
    assert patch_bundle["tracked_diff_sha256"] == provenance["git"][
        "tracked_diff_sha256"
    ]
    assert patch_bundle["untracked_files"][0]["relative_path"] == "untracked.py"
    assert provenance["dependencies_sha256"]
    assert provenance["critical_file_sha256"]["model.py"]

    falsely_clean = deepcopy(provenance)
    falsely_clean["git"]["dirty"] = False
    with pytest.raises(RuntimeProvenanceError, match="git_dirty"):
        validate_runtime_provenance(falsely_clean, project_root=repo)


def test_runtime_provenance_rejects_missing_required_field(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    provenance = capture_runtime_provenance(
        project_root=repo,
        entrypoint="tests.synthetic_run",
        command_argv=["python", "run.py"],
        resolved_config={"mode": "test"},
        seeds={"model": 42},
        critical_files=("model.py",),
        data_identifiers={"snapshot_id": "snapshot-test"},
        patch_path=tmp_path / "runtime_git_patch.json",
    )
    incomplete = dict(provenance)
    incomplete.pop("critical_file_sha256")

    with pytest.raises(RuntimeProvenanceError, match="critical_file_sha256"):
        validate_runtime_provenance(incomplete)
