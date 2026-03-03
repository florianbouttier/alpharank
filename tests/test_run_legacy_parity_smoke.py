import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_LEGACY_E2E", "0") != "1",
    reason="Set RUN_LEGACY_E2E=1 to run expensive end-to-end parity smoke test.",
)
def test_run_legacy_parity_smoke():
    repo_root = Path(__file__).parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_legacy.py"),
        "--parity-check",
        "--n-trials",
        "1",
        "--n-jobs",
        "1",
        "--checkpoints-dir",
        str(repo_root / "outputs" / "checkpoints_smoke"),
    ]
    subprocess.run(cmd, check=True)
