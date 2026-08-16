from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def copy_snapshot_file(source: Path | str, destination: Path | str) -> str:
    """Create an independent byte-identical file, preferring APFS copy-on-write."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        clone = subprocess.run(
            ["cp", "-c", "-p", str(source_path), str(destination_path)],
            capture_output=True,
            check=False,
        )
        if clone.returncode == 0:
            return "apfs_clone"
        destination_path.unlink(missing_ok=True)
    shutil.copy2(source_path, destination_path)
    return "physical_copy"
