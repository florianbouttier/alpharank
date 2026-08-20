from __future__ import annotations

from pathlib import Path

from alpharank.quality.error_handling import audit_error_handling

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_code_respects_explicit_error_handling_policy() -> None:
    report = audit_error_handling(ROOT)

    assert report["passed"] is True
    assert report["checked_file_count"] >= 200


def test_policy_rejects_silent_and_unstructured_failures(tmp_path: Path) -> None:
    library_path = tmp_path / "src" / "alpharank" / "broken.py"
    library_path.parent.mkdir(parents=True)
    library_path.write_text(
        "def broken():\n"
        "    try:\n"
        "        print('hidden')\n"
        "    except:\n"
        "        pass\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "broken.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "def broken():\n"
        "    try:\n"
        "        raise RuntimeError('boom')\n"
        "    except Exception:\n"
        "        return 0\n",
        encoding="utf-8",
    )

    report = audit_error_handling(tmp_path, (library_path, script_path))

    assert report["passed"] is False
    assert {row["code"] for row in report["violations"]} == {
        "bare_except",
        "broad_exception",
        "library_print",
    }
