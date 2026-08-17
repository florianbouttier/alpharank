#!/usr/bin/env python3
"""Fail on broken relative links in tracked Markdown documentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def tracked_markdown(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line for line in completed.stdout.splitlines() if line]


def broken_links(root: Path) -> list[str]:
    errors: list[str] = []
    for document in tracked_markdown(root):
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_target = target.split("#", 1)[0]
            if not relative_target:
                continue
            resolved = (document.parent / relative_target).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{document.relative_to(root)}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{document.relative_to(root)}: missing link target: {target}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = broken_links(args.root.resolve())
    if errors:
        raise SystemExit("\n".join(errors))
    print("Tracked Markdown links are valid")


if __name__ == "__main__":
    main()
