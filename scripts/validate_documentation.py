#!/usr/bin/env python3
"""Validate the repository's local documentation structure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_README_DIRS = (
    "configs",
    "configs/data_contracts",
    "configs/data_quality",
    "configs/quality",
    "configs/research",
    "data",
    "docs",
    "docs/research",
    "scripts",
    "scripts/_old",
    "scripts/experiments",
    "scripts/open_source",
    "scripts/quality",
    "src",
    "src/alpharank",
    "src/alpharank/backtest",
    "src/alpharank/data",
    "src/alpharank/data/open_source",
    "src/alpharank/data/open_source/reference",
    "src/alpharank/data/prices",
    "src/alpharank/features",
    "src/alpharank/models",
    "src/alpharank/multihorizon",
    "src/alpharank/portfolio",
    "src/alpharank/portfolio/adapters",
    "src/alpharank/quality",
    "src/alpharank/strategy",
    "src/alpharank/utils",
    "src/alpharank/visualization",
    "tests",
)

DIRECTORY_INDEX_DIRS = (
    "configs",
    "data",
    "docs/research",
    "scripts",
    "src",
    "src/alpharank",
    "src/alpharank/data",
    "src/alpharank/data/open_source",
    "src/alpharank/portfolio",
)

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IGNORED_CHILDREN = {"__pycache__", ".pytest_cache"}


def _local_markdown_links(path: Path) -> list[Path]:
    links: list[Path] = []
    for raw_target in LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
        target = raw_target.split("#", 1)[0].strip().strip("<>")
        if not target or "://" in target or target.startswith(("mailto:", "/")):
            continue
        links.append((path.parent / target).resolve())
    return links


def validate(root: Path) -> list[str]:
    errors: list[str] = []

    for relative_dir in REQUIRED_README_DIRS:
        readme = root / relative_dir / "README.md"
        if not readme.is_file():
            errors.append(f"missing README: {readme.relative_to(root)}")

    for relative_dir in DIRECTORY_INDEX_DIRS:
        directory = root / relative_dir
        readme = directory / "README.md"
        if not readme.is_file():
            continue
        content = readme.read_text(encoding="utf-8")
        child_dirs = sorted(
            child.name
            for child in directory.iterdir()
            if child.is_dir()
            and child.name not in IGNORED_CHILDREN
            and not child.name.endswith(".egg-info")
        )
        for child in child_dirs:
            if f"`{child}/`" not in content:
                errors.append(
                    f"unexplained child directory: {relative_dir}/{child}/"
                )

    markdown_roots = (root / "README.md", root / "docs", root / "configs")
    markdown_files: list[Path] = []
    for markdown_root in markdown_roots:
        if markdown_root.is_file():
            markdown_files.append(markdown_root)
        elif markdown_root.is_dir():
            markdown_files.extend(markdown_root.rglob("*.md"))

    for markdown_file in markdown_files:
        for target in _local_markdown_links(markdown_file):
            if not target.exists():
                errors.append(
                    "broken local link: "
                    f"{markdown_file.relative_to(root)} -> {target}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    if errors:
        print("Documentation validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Documentation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
