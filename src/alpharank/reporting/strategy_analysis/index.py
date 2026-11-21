"""Publish only the canonical generator-owned local report index atomically."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from alpharank.governance_contracts.common import sha256_path

INDEX_MARKER = "<!-- alpharank-complete-performance-index -->"


def publish_report_index(report: Path) -> Path:
    """Update only the owned pointer; never overwrite an unrelated user page."""
    directory = report.parent.parent
    index = directory / "index.html"
    if index.exists() and INDEX_MARKER not in index.read_text(encoding="utf-8"):
        raise ValueError(f"Refusing to replace an unrelated performance index: {index}")
    target = quote(report.relative_to(directory).as_posix(), safe="/")
    html = (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0;url={target}">'
        "<title>AlphaRank · Performance</title></head><body>"
        f'{INDEX_MARKER}<a href="{target}">Ouvrir le rapport HTML complet</a>'
        f"<p>SHA-256 : {sha256_path(report)}</p></body></html>\n"
    )
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(html)
        temporary = Path(handle.name)
    temporary.replace(index)
    return index
