"""HTML rendering shared by data benchmark overview and detail outputs."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import polars as pl


def _frame_to_html(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "<p>No rows.</p>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    body_rows = []
    for row in df.iter_rows(named=True):
        cells = "".join(f"<td>{_format_html_value(row[column])}</td>" for column in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def _format_html_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith("<a href=") and value.endswith("</a>"):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return escape(str(value))


def _write_html_page(
    *,
    output_path: Path,
    title: str,
    subtitle: str,
    sections: list[tuple[str, pl.DataFrame]],
    navigation: str,
) -> None:
    section_html = "".join(
        f'<div class="section"><h2>{escape(name)}</h2>{_frame_to_html(frame)}</div>'
        for name, frame in sections
    )
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 12px; vertical-align: top; }}
    th {{ background: #eff6ff; position: sticky; top: 0; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    .muted {{ color: #64748b; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class=\"muted\">{escape(subtitle)}</div>
  {navigation}
  {section_html}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
