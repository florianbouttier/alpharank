from __future__ import annotations

import base64
import gzip
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio import ValuationWindow, frozen_valuation_curves
from alpharank.reporting._performance_report_html import render_performance_report_html
from alpharank.reporting.performance_followup import (
    attach_frozen_followup,
    write_frozen_performance_report,
)


def _payload() -> dict:
    return {
        "generated_at_utc": "2026-10-30T10:00:00Z",
        "strategy_order": ["Frozen", "SPY · Total return"],
        "metric_fields": ["cagr", "max_drawdown"],
        "calendar": {"start": "2011-08-01", "end": "2026-09-01", "months": 182},
        "metric_windows": {"historical": [[0.1, -0.2], [0.05, -0.1]]},
        "monthly": [{"strategy": "Frozen", "net_return": 0.1}],
        "holdings": [{"ticker": "UNCHANGED", "target_weight": 1.0}],
        "current_portfolio": {
            "holding_month": "2026-10-01",
            "holdings": [{"strategy": "Frozen", "ticker": "A", "target_weight": 1.0}],
        },
        "portfolio_composer": {"combination_masks": [1, 2, 3]},
        "strategy_frontiers": {"windows": ["historical"]},
        "methodologies": [{"title": "Frozen", "pseudo_code": ["Keep this method"]}],
        "lineage": {"composition_id": "unchanged-snapshot"},
    }


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A", "A", "A", "SPY.US", "SPY.US", "SPY.US"],
            "date": [date(2026, 9, 30), date(2026, 10, 30), date(2026, 11, 20)] * 2,
            "adjusted_close": [100.0, 90.0, 80.0, 100.0, 105.0, 110.0],
        }
    )


def _curves() -> pl.DataFrame:
    return frozen_valuation_curves(
        pl.from_dicts(_payload()["current_portfolio"]["holdings"]),
        _prices(),
        entry_date=date(2026, 9, 30),
        windows=[ValuationWindow("Suivi", date(2026, 9, 30), date(2026, 11, 20))],
    )


def _decode(html: str) -> dict:
    match = re.search(r'const PAYLOAD_GZIP_BASE64="([^"]+)"', html)
    assert match is not None
    return json.loads(gzip.decompress(base64.b64decode(match.group(1))))


def test_followup_preserves_every_historical_field_and_does_not_mutate_source() -> None:
    source = _payload()
    before = deepcopy(source)
    combined = attach_frozen_followup(
        source, _curves(), provenance={"generated_at_utc": "2026-11-21T12:00:00Z"}
    )

    assert source == before
    for key, value in source.items():
        if key != "generated_at_utc":
            assert combined[key] == value
    recent = combined["recent_valuation"]
    assert recent["latest_close"] == "2026-11-20"
    assert recent["metric_fields"] == ["total_return", "max_drawdown"]
    assert recent["windows"]["Suivi"]["metrics"]["Frozen"]["total_return"] == pytest.approx(-0.2)


def test_followup_rejects_missing_canonical_strategy() -> None:
    with pytest.raises(ValueError, match="exactly"):
        attach_frozen_followup(
            _payload(),
            _curves().filter(pl.col("strategy") == "Frozen"),
            provenance={"generated_at_utc": "2026-11-21T12:00:00Z"},
        )


def test_followup_rejects_misaligned_dates_in_one_curve() -> None:
    incomplete = _curves().filter(
        ~((pl.col("strategy") == "Frozen") & (pl.col("date") == date(2026, 10, 30)))
    )
    with pytest.raises(ValueError, match="common calendar"):
        attach_frozen_followup(
            _payload(), incomplete, provenance={"generated_at_utc": "2026-11-21T12:00:00Z"}
        )


def test_writer_delivers_complete_html_and_updates_only_its_owned_index(tmp_path: Path) -> None:
    provenance = {"generated_at_utc": "2026-11-21T12:00:00Z"}
    first = write_frozen_performance_report(
        _payload(), _curves(), provenance=provenance, output_dir=tmp_path / "reports" / "first"
    )
    first_hash = sha256_path(first["report"])
    second = write_frozen_performance_report(
        _payload(), _curves(), provenance=provenance, output_dir=tmp_path / "reports" / "second"
    )
    html = second["report"].read_text()

    assert "url=second/backtest_performance_report.html" in second["index"].read_text()
    assert sha256_path(first["report"]) == first_hash
    assert _decode(html)["portfolio_composer"] == _payload()["portfolio_composer"]
    assert "AlphaRankAnalysis.mount" in html
    for field in ("monthly", "holdings", "current_portfolio", "methodologies", "lineage"):
        assert _decode(html)[field] == _payload()[field]


def test_writer_never_replaces_an_unrelated_index(tmp_path: Path) -> None:
    index = tmp_path / "index.html"
    index.write_text("User-owned page", encoding="utf-8")

    with pytest.raises(ValueError, match="unrelated"):
        write_frozen_performance_report(
            _payload(),
            _curves(),
            provenance={"generated_at_utc": "2026-11-21T12:00:00Z"},
            output_dir=tmp_path / "report",
        )
    assert index.read_text() == "User-owned page"


def test_price_followup_cli_cannot_deliver_markdown_without_the_complete_html(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.html"
    source.write_text(render_performance_report_html(_payload()), encoding="utf-8")
    objects = tmp_path / "raw" / "objects"
    receipts = tmp_path / "raw" / "receipts"
    objects.mkdir(parents=True)
    receipts.mkdir()
    raw = objects / "prices.json"
    rows = [dict(row, date=row["date"].isoformat()) for row in _prices().to_dicts()]
    raw.write_text(json.dumps({"prices": rows}), encoding="utf-8")
    receipt = receipts / "download.json"
    receipt.write_text(
        json.dumps(
            {
                "source_name": "yfinance",
                "response_status": 200,
                "payload_object_path": "objects/prices.json",
                "payload_sha256": sha256_path(raw),
                "requested_scope": {"report_sha256": sha256_path(source)},
                "retrieved_at": "2026-11-21T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[3] / "scripts/research/value_frozen_portfolios.py"
    output = tmp_path / "reports" / "followup"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-report",
            str(source),
            "--analysis-frequency",
            "monthly_archive",
            "--raw-receipt",
            str(receipt),
            "--entry-date",
            "2026-09-30",
            "--as-of",
            "2026-11-21",
            "--window",
            "Suivi:2026-09-30:2026-11-20",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "backtest_performance_report.html").is_file()
    assert "backtest_performance_report.html" in result.stdout
    assert "recent_valuation" in _decode((output / "backtest_performance_report.html").read_text())
