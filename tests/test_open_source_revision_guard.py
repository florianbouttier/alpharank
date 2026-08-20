from __future__ import annotations

import inspect
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.open_source.ingestion import (
    _audit_and_validate_historical_revisions,
    _run_open_source_ingestion_in_place,
)
from alpharank.data.open_source.refresh_policy import SourceRefreshPolicy
from alpharank.data.open_source.revision_guard import audit_historical_revisions
from alpharank.data.open_source.storage import OpenSourceLivePaths


def _income(value: str, *, date: str = "2020-03-31") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "date": [date],
            "filing_date": ["2020-05-01"],
            "totalRevenue": [value],
        }
    )


def test_revision_guard_blocks_changed_old_fundamentals(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    candidate = tmp_path / "candidate"
    previous.mkdir()
    candidate.mkdir()
    _income("100").write_parquet(previous / "US_Income_statement.parquet")
    candidate_path = candidate / "US_Income_statement.parquet"
    _income("101").write_parquet(candidate_path)

    result = audit_historical_revisions(
        previous_output_dir=previous,
        candidate_paths={candidate_path.name: candidate_path},
        expected_through="2026-08-13",
        guard_days=730,
    )

    assert result["historical_revisions_detected"] is True
    assert result["blocked_datasets"] == ["income_statement"]
    assert result["datasets"]["income_statement"]["changed_common_rows"] == 1


def test_revision_guard_ignores_rows_inside_mutable_window(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    candidate = tmp_path / "candidate"
    previous.mkdir()
    candidate.mkdir()
    _income("100", date="2026-03-31").write_parquet(previous / "US_Income_statement.parquet")
    candidate_path = candidate / "US_Income_statement.parquet"
    _income("101", date="2026-03-31").write_parquet(candidate_path)

    result = audit_historical_revisions(
        previous_output_dir=previous,
        candidate_paths={candidate_path.name: candidate_path},
        expected_through="2026-08-13",
        guard_days=730,
    )

    assert result["historical_revisions_detected"] is False


def test_ingestion_revision_guard_writes_report_and_contract(tmp_path: Path) -> None:
    paths = OpenSourceLivePaths(tmp_path / "open_source" / "official")
    paths.ensure()
    candidate = tmp_path / "candidate" / "US_Income_statement.parquet"
    candidate.parent.mkdir()
    _income("100").write_parquet(paths.output_dir / candidate.name)
    _income("101").write_parquet(candidate)
    contract: dict[str, object] = {}

    report = _audit_and_validate_historical_revisions(
        paths=paths,
        run_id="test_run",
        legacy_paths={"income": candidate},
        expected_through="2026-08-13",
        source_refresh_policy=SourceRefreshPolicy(
            allow_historical_revisions=True,
            historical_revision_review_note="Reviewed migration ticket LIVE-014.",
        ),
        source_refresh_contract=contract,
    )

    report_path = paths.run_dir("test_run") / "historical_revision_guard.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == report
    assert contract["historical_revision_guard"] == report
    assert report["override_enabled"] is True
    assert report["approval_recorded"] is True
    assert report["revision_review_note"] == "Reviewed migration ticket LIVE-014."


def test_ingestion_revision_override_requires_review_note(tmp_path: Path) -> None:
    paths = OpenSourceLivePaths(tmp_path / "open_source" / "official")
    paths.ensure()
    candidate = tmp_path / "candidate" / "US_Income_statement.parquet"
    candidate.parent.mkdir()
    _income("100").write_parquet(paths.output_dir / candidate.name)
    _income("101").write_parquet(candidate)

    with pytest.raises(RuntimeError, match="requires a non-empty review note"):
        _audit_and_validate_historical_revisions(
            paths=paths,
            run_id="test_run",
            legacy_paths={"income": candidate},
            expected_through="2026-08-13",
            source_refresh_policy=SourceRefreshPolicy(
                allow_historical_revisions=True,
            ),
            source_refresh_contract={},
        )

    report = json.loads(
        (paths.run_dir("test_run") / "historical_revision_guard.json").read_text()
    )
    assert report["override_enabled"] is True
    assert report["approval_recorded"] is False
    assert report["revision_review_note"] is None


def test_ingestion_revision_guard_blocks_without_explicit_override(tmp_path: Path) -> None:
    paths = OpenSourceLivePaths(tmp_path / "open_source" / "official")
    paths.ensure()
    candidate = tmp_path / "candidate" / "US_Income_statement.parquet"
    candidate.parent.mkdir()
    _income("100").write_parquet(paths.output_dir / candidate.name)
    _income("101").write_parquet(candidate)

    with pytest.raises(RuntimeError, match="No package was published"):
        _audit_and_validate_historical_revisions(
            paths=paths,
            run_id="test_run",
            legacy_paths={"income": candidate},
            expected_through="2026-08-13",
            source_refresh_policy=SourceRefreshPolicy(),
            source_refresh_contract={},
        )

    assert (paths.run_dir("test_run") / "historical_revision_guard.json").exists()


def test_full_ingestion_guards_historical_revisions_before_publication() -> None:
    source = inspect.getsource(_run_open_source_ingestion_in_place)

    price_guard_position = source.index("_prepare_canonical_hybrid_price_merge(")
    guard_position = source.index("_audit_and_validate_historical_revisions(")
    publish_position = source.index("publish_open_source_output_package(")

    assert price_guard_position < publish_position
    assert guard_position < publish_position
    assert "latest_composed_manifest_path" in source
