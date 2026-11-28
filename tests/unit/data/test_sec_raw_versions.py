from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from alpharank.data.sources.sec_raw_versions import (
    build_sec_raw_version_candidate,
    rebuild_full_companyfacts_versions,
)


def _frame(rows: list[tuple[str, str, float, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [row[0] for row in rows],
            "statement": ["shares"] * len(rows),
            "metric": ["outstanding_shares"] * len(rows),
            "date": ["2020-12-31"] * len(rows),
            "filing_date": [row[1] for row in rows],
            "value": [row[2] for row in rows],
            "source": ["sec_companyfacts"] * len(rows),
            "ingested_at": [row[3] for row in rows],
        }
    )


def test_full_companyfacts_rebuild_preserves_versions_and_unrefreshed_tickers() -> None:
    retained = _frame(
        [
            ("AAA.US", "2020-02-01", 90.0, "old"),
            ("AAA.US", "2022-02-01", 110.0, "old"),
            ("FAILED.US", "2021-02-01", 50.0, "old"),
        ]
    )
    refresh = _frame(
        [
            ("AAA.US", "2021-02-01", 100.0, "new"),
            ("AAA.US", "2022-02-01", 110.0, "new"),
        ]
    )

    rebuilt = rebuild_full_companyfacts_versions(
        retained=retained,
        full_refresh=refresh,
    )

    assert rebuilt.select("ticker", "filing_date", "value").rows() == [
        ("AAA.US", "2020-02-01", 90.0),
        ("AAA.US", "2021-02-01", 100.0),
        ("AAA.US", "2022-02-01", 110.0),
        ("FAILED.US", "2021-02-01", 50.0),
    ]


def test_raw_candidate_merges_every_run_delta(tmp_path: Path) -> None:
    retained_dir = tmp_path / "retained"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "output"
    retained_dir.mkdir()
    run_dir.mkdir()

    _frame(
        [
            ("AAA.US", "2021-02-01", 100.0, "2026-08-01T00:00:00Z"),
            ("AAA.US", "2022-02-01", 110.0, "2026-08-01T00:00:00Z"),
        ]
    ).write_parquet(retained_dir / "financials_sec_companyfacts.parquet")
    _frame([("AAA.US", "2022-02-01", 111.0, "2026-09-01T00:00:00Z")]).write_parquet(
        run_dir / "financials_sec_companyfacts.parquet"
    )
    _write_delta_inputs(retained_dir, run_dir)

    report = build_sec_raw_version_candidate(
        retained_raw_dir=retained_dir,
        run_raw_dir=run_dir,
        output_dir=output_dir,
    )

    general = pl.read_parquet(output_dir / "general_reference_lineage.parquet")
    filing = pl.read_parquet(output_dir / "financials_sec_filing.parquet")
    assert general.sort("ticker").select("ticker", "Sector").rows() == [
        ("AAA.US", "Technology"),
        ("BBB.US", "Health Care"),
    ]
    assert filing.select("filing_date").sort("filing_date").to_series().to_list() == [
        "2021-02-01",
        "2022-02-01",
    ]
    assert report["datasets"]["earnings_sec_calendar.parquet"]["output_rows"] == 2
    facts = pl.read_parquet(output_dir / "financials_sec_companyfacts.parquet")
    assert facts.select("filing_date", "value").rows() == [
        ("2021-02-01", 100.0),
        ("2022-02-01", 111.0),
    ]
    observations = pl.read_parquet(output_dir / "companyfacts_version_observations.parquet")
    assert observations.select(
        "filing_date", "observation", "retained_value", "observed_value"
    ).rows() == [
        ("2021-02-01", "absent_retained_version", 100.0, None),
        ("2022-02-01", "observed_value_revision", 110.0, 111.0),
    ]
    assert report["retained_raw_sources"]["financials_sec_companyfacts.parquet"]["sha256"]
    assert report["run_raw_sources"]["financials_sec_companyfacts.parquet"]["sha256"]
    assert report["cumulative_raw_outputs"]["financials_sec_companyfacts.parquet"]["sha256"]
    assert report["companyfacts_version_observations"]["counts"] == {
        "absent_retained_version": 1,
        "observed_value_revision": 1,
    }


def test_raw_candidate_refuses_to_replace_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    previous_evidence = output / "seal.json"
    previous_evidence.write_text("retained\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_sec_raw_version_candidate(
            retained_raw_dir=tmp_path / "retained",
            run_raw_dir=tmp_path / "delta",
            output_dir=output,
        )

    assert previous_evidence.read_text(encoding="utf-8") == "retained\n"


def test_raw_candidate_refuses_conflicting_sec_issuer_identities(tmp_path: Path) -> None:
    retained_dir = tmp_path / "retained"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "output"
    retained_dir.mkdir()
    run_dir.mkdir()
    _write_delta_inputs(retained_dir, run_dir)
    for directory, cik in ((retained_dir, "0000000001"), (run_dir, "0000000002")):
        _frame([("AAA.US", "2022-02-01", 110.0, "2026-08-01T00:00:00Z")]).write_parquet(
            directory / "financials_sec_companyfacts.parquet"
        )
        pl.DataFrame({"ticker": ["AAA.US"], "sec_cik": [cik]}).write_parquet(
            directory / "general_reference_lineage.parquet"
        )

    with pytest.raises(RuntimeError, match="disagree on issuer CIK.*AAA.US"):
        build_sec_raw_version_candidate(
            retained_raw_dir=retained_dir,
            run_raw_dir=run_dir,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def _write_delta_inputs(retained_dir: Path, run_dir: Path) -> None:
    retained_at = "2026-08-01T00:00:00Z"
    refreshed_at = "2026-09-01T00:00:00Z"
    retained_financial = _frame([("AAA.US", "2021-02-01", 100.0, retained_at)])
    run_financial = _frame([("AAA.US", "2022-02-01", 110.0, refreshed_at)])
    retained_financial.write_parquet(retained_dir / "financials_sec_filing.parquet")
    run_financial.write_parquet(run_dir / "financials_sec_filing.parquet")
    retained_earnings = _earnings_frame("AAA.US", retained_at)
    run_earnings = _earnings_frame("BBB.US", refreshed_at)
    for name in ("earnings_sec_calendar.parquet", "earnings_sec_actuals.parquet"):
        retained_earnings.write_parquet(retained_dir / name)
        run_earnings.write_parquet(run_dir / name)
    pl.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US"],
            "source": ["sec", "sec"],
            "Sector": ["Industrials", "Health Care"],
            "ingested_at": [retained_at, retained_at],
        }
    ).write_parquet(retained_dir / "general_reference_lineage.parquet")
    pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "source": ["sec"],
            "Sector": ["Technology"],
            "ingested_at": [refreshed_at],
        }
    ).write_parquet(run_dir / "general_reference_lineage.parquet")


def _earnings_frame(ticker: str, ingested_at: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [ticker],
            "period_end": ["2022-03-31"],
            "reportDate": ["2022-05-01"],
            "accession_number": [f"accession-{ticker}"],
            "source": ["sec"],
            "ingested_at": [ingested_at],
        }
    )
