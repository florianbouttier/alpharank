from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl

from alpharank.replay.refresh_compare import TableSpec, compare_frames
from alpharank.replay.refresh_drift import (
    ReplayAuditInputs,
    audit_blocked_refresh,
    audit_refresh_replay,
)
from alpharank.replay.refresh_provenance import (
    compare_provenance_pairs,
    mapping_differences,
    stable_config,
)


def test_compare_frames_ignores_rows_after_historical_cutoff() -> None:
    spec = TableSpec("prices", "prices.parquet", ("ticker", "date"), "date", "ticker")
    baseline = pl.DataFrame({"ticker": ["A"], "date": [date(2020, 1, 1)], "adjusted_close": [10.0]})
    candidate = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": [date(2020, 1, 1), date(2020, 2, 1)],
            "adjusted_close": [10.0, 11.0],
        }
    )

    diff = compare_frames(
        baseline,
        candidate,
        spec=spec,
        historical_cutoff=date(2020, 1, 31),
        materiality_tolerance=0.0,
    )

    assert not diff.has_historical_drift
    assert diff.summary["candidate_rows"] == 1


def test_compare_frames_retains_exact_changed_key_and_columns() -> None:
    spec = TableSpec("prices", "prices.parquet", ("ticker", "date"), "date", "ticker")
    baseline = pl.DataFrame({"ticker": ["A"], "date": [date(2020, 1, 1)], "adjusted_close": [10.0]})
    candidate = baseline.with_columns(pl.lit(10.5).alias("adjusted_close"))

    diff = compare_frames(
        baseline,
        candidate,
        spec=spec,
        historical_cutoff=date(2020, 1, 31),
        materiality_tolerance=0.0,
    )

    assert diff.changed_keys.to_dicts() == [
        {
            "ticker": "A",
            "date": date(2020, 1, 1),
            "changed_columns": ["adjusted_close"],
        }
    ]
    assert diff.summary["maximum_numeric_absolute_difference"] == 0.5


def test_blocked_refresh_preserves_gate_and_hashes_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260824_001503"
    run_dir.mkdir()
    (run_dir / "price_revision_guard.json").write_text(
        json.dumps(
            {
                "failure_reasons": ["unreviewed_historical_return_revisions"],
                "historical_daily_return_revisions_over_threshold": 45,
                "historical_return_revision_tickers": 30,
                "historical_return_revision_examples": [{"ticker": "MNST.US"}],
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame({"ticker": ["MNST.US"]}).write_parquet(run_dir / "revisions.parquet")
    (run_dir / "price_validated_key_coverage.json").write_text(
        json.dumps(
            {
                "provider_complete": False,
                "raw_archive": {"manifest_path": "/raw/yahoo/manifest.json"},
                "definitive_resolution": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "price_composition.json").write_text(
        json.dumps(
            {
                "refreshable_active_ticker_count": 502,
                "active_yahoo_rows": 2_480_115,
                "preserved_history_rows": 1_232_112,
                "preserved_history_tickers": 338,
            }
        ),
        encoding="utf-8",
    )

    report = audit_blocked_refresh(run_dir, tmp_path / "baseline", tmp_path / "audit")

    assert report["status"] == "blocked_before_replay"
    assert not report["promotion_allowed_by_this_gate"]
    assert report["failed_gate"]["historical_daily_return_revisions_over_threshold"] == 45
    assert {Path(item["path"]).name for item in report["evidence"]} == {
        "price_composition.json",
        "price_revision_guard.json",
        "price_validated_key_coverage.json",
        "revisions.parquet",
    }
    assert all(len(item["sha256"]) == 64 for item in report["evidence"])
    assert not report["model_execution"]["legacy_candidate_executed"]
    statuses = {item["source"]: item["status"] for item in report["source_statuses"]}
    assert statuses["yahoo_prices"] == "downloaded_quarantined"
    assert statuses["eodhd_price_seed"] == "retained_not_redownloadable"
    assert statuses["sec_companyfacts"] == "not_started_blocked_upstream"


def test_blocked_refresh_prefers_combined_price_publication_guard(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260827_001503"
    run_dir.mkdir()
    (run_dir / "price_revision_guard.json").write_text(
        json.dumps({"passed": True, "blocking_reasons": []}), encoding="utf-8"
    )
    (run_dir / "price_publication_guard.json").write_text(
        json.dumps(
            {
                "passed": False,
                "blocking_reasons": ["unreviewed_extreme_adjusted_price_moves"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "acquisition_status.json").write_text(json.dumps({"sources": []}), encoding="utf-8")
    (run_dir / "price_composition.json").write_text("{}", encoding="utf-8")

    report = audit_blocked_refresh(run_dir, tmp_path / "baseline", tmp_path / "audit")

    assert report["failed_gate"]["name"] == "price_publication_guard"
    assert report["failed_gate"]["reasons"] == ["unreviewed_extreme_adjusted_price_moves"]


def test_blocked_refresh_uses_completed_acquisition_statuses(tmp_path: Path) -> None:
    from alpharank.replay.refresh_sources import blocked_refresh_source_statuses

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "price_composition.json").write_text(
        json.dumps({"preserved_history_rows": 10}), encoding="utf-8"
    )
    (run_dir / "acquisition_status.json").write_text(
        json.dumps(
            {
                "sources": [
                    {"source": "yahoo_prices", "status": "downloaded_quarantined"},
                    {"source": "sec_companyfacts", "status": "downloaded"},
                ]
            }
        ),
        encoding="utf-8",
    )

    statuses = {item["source"]: item["status"] for item in blocked_refresh_source_statuses(run_dir)}
    assert statuses["yahoo_prices"] == "downloaded_quarantined"
    assert statuses["sec_companyfacts"] == "downloaded"
    assert statuses["eodhd_price_seed"] == "retained_not_redownloadable"


def test_complete_audit_accepts_identical_historical_portfolios(tmp_path: Path) -> None:
    inputs = _complete_fixture(tmp_path)

    report = audit_refresh_replay(inputs, tmp_path / "audit")

    assert report["status"] == "identical_historical_portfolios"
    assert report["promotion_allowed_by_this_gate"]
    assert report["portfolio_attribution"]["portfolio_drift_rows"] == 0


def test_complete_audit_retains_a_common_replay_gate_failure(tmp_path: Path) -> None:
    inputs = replace(
        _complete_fixture(tmp_path),
        candidate_common=None,
        common_replay_failure="Selected Boosting holding CVC.US uses a censored target.",
    )

    report = audit_refresh_replay(inputs, tmp_path / "audit")

    assert report["status"] == "common_replay_blocked"
    assert not report["promotion_allowed_by_this_gate"]
    assert report["common_replay_failure"] == (
        "Selected Boosting holding CVC.US uses a censored target."
    )
    assert {item["stage"] for item in report["replay_comparison"]} == {
        "legacy_portfolio",
        "legacy_simulation",
        "boosting_signal",
    }
    assert report["portfolio_attribution"]["portfolio_drift_rows"] is None
    assert report["portfolio_attribution"]["review_status"] == ("blocked_by_common_replay_gate")


def test_complete_audit_classifies_changed_code_before_data_attribution(tmp_path: Path) -> None:
    inputs = _complete_fixture(tmp_path)
    candidate_holdings = inputs.candidate_common / "comparison_common_holdings.parquet"
    pl.read_parquet(candidate_holdings).with_columns(
        pl.lit(0.6).alias("target_weight")
    ).write_parquet(candidate_holdings)
    manifest_path = inputs.candidate_common / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_provenance"]["git"]["head"] = "candidate"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_refresh_replay(inputs, tmp_path / "audit")

    assert report["status"] == "code_config_runtime_drift"
    assert report["portfolio_attribution"]["first_divergent_stage"] == "common_portfolio"
    assert report["portfolio_attribution"]["portfolio_drift_rows"] == 1


def test_stable_config_ignores_run_paths_but_preserves_policy() -> None:
    baseline = {
        "run_output_dir": "/baseline/run",
        "source_input_files": {"prices": "/baseline/prices.parquet"},
        "source_input_sha256": {"prices": "baseline"},
        "minimum_liquidity": 1_000_000,
    }
    candidate = {
        "run_output_dir": "/candidate/run",
        "source_input_files": {"prices": "/candidate/prices.parquet"},
        "source_input_sha256": {"prices": "candidate"},
        "minimum_liquidity": 1_000_000,
    }

    assert stable_config(baseline) == stable_config(candidate) == {"minimum_liquidity": 1_000_000}


def test_provenance_lists_exact_runtime_difference(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_manifest(baseline)
    _write_manifest(candidate)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["runtime_provenance"]["resolved_config"]["run_output_dir"] = "/candidate"
    payload["runtime_provenance"]["dependencies"] = {"polars": "2.0"}
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    report = compare_provenance_pairs({"legacy": (baseline, candidate)})

    stage = report["stages"]["legacy"]
    assert stage["config_identical"]
    assert not stage["runtime_identical"]
    assert stage["runtime_differences"] == [
        {"path": "$.dependencies.polars", "baseline": "<missing>", "candidate": "2.0"}
    ]


def test_mapping_differences_uses_machine_readable_paths() -> None:
    assert mapping_differences({"policy": {"threshold": 1}}, {"policy": {"threshold": 2}}) == [
        {"path": "$.policy.threshold", "baseline": 1, "candidate": 2}
    ]


def _complete_fixture(tmp_path: Path) -> ReplayAuditInputs:
    baseline_snapshot = tmp_path / "baseline_snapshot"
    candidate_snapshot = tmp_path / "candidate_snapshot"
    for root in (baseline_snapshot, candidate_snapshot):
        _write_snapshot(root)
    roots = {}
    for label in (
        "baseline_legacy",
        "candidate_legacy",
        "baseline_boosting",
        "candidate_boosting",
        "baseline_common",
        "candidate_common",
    ):
        roots[label] = tmp_path / label
        roots[label].mkdir()
    _write_replay_artifacts(roots)
    return ReplayAuditInputs(
        baseline_snapshot=baseline_snapshot,
        candidate_snapshot=candidate_snapshot,
        historical_cutoff=date(2020, 1, 31),
        **roots,
    )


def _write_snapshot(root: Path) -> None:
    root.mkdir()
    price = pl.DataFrame({"ticker": ["A"], "date": [date(2020, 1, 1)], "adjusted_close": [10.0]})
    price.write_parquet(root / "US_Finalprice.parquet")
    price.write_parquet(root / "SP500Price.parquet")
    pl.DataFrame({"Code": ["A"], "Sector": ["Tech"]}).write_parquet(root / "US_General.parquet")
    statement = pl.DataFrame(
        {
            "ticker": ["A"],
            "date": [date(2019, 12, 31)],
            "filing_date": [date(2020, 1, 15)],
            "value": [1.0],
        }
    )
    for name in ("US_Income_statement", "US_Balance_sheet", "US_Cash_flow"):
        statement.write_parquet(root / f"{name}.parquet")
    pl.DataFrame(
        {
            "ticker": ["A"],
            "date": [date(2019, 12, 31)],
            "reportDate": [date(2020, 1, 15)],
            "epsActual": [1.0],
        }
    ).write_parquet(root / "US_Earnings.parquet")
    pl.DataFrame({"Date": ["2020-01-01"], "Ticker": ["A"], "Name": ["Alpha"]}).write_csv(
        root / "SP500_Constituents.csv"
    )


def _write_replay_artifacts(roots: dict[str, Path]) -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["Legacy"],
            "decision_month": [date(2020, 1, 1)],
            "holding_month": [date(2020, 2, 1)],
            "ticker": ["A"],
            "target_weight": [1.0],
        }
    )
    monthly = holdings.drop("ticker", "target_weight").with_columns(pl.lit(0.1).alias("net_return"))
    predictions = pl.DataFrame(
        {
            "method": ["classification"],
            "horizon": [6],
            "decision_month": [date(2020, 1, 1)],
            "ticker": ["A"],
            "score": [0.5],
        }
    )
    for label in ("baseline_legacy", "candidate_legacy"):
        holdings.write_parquet(roots[label] / "legacy_common_holdings.parquet")
        monthly.write_parquet(roots[label] / "legacy_common_monthly.parquet")
        _write_manifest(roots[label] / "data_input_manifest.json")
    for label in ("baseline_boosting", "candidate_boosting"):
        (roots[label] / "classification_h06").mkdir()
        predictions.write_parquet(roots[label] / "classification_h06/predictions.parquet")
        _write_manifest(roots[label] / "manifest.json")
    for label in ("baseline_common", "candidate_common"):
        holdings.write_parquet(roots[label] / "comparison_common_holdings.parquet")
        monthly.write_parquet(roots[label] / "comparison_common_monthly.parquet")
        _write_manifest(roots[label] / "manifest.json")


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "runtime_provenance": {
                    "git": {"head": "baseline"},
                    "critical_file_sha256": {"model.py": "hash"},
                    "resolved_config": {"random_seed": 42},
                    "runtime": {"python_version": "3.11"},
                    "dependencies_sha256": "dependencies",
                    "seeds": {"random_seed": 42},
                }
            }
        ),
        encoding="utf-8",
    )
