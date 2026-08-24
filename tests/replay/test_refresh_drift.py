from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from alpharank.replay.refresh_compare import TableSpec, compare_frames
from alpharank.replay.refresh_drift import (
    ReplayAuditInputs,
    audit_blocked_refresh,
    audit_refresh_replay,
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

    report = audit_blocked_refresh(run_dir, tmp_path / "baseline", tmp_path / "audit")

    assert report["status"] == "blocked_before_replay"
    assert not report["promotion_allowed_by_this_gate"]
    assert report["failed_gate"]["historical_daily_return_revisions_over_threshold"] == 45
    assert {Path(item["path"]).name for item in report["evidence"]} == {
        "price_revision_guard.json",
        "revisions.parquet",
    }
    assert all(len(item["sha256"]) == 64 for item in report["evidence"])
    assert not report["model_execution"]["legacy_candidate_executed"]


def test_complete_audit_accepts_identical_historical_portfolios(tmp_path: Path) -> None:
    inputs = _complete_fixture(tmp_path)

    report = audit_refresh_replay(inputs, tmp_path / "audit")

    assert report["status"] == "identical_historical_portfolios"
    assert report["promotion_allowed_by_this_gate"]
    assert report["portfolio_attribution"]["portfolio_drift_rows"] == 0


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
