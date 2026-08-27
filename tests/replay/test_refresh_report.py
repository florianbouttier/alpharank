from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl

from alpharank.replay.refresh_attribution import (
    RefreshAttributionInputs,
    ScenarioArtifacts,
    build_refresh_attribution,
)
from alpharank.reporting.refresh_replay_html import write_refresh_replay_html


def test_refresh_report_separates_price_and_sec_effects(tmp_path: Path) -> None:
    scenarios, audit_path = _report_fixture(tmp_path)

    report = build_refresh_attribution(
        RefreshAttributionInputs(audit_report=audit_path, scenarios=scenarios)
    )

    headline = report["headline"]
    assert isinstance(headline, dict)
    assert headline["legacy_price_only_events"] == 0
    assert headline["legacy_sec_only_events"] == headline["legacy_full_changed_common"] == 2
    assert headline["boosting_price_score_changed"] == 1
    assert headline["boosting_sec_score_changed"] == 1
    focus = report["focus"]
    assert isinstance(focus, dict)
    assert focus["price"]["strictly_identical"]
    assert focus["sec"]["baseline_rows"] == 0
    assert focus["sec"]["candidate_rows"] == 4


def test_refresh_report_html_is_offline_and_explains_cvc(tmp_path: Path) -> None:
    scenarios, audit_path = _report_fixture(tmp_path)
    report = build_refresh_attribution(
        RefreshAttributionInputs(audit_report=audit_path, scenarios=scenarios)
    )
    output = tmp_path / "refresh_replay_report.html"

    write_refresh_replay_html(json.loads(json.dumps(report, default=str)), output)

    html = output.read_text(encoding="utf-8")
    assert "CVC.US déclenche l'arrêt final" in html
    assert "Le drift Legacy est quasi entièrement SEC" in html
    assert "Prix communs modifiés" in html
    assert "https://" not in html
    assert "filterTable" in html


def _report_fixture(tmp_path: Path) -> tuple[tuple[ScenarioArtifacts, ...], Path]:
    baseline = _scenario(tmp_path, "baseline", legacy_weight=0.5, cvc_score=0.1)
    price_only = _scenario(
        tmp_path,
        "price_only",
        legacy_weight=0.5,
        cvc_score=0.1,
        other_score=0.21,
    )
    sec_only = _scenario(tmp_path, "sec_only", legacy_weight=0.7, cvc_score=0.3)
    full = _scenario(tmp_path, "full", legacy_weight=0.7, cvc_score=0.3)
    _write_snapshots(baseline.legacy_run / "input_snapshot", full.legacy_run / "input_snapshot")
    audit_path = tmp_path / "refresh_replay_report.json"
    audit_path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-08-27T12:00:00+00:00",
                "historical_cutoff": "2026-07-01",
                "status": "common_replay_blocked",
                "promotion_allowed_by_this_gate": False,
                "common_replay_failure": "CVC.US has a censored terminal return",
                "inputs": {
                    "baseline_snapshot": str(baseline.legacy_run / "input_snapshot"),
                    "candidate_snapshot": str(full.legacy_run / "input_snapshot"),
                },
                "snapshot_comparison": [
                    {
                        "table": "final_price",
                        "baseline_rows": 2,
                        "candidate_rows": 2,
                        "added_rows": 0,
                        "removed_rows": 0,
                        "changed_common_rows": 0,
                    }
                ],
                "provenance_comparison": {
                    "all_code_identical": True,
                    "all_config_identical": True,
                    "all_runtime_identical": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        baseline,
        price_only,
        replace(sec_only, common_status="bloqué sur CVC.US", common_run=None),
        replace(full, common_status="bloqué sur CVC.US", common_run=None),
    ), audit_path


def _scenario(
    root: Path,
    name: str,
    *,
    legacy_weight: float,
    cvc_score: float,
    other_score: float = 0.2,
) -> ScenarioArtifacts:
    scenario = root / name
    legacy_run = scenario / "legacy"
    boosting_run = scenario / "boosting"
    common_run = scenario / "common"
    (boosting_run / "classification_h06").mkdir(parents=True)
    legacy_run.mkdir(parents=True)
    common_run.mkdir(parents=True)
    _legacy_frame(legacy_weight).write_parquet(legacy_run / "legacy_common_holdings.parquet")
    (legacy_run / "data_input_manifest.json").write_text("{}", encoding="utf-8")
    _prediction_frame(cvc_score, other_score).write_parquet(
        boosting_run / "classification_h06" / "predictions.parquet"
    )
    _feature_manifest(cvc_score).write_csv(
        boosting_run / "classification_h06" / "fold_feature_manifest.csv"
    )
    (boosting_run / "manifest.json").write_text("{}", encoding="utf-8")
    _legacy_frame(legacy_weight).write_parquet(common_run / "comparison_common_holdings.parquet")
    (common_run / "manifest.json").write_text("{}", encoding="utf-8")
    return ScenarioArtifacts(
        name,
        name.replace("_", " "),
        legacy_run,
        boosting_run,
        "passe",
        common_run,
    )


def _legacy_frame(weight: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": ["Legacy", "Legacy"],
            "decision_month": [date(2016, 6, 1), date(2016, 6, 1)],
            "holding_month": [date(2016, 7, 1), date(2016, 7, 1)],
            "ticker": ["AAA.US", "CVC.US"],
            "target_weight": [weight, 1.0 - weight],
        }
    )


def _prediction_frame(cvc_score: float, other_score: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "method": ["classification", "classification"],
            "horizon": [6, 6],
            "decision_month": [date(2016, 6, 1), date(2016, 6, 1)],
            "ticker": ["AAA.US", "CVC.US"],
            "score": [other_score, cvc_score],
            "calibrated_probability": [other_score, cvc_score],
            "future_return_1m": [0.01, 0.0],
            "benchmark_future_return_1m": [0.01, 0.01],
            "fold": [5, 5],
        }
    )


def _feature_manifest(cvc_score: float) -> pl.DataFrame:
    pairs = "[[10, 20], [30, 40]]" if cvc_score == 0.1 else "[[10, 20], [50, 60]]"
    return pl.DataFrame(
        {
            "fold": [5],
            "train_rows": [100],
            "validation_rows": [20],
            "test_rows": [30],
            "winner_pairs": [pairs],
        }
    )


def _write_snapshots(baseline: Path, candidate: Path) -> None:
    baseline.mkdir(parents=True)
    candidate.mkdir(parents=True)
    prices = pl.DataFrame(
        {
            "ticker": ["CVC.US", "CVC.US"],
            "date": [date(2016, 6, 17), date(2016, 6, 20)],
            "adjusted_close": [31.0, 31.2],
        }
    )
    prices.write_parquet(baseline / "US_Finalprice.parquet")
    prices.write_parquet(candidate / "US_Finalprice.parquet")
    tables = (
        ("US_Income_statement.parquet", "filing_date"),
        ("US_Balance_sheet.parquet", "filing_date"),
        ("US_Cash_flow.parquet", "filing_date"),
        ("US_Earnings.parquet", "reportDate"),
    )
    for filename, filing_column in tables:
        pl.DataFrame(schema={"ticker": pl.String, filing_column: pl.String}).write_parquet(
            baseline / filename
        )
        pl.DataFrame({"ticker": ["CVC.US"], filing_column: ["2016-05-05"]}).write_parquet(
            candidate / filename
        )
