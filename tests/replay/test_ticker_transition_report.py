from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.replay.ticker_transition_report import (
    TickerTransitionReplayInputs,
    build_ticker_transition_replay_report,
    write_report_bundle,
)
from alpharank.reporting.ticker_transition_replay_html import (
    write_ticker_transition_replay_html,
)


def test_ticker_transition_report_attributes_return_without_signal_drift(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)

    report = build_ticker_transition_replay_report(inputs)

    assert report["status"] == "passed"
    assert report["prices"]["added_rows"] == 2
    assert report["prices"]["changed_common_rows"] == 0
    assert report["predictions"]["changed_common_scores"] == 0
    assert report["predictions"]["candidate_causal_rank"]["rank"] == 14
    assert report["legacy"]["only_candidate_holdings"] == 0
    assert len(report["portfolios"]["focus_month"]) == 2


def test_ticker_transition_report_bundle_is_offline_and_hashed(tmp_path: Path) -> None:
    report = build_ticker_transition_replay_report(_fixture(tmp_path))
    output_json = tmp_path / "report.json"
    output_html = tmp_path / "report.html"

    manifest_path = write_report_bundle(
        report,
        output_json=output_json,
        output_html=output_html,
        html_writer=write_ticker_transition_replay_html,
    )

    html = output_html.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "SATS.US avait bien un prix en mai" in html
    assert "Top‑15 et Top‑20 terminent désormais le replay" in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
    assert manifest["report_json"]["sha256"] == _sha256(output_json)
    assert manifest["report_html"]["sha256"] == _sha256(output_html)


def _fixture(root: Path) -> TickerTransitionReplayInputs:
    baseline_legacy = root / "baseline_legacy"
    candidate_legacy = root / "candidate_legacy"
    baseline_boosting = root / "baseline_boosting"
    candidate_boosting = root / "candidate_boosting"
    baseline_trend = root / "baseline_trend"
    candidate_common = root / "candidate_common"
    candidate_trend = root / "candidate_trend"
    _write_legacy_runs(baseline_legacy, candidate_legacy)
    _write_boosting_runs(baseline_boosting, candidate_boosting)
    _write_replay_runs(baseline_trend, candidate_common, candidate_trend)
    return TickerTransitionReplayInputs(
        baseline_legacy_run=baseline_legacy,
        candidate_legacy_run=candidate_legacy,
        baseline_boosting_run=baseline_boosting,
        candidate_boosting_run=candidate_boosting,
        baseline_trend_run=baseline_trend,
        candidate_common_run=candidate_common,
        candidate_trend_run=candidate_trend,
        target_ticker="SATS.US",
        provider_ticker="ECHO.US",
        focus_decision_month=date(2026, 4, 1),
        expected_causal_rank=14,
        generated_at_utc=datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc),
    )


def _write_legacy_runs(baseline: Path, candidate: Path) -> None:
    for path in (baseline, candidate):
        (path / "input_snapshot").mkdir(parents=True)
        _holdings().write_parquet(path / "legacy_common_holdings.parquet")
        _performance(("Combined_Equal",)).write_csv(path / "legacy_common_performance.csv")
    _prices(("2026-04-24",)).write_parquet(baseline / "input_snapshot/US_Finalprice.parquet")
    _prices(("2026-04-24", "2026-04-27", "2026-04-28")).write_parquet(
        candidate / "input_snapshot/US_Finalprice.parquet"
    )
    lineage = candidate / "input_snapshot/lineage/prices"
    lineage.mkdir(parents=True)
    _transition_audit().write_parquet(lineage / "price_ticker_transition_audit.parquet")
    (lineage / "price_ticker_transition_policy.json").write_text(
        json.dumps({"policy_id": "price_ticker_transition_return_overlay_v1"}),
        encoding="utf-8",
    )
    (candidate / "input_snapshot/snapshot_manifest.json").write_text(
        json.dumps(
            {
                "composition_id": "fixture-composition",
                "generated_at": "2026-08-29T12:00:00+00:00",
                "output_sha256": {"US_Finalprice.parquet": "fixture-price-hash"},
                "data_freshness": {"prices": {"max_market_date": "2026-04-28"}},
            }
        ),
        encoding="utf-8",
    )


def _write_boosting_runs(baseline: Path, candidate: Path) -> None:
    for path, is_candidate in ((baseline, False), (candidate, True)):
        output = path / "classification_h06"
        output.mkdir(parents=True)
        _predictions(is_candidate=is_candidate).write_parquet(output / "predictions.parquet")


def _write_replay_runs(baseline: Path, common: Path, trend: Path) -> None:
    for path in (baseline, common, trend):
        path.mkdir(parents=True)
    registry = _trend_registry()
    registry.write_parquet(baseline / "causal_trend_eligibility.parquet")
    registry.write_parquet(trend / "causal_trend_eligibility.parquet")
    shared = ("Boosting Top 5 | Causal trend", "Boosting Top 10 | Causal trend")
    _performance(shared).write_csv(baseline / "comparison_common_performance.csv")
    _performance(("Boosting Top 15", "Boosting Top 20")).write_csv(
        common / "comparison_common_performance.csv"
    )
    strategies = (*shared, "Boosting Top 15 | Causal trend", "Boosting Top 20 | Causal trend")
    _performance(strategies).write_csv(trend / "comparison_common_performance.csv")
    _monthly().write_csv(trend / "comparison_common_monthly.csv")
    (common / "manifest.json").write_text(_manifest(publication_eligible=True), encoding="utf-8")
    (trend / "manifest.json").write_text(_manifest(publication_eligible=False), encoding="utf-8")


def _prices(dates: tuple[str, ...]) -> pl.DataFrame:
    count = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "open": [100.0 + index for index in range(count)],
            "high": [101.0 + index for index in range(count)],
            "low": [99.0 + index for index in range(count)],
            "close": [100.5 + index for index in range(count)],
            "volume": [1_000_000.0] * count,
            "adjusted_close": [100.5 + index for index in range(count)],
            "ticker": ["SATS.US"] * count,
        }
    )


def _transition_audit() -> pl.DataFrame:
    dates = ("2026-04-27", "2026-04-28")
    return pl.DataFrame(
        {
            "target_ticker": ["SATS.US"] * 2,
            "provider_ticker": ["ECHO.US"] * 2,
            "date": dates,
            "provider_daily_return": [0.01, 0.02],
            "provider_adjusted_close": [101.0, 103.02],
            "selected_adjusted_close": [101.5, 103.53],
            "return_source_vintage_id": ["fixture-vintage"] * 2,
            "evidence_url": ["https://example.test/evidence"] * 2,
        }
    )


def _predictions(*, is_candidate: bool) -> pl.DataFrame:
    tickers = [f"T{index:02d}.US" for index in range(1, 14)] + ["SATS.US"]
    scores = [0.30 - index * 0.01 for index in range(13)] + [0.10]
    return pl.DataFrame(
        {
            "decision_month": [date(2026, 4, 1)] * 14,
            "ticker": tickers,
            "fold": [15] * 14,
            "method": ["classification"] * 14,
            "horizon": [6] * 14,
            "score": scores,
            "calibrated_probability": scores,
            "future_return_1m": [0.02] * 13 + ([0.049131] if is_candidate else [0.0]),
            "benchmark_future_return_1m": [0.052626] * 14,
            "future_excess_return_1m": [-0.032626] * 13
            + ([-0.003495] if is_candidate else [-0.052626]),
            "target_status_1m": ["evaluable"] * 13
            + (["evaluable"] if is_candidate else ["approved_censored_last_observation"]),
        }
    )


def _trend_registry() -> pl.DataFrame:
    tickers = [f"T{index:02d}.US" for index in range(1, 14)] + ["SATS.US"]
    return pl.DataFrame(
        {
            "decision_month": [date(2026, 4, 1)] * 14,
            "ticker": tickers,
            "fold": [15] * 14,
            "trend_eligible": [True] * 14,
            "trend_positive_pair_fraction": [1.0] * 14,
        }
    )


def _holdings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "holding_month": [date(2026, 5, 1)],
            "ticker": ["AAA.US"],
            "strategy": ["Legacy"],
            "realized_return": [0.03],
            "target_weight": [1.0],
        }
    )


def _performance(strategies: tuple[str, ...]) -> pl.DataFrame:
    count = len(strategies)
    return pl.DataFrame(
        {
            "strategy": strategies,
            "start_holding_month": ["2011-08-01"] * count,
            "end_holding_month": ["2026-07-01"] * count,
            "months": [180] * count,
            "total_return": [10.0] * count,
            "cagr": [0.18] * count,
            "sharpe": [0.70] * count,
            "max_drawdown": [-0.25] * count,
        }
    )


def _monthly() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": [
                "Boosting Top 15 | Causal trend",
                "Boosting Top 20 | Causal trend",
            ],
            "decision_month": ["2026-04-01"] * 2,
            "holding_month": ["2026-05-01"] * 2,
            "gross_return": [0.18, 0.14],
            "turnover": [0.4, 0.45],
            "transaction_cost": [0.0004, 0.00045],
            "net_return": [0.1796, 0.13955],
            "benchmark_return": [0.052626] * 2,
            "active_return": [0.126974, 0.086924],
            "n_positions": [15, 20],
        }
    )


def _manifest(*, publication_eligible: bool) -> str:
    return json.dumps(
        {
            "comparison_eligible": True,
            "publication_eligible": publication_eligible,
            "methodology_status": "post_hoc_research_diagnostic",
            "lineage_check": {"passed": True},
            "runtime_provenance": {"git": {"head": "fixture-head", "dirty": False}},
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
