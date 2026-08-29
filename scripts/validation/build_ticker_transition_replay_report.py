#!/usr/bin/env python3
"""Build a hashed offline report for one ticker-transition replay."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

from alpharank.replay.ticker_transition_report import (
    TickerTransitionReplayInputs,
    build_ticker_transition_replay_report,
    write_report_bundle,
)
from alpharank.reporting.ticker_transition_replay_html import (
    write_ticker_transition_replay_html,
)


def parse_args() -> argparse.Namespace:
    """Parse immutable baseline, candidate and focus inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-legacy-run", type=Path, required=True)
    parser.add_argument("--candidate-legacy-run", type=Path, required=True)
    parser.add_argument("--baseline-boosting-run", type=Path, required=True)
    parser.add_argument("--candidate-boosting-run", type=Path, required=True)
    parser.add_argument("--baseline-trend-run", type=Path, required=True)
    parser.add_argument("--candidate-common-run", type=Path, required=True)
    parser.add_argument("--candidate-trend-run", type=Path, required=True)
    parser.add_argument("--target-ticker", required=True)
    parser.add_argument("--provider-ticker", required=True)
    parser.add_argument("--focus-decision-month", type=date.fromisoformat, required=True)
    parser.add_argument("--expected-causal-rank", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Build the evidence payload, HTML view and hash manifest."""

    args = parse_args()
    inputs = TickerTransitionReplayInputs(
        baseline_legacy_run=args.baseline_legacy_run,
        candidate_legacy_run=args.candidate_legacy_run,
        baseline_boosting_run=args.baseline_boosting_run,
        candidate_boosting_run=args.candidate_boosting_run,
        baseline_trend_run=args.baseline_trend_run,
        candidate_common_run=args.candidate_common_run,
        candidate_trend_run=args.candidate_trend_run,
        target_ticker=args.target_ticker,
        provider_ticker=args.provider_ticker,
        focus_decision_month=args.focus_decision_month,
        expected_causal_rank=args.expected_causal_rank,
        generated_at_utc=datetime.now(timezone.utc),
    )
    report = build_ticker_transition_replay_report(inputs)
    manifest_path = write_report_bundle(
        report,
        output_json=args.output_json,
        output_html=args.output_html,
        html_writer=write_ticker_transition_replay_html,
    )
    print(f"Report JSON: {args.output_json.resolve()}")
    print(f"Human report: {args.output_html.resolve()}")
    print(f"Hash manifest: {manifest_path.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
