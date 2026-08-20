"""Strict validation for causal Legacy-v2 replay artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.portfolio.contracts import validate_causal_timing, validate_holdings
from alpharank.portfolio.costs import TransactionCostModel
from alpharank.portfolio.simulation import simulate_weighted_portfolio

LEGACY_V2_TOLERANCE = 1e-12
HOLDING_MONTH_MEMBERSHIP_POLICY_ID = "holding_month_membership_before_ranking_v1"


def require_holding_month_membership(
    signal_candidates: pl.DataFrame,
    monthly_membership: pl.DataFrame,
) -> pl.DataFrame:
    """Keep only candidates that belong to the index for the holding month.

    Legacy candidate features are dated in the signal month and shifted one
    month by ``StrategyLearner`` before ranking.  The monthly constituent file
    is effective at the start of its named month, so the execution universe
    must be joined to signal rows with a one-month offset before that shift.
    This gate uses membership only; it never observes holding-period prices or
    returns.
    """

    required_candidates = {"year_month", "ticker"}
    required_membership = {"year_month", "ticker"}
    missing_candidates = sorted(required_candidates - set(signal_candidates.columns))
    missing_membership = sorted(required_membership - set(monthly_membership.columns))
    if missing_candidates:
        raise ValueError(
            f"Legacy v2 candidates lack membership keys: {missing_candidates}"
        )
    if missing_membership:
        raise ValueError(
            f"Legacy v2 membership lacks required keys: {missing_membership}"
        )
    execution_membership = (
        monthly_membership.select(
            pl.col("ticker").cast(pl.String),
            pl.col("year_month")
            .cast(pl.Date, strict=False)
            .dt.offset_by("-1mo")
            .alias("year_month"),
        )
        .unique()
    )
    return (
        signal_candidates.with_columns(
            pl.col("year_month").cast(pl.Date, strict=False),
            pl.col("ticker").cast(pl.String),
        )
        .join(
            execution_membership,
            on=["year_month", "ticker"],
            how="inner",
            validate="m:1",
        )
        .sort(["year_month", "ticker"])
    )


def validate_legacy_v2_replay(
    run_dir: Path,
    *,
    expected_composition_id: str,
    require_clean_runtime: bool = True,
    tolerance: float = LEGACY_V2_TOLERANCE,
) -> dict[str, Any]:
    """Recalculate every saved Legacy-v2 cost scenario and fail on drift."""

    root = run_dir.resolve()
    data_manifest = _read_json(root / "data_input_manifest.json")
    replay_manifest = _read_json(root / "legacy_v2_replay_manifest.json")
    identity = data_manifest.get("run_config", {}).get("methodology_identity", {})
    if identity.get("methodology_version") != "v2-causal":
        raise RuntimeError("Legacy run is not bound to methodology v2-causal")
    if identity.get("composition_id") != expected_composition_id:
        raise RuntimeError("Legacy run composition differs from the causal snapshot")
    run_config = data_manifest.get("run_config", {})
    if run_config.get("n_trials") != 30 or run_config.get("n_jobs") != 1:
        raise RuntimeError("Legacy v2 production search must use 30 trials and n_jobs=1")
    if run_config.get("price_eligibility_policy_id") != "monthly_price_eligibility_v1":
        raise RuntimeError("Legacy v2 price eligibility policy drifted")
    runtime_git = data_manifest.get("runtime_provenance", {}).get("git", {})
    if require_clean_runtime and runtime_git.get("dirty") is not False:
        raise RuntimeError("Legacy v2 promotion run must come from a clean worktree")

    if replay_manifest.get("scope") != "alpharank_legacy_v2_replay":
        raise RuntimeError("Invalid Legacy v2 replay scope")
    if replay_manifest.get("execution_policy", {}).get("identifier") != "next_session_open_v1":
        raise RuntimeError("Legacy v2 does not use next_session_open_v1")
    if replay_manifest.get("missing_return_policy") != "raise":
        raise RuntimeError("Legacy v2 missing returns do not fail closed")
    if replay_manifest.get("canonical_cost_scenario_id") != "standard_10bps":
        raise RuntimeError("Legacy v2 canonical cost scenario drifted")
    if (
        replay_manifest.get("candidate_membership_policy", {}).get("policy_id")
        != HOLDING_MONTH_MEMBERSHIP_POLICY_ID
    ):
        raise RuntimeError("Legacy v2 holding-month membership gate drifted")

    holdings_path = Path(replay_manifest["artifacts"]["holdings"]["path"])
    monthly_path = Path(replay_manifest["artifacts"]["monthly"]["path"])
    for label, path in (("holdings", holdings_path), ("monthly", monthly_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Legacy v2 {label} artifact is missing: {path}")
        expected = replay_manifest["artifacts"][label]["sha256"]
        if _sha256(path) != expected:
            raise RuntimeError(f"Legacy v2 {label} artifact hash mismatch")

    holdings = pl.read_parquet(holdings_path)
    monthly = pl.read_parquet(monthly_path)
    validate_holdings(holdings)
    validate_causal_timing(holdings)
    if holdings.filter(pl.col("realized_return").is_null()).height:
        raise RuntimeError("Legacy v2 contains selected holdings without returns")
    if holdings["execution_policy_id"].unique().to_list() != ["next_session_open_v1"]:
        raise RuntimeError("Legacy v2 holdings contain mixed execution policies")

    scenario_reports: list[dict[str, Any]] = []
    rebuilt_parts: list[pl.DataFrame] = []
    for raw_model in replay_manifest.get("cost_scenarios", []):
        model = TransactionCostModel(**raw_model)
        for strategy_holdings in holdings.partition_by("strategy", maintain_order=True):
            rebuilt_parts.append(
                simulate_weighted_portfolio(
                    strategy_holdings,
                    transaction_cost_model=model,
                    missing_return_policy="raise",
                    causal_timing_policy="require_explicit",
                )
            )
    if not rebuilt_parts:
        raise RuntimeError("Legacy v2 has no declared cost scenarios")
    rebuilt = pl.concat(rebuilt_parts, how="diagonal_relaxed")
    keys = ["strategy", "decision_month", "holding_month", "cost_scenario_id"]
    numeric = [
        "gross_return",
        "turnover",
        "spread_cost",
        "slippage_cost",
        "impact_cost",
        "commission_cost",
        "fx_cost",
        "transaction_cost",
        "net_return",
        "benchmark_return",
    ]
    joined = rebuilt.join(
        monthly.select(*keys, *[pl.col(column).alias(f"saved_{column}") for column in numeric]),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != rebuilt.height or joined.height != monthly.height:
        raise RuntimeError("Legacy v2 saved and rebuilt monthly calendars differ")
    for column in numeric:
        maximum_error = float(
            joined.select((pl.col(column) - pl.col(f"saved_{column}")).abs().max()).item()
        )
        scenario_reports.append(
            {"field": column, "maximum_absolute_error": maximum_error}
        )
        if maximum_error > tolerance:
            raise RuntimeError(f"Legacy v2 replay mismatch for {column}: {maximum_error}")

    costs = (
        monthly.group_by("cost_scenario_id")
        .agg(pl.col("transaction_cost").sum().alias("total_cost"))
        .sort("total_cost")
    )
    if costs["cost_scenario_id"].to_list() != [
        "zero",
        "standard_10bps",
        "stress_30bps",
    ]:
        raise RuntimeError("Legacy v2 cost sensitivity is absent or non-monotonic")
    first_selection = holdings.sort(
        ["strategy", "decision_month", "ticker"]
    ).group_by("strategy", maintain_order=True).first()
    return {
        "passed": True,
        "composition_id": expected_composition_id,
        "holdings_rows": holdings.height,
        "monthly_rows": monthly.height,
        "strategy_count": holdings["strategy"].n_unique(),
        "scenario_count": costs.height,
        "first_selection_rows": first_selection.height,
        "maximum_absolute_replay_error": max(
            row["maximum_absolute_error"] for row in scenario_reports
        ),
        "holdings_sha256": _sha256(holdings_path),
        "monthly_sha256": _sha256(monthly_path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
