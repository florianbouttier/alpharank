"""Build and validate the causal Legacy/Boosting/SPY comparison replay."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.boosting_v2 import validate_boosting_v2_replay
from alpharank.causal_snapshot import validate_causal_v2_snapshot
from alpharank.governance import reserve_run_directory
from alpharank.legacy_v2 import (
    HOLDING_MONTH_MEMBERSHIP_POLICY_ID,
    require_holding_month_membership,
)
from alpharank.legacy_v2 import validate_legacy_v2_replay
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.attribution import provisional_return_cagr_attribution
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.contracts import validate_causal_timing, validate_holdings
from alpharank.portfolio.costs import TransactionCostModel
from alpharank.portfolio.execution import apply_next_session_open_holding_returns
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.terminal_event_registry import (
    load_terminal_event_registry,
)
from alpharank.backtest.datasets import prepare_constituents_monthly


COMMON_V2_TOLERANCE = 1e-12


def standard_v2_cost_model() -> TransactionCostModel:
    return TransactionCostModel(
        "standard_10bps",
        spread_bps=3.0,
        slippage_bps=2.0,
        impact_bps=2.0,
        commission_bps=2.0,
        fx_bps=1.0,
        fx_turnover_fraction=1.0,
    )


def gate_boosting_predictions_for_holding_membership(
    predictions: pl.DataFrame,
    membership: pl.DataFrame,
) -> pl.DataFrame:
    """Apply the same pre-ranking execution-universe gate as Legacy v2."""

    if "decision_month" not in predictions.columns:
        raise ValueError("Boosting predictions require decision_month")
    return require_holding_month_membership(
        predictions.rename({"decision_month": "year_month"}),
        membership,
    ).rename({"year_month": "decision_month"})


def gate_boosting_predictions_for_execution_open(
    predictions: pl.DataFrame,
    daily_prices: pl.DataFrame,
) -> pl.DataFrame:
    """Keep candidates executable at the first market session of holding."""

    required_prices = {"ticker", "date", "open"}
    missing = sorted(required_prices - set(daily_prices.columns))
    if missing:
        raise ValueError(f"Execution-open gate lacks price fields: {missing}")
    valid = daily_prices.select(
        pl.col("ticker").cast(pl.String),
        pl.col("date").cast(pl.Date, strict=False),
        pl.col("open").cast(pl.Float64, strict=False),
    ).filter(
        pl.col("date").is_not_null()
        & pl.col("open").is_finite()
        & (pl.col("open") > 0.0)
    ).with_columns(
        pl.col("date").dt.truncate("1mo").alias("holding_month")
    )
    market_open = valid.group_by("holding_month").agg(
        pl.col("date").min().alias("market_first_session")
    )
    executable = (
        valid.group_by("ticker", "holding_month")
        .agg(pl.col("date").min().alias("ticker_first_session"))
        .join(market_open, on="holding_month", how="inner", validate="m:1")
        .filter(pl.col("ticker_first_session") == pl.col("market_first_session"))
        .select(
            "ticker",
            pl.col("holding_month")
            .dt.offset_by("-1mo")
            .alias("decision_month"),
        )
        .unique()
    )
    return predictions.join(
        executable,
        on=["decision_month", "ticker"],
        how="inner",
        validate="m:1",
    ).sort(["decision_month", "ticker"])


def gate_boosting_predictions_for_pre_execution_blocks(
    predictions: pl.DataFrame,
    pre_execution_blocks: pl.DataFrame,
) -> pl.DataFrame:
    """Reject publicly suspended candidates before Top-N ranking."""

    required_predictions = {"decision_month", "ticker"}
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    if missing_predictions:
        raise ValueError(
            f"Pre-execution gate lacks prediction fields: {missing_predictions}"
        )
    if pre_execution_blocks.is_empty():
        return predictions.sort(["decision_month", "ticker"])
    required_blocks = {
        "terminal_event_id",
        "ticker",
        "effective_date",
        "known_at",
        "entry_allowed",
    }
    missing_blocks = sorted(required_blocks - set(pre_execution_blocks.columns))
    if missing_blocks:
        raise ValueError(f"Pre-execution gate lacks event fields: {missing_blocks}")
    normalized_blocks = pre_execution_blocks.select(
        pl.col("terminal_event_id").cast(pl.String),
        pl.col("ticker").cast(pl.String),
        pl.col("effective_date").cast(pl.String).str.to_date(strict=False),
        pl.col("known_at"),
        pl.col("entry_allowed").cast(pl.Boolean, strict=False),
    )
    if normalized_blocks.filter(
        pl.col("effective_date").is_null()
        | pl.col("known_at").is_null()
        | (pl.col("entry_allowed") != False)  # noqa: E712
    ).height:
        raise ValueError("Pre-execution blocks must be dated, sourced rejections.")
    blocked_keys = normalized_blocks.select(
        "ticker",
        pl.col("effective_date")
        .dt.truncate("1mo")
        .dt.offset_by("-1mo")
        .alias("decision_month"),
    ).unique()
    normalized_predictions = predictions.with_columns(
        pl.col("decision_month").cast(pl.String).str.to_date(strict=False)
    )
    if normalized_predictions.select(pl.col("decision_month").is_null().any()).item():
        raise ValueError("Pre-execution gate requires valid decision months.")
    return normalized_predictions.join(
        blocked_keys,
        on=["decision_month", "ticker"],
        how="anti",
    ).sort(["decision_month", "ticker"])


def build_common_v2_comparison(
    *,
    legacy_run_dir: Path,
    boosting_run_dir: Path,
    causal_snapshot_dir: Path,
    output_dir: Path,
    top_n_values: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Resimulate every investable strategy on one next-open/cost contract."""

    destination = reserve_run_directory(output_dir)
    causal = validate_causal_v2_snapshot(causal_snapshot_dir)
    composition_id = causal["composition_id"]
    legacy_validation = validate_legacy_v2_replay(
        legacy_run_dir, expected_composition_id=composition_id
    )
    boosting_validation = validate_boosting_v2_replay(
        boosting_run_dir, expected_composition_id=composition_id
    )
    snapshot = causal_snapshot_dir / "input_snapshot"
    prices = pl.read_parquet(snapshot / "US_Finalprice.parquet")
    terminal_registry = load_terminal_event_registry()
    legacy_manifest = _read_json(legacy_run_dir / "legacy_v2_replay_manifest.json")
    legacy_holdings = pl.read_parquet(
        Path(legacy_manifest["artifacts"]["holdings"]["path"])
    ).filter(pl.col("strategy") == "Combined_Frequency").with_columns(
        pl.lit("Legacy").alias("strategy")
    )
    benchmark_by_month = legacy_holdings.select(
        "holding_month", "benchmark_return"
    ).unique()
    legacy_months = legacy_holdings.select("holding_month").unique()
    execution_columns = {
        "feature_max_asof_at",
        "signal_cutoff_at",
        "execution_at",
        "first_return_observation_at",
        "holding_return_end_at",
        "scheduled_holding_end_at",
        "holding_observation_gap_calendar_days",
        "execution_policy_id",
        "return_resolution",
        "return_resolution_reason",
        "manual_review_status",
    }
    legacy_holdings = apply_next_session_open_holding_returns(
        legacy_holdings.drop(
            *[column for column in execution_columns if column in legacy_holdings.columns]
        ),
        prices.select(["ticker", "date", "open", "close", "adjusted_close"]),
    )

    predictions = pl.read_parquet(
        boosting_run_dir / "classification_h06" / "predictions.parquet"
    )
    monthly_membership = prepare_constituents_monthly(
        pl.read_csv(snapshot / "SP500_Constituents.csv")
    ).select("year_month", "ticker")
    predictions_before_membership_gate = predictions.height
    predictions = gate_boosting_predictions_for_holding_membership(
        predictions,
        monthly_membership,
    )
    predictions_after_membership_gate = predictions.height
    membership_rows_removed = (
        predictions_before_membership_gate - predictions_after_membership_gate
    )
    predictions_before_execution_gate = predictions.height
    predictions = gate_boosting_predictions_for_execution_open(
        predictions,
        prices.select(["ticker", "date", "open"]),
    )
    predictions_after_execution_gate = predictions.height
    execution_rows_removed = (
        predictions_before_execution_gate - predictions_after_execution_gate
    )
    predictions_before_pre_execution_block_gate = predictions.height
    predictions = gate_boosting_predictions_for_pre_execution_blocks(
        predictions,
        terminal_registry.pre_execution_blocks(),
    )
    predictions_after_pre_execution_block_gate = predictions.height
    pre_execution_block_rows_removed = (
        predictions_before_pre_execution_block_gate
        - predictions_after_pre_execution_block_gate
    )
    complete_decisions = (
        predictions.group_by("decision_month")
        .agg(
            pl.col("benchmark_future_return_1m")
            .is_not_null()
            .all()
            .alias("benchmark_month_complete")
        )
        .filter(pl.col("benchmark_month_complete"))
        .select("decision_month")
    )
    predictions = predictions.join(
        complete_decisions, on="decision_month", how="inner"
    )
    holdings_parts: list[pl.DataFrame] = []
    price_columns = ["ticker", "date", "open", "close", "adjusted_close"]
    for top_n in top_n_values:
        raw_holdings = boosting_predictions_to_holdings(
            predictions,
            strategy=f"Boosting Top {top_n}",
            top_n=top_n,
        ).join(legacy_months, on="holding_month", how="inner")
        causal_holdings = (
            apply_next_session_open_holding_returns(
                raw_holdings,
                prices.select(price_columns),
            )
            .drop("benchmark_return")
            .join(
                benchmark_by_month,
                on="holding_month",
                how="left",
                validate="m:1",
            )
        )
        holdings_parts.append(causal_holdings)
    boosting_holdings = pl.concat(holdings_parts, how="diagonal_relaxed")
    common_months = boosting_holdings.select("holding_month").unique()
    legacy_holdings = legacy_holdings.join(
        common_months, on="holding_month", how="inner"
    )
    holdings = pl.concat(
        [legacy_holdings, boosting_holdings], how="diagonal_relaxed"
    )
    cost_model = standard_v2_cost_model()
    monthly_parts = [
        simulate_weighted_portfolio(
            strategy_holdings,
            transaction_cost_model=cost_model,
            missing_return_policy="raise",
            causal_timing_policy="require_explicit",
        )
        for strategy_holdings in holdings.partition_by(
            "strategy", maintain_order=True
        )
    ]
    investable_monthly = pl.concat(monthly_parts, how="diagonal_relaxed")
    provisional_journal = holdings.filter(
        pl.col("return_resolution") == "provisional_last_observation"
    ).select(
        "strategy",
        "decision_month",
        "holding_month",
        "ticker",
        "selection_rank",
        "target_weight",
        "realized_return",
        "execution_at",
        "holding_return_end_at",
        "scheduled_holding_end_at",
        "holding_observation_gap_calendar_days",
        "return_resolution",
        "return_resolution_reason",
        "manual_review_status",
    ).sort(["holding_month", "strategy", "selection_rank"])
    provisional_journal_path = destination / "provisional_holding_journal.parquet"
    provisional_journal_csv_path = destination / "provisional_holding_journal.csv"
    provisional_journal.write_parquet(provisional_journal_path)
    provisional_journal.write_csv(provisional_journal_csv_path)
    provisional_attribution = provisional_return_cagr_attribution(
        holdings,
        investable_monthly,
    )
    provisional_attribution_path = destination / "provisional_cagr_attribution.csv"
    provisional_attribution.write_csv(provisional_attribution_path)
    provisional_report_path = destination / "provisional_terminal_impact_report.md"
    provisional_report_path.write_text(
        _render_provisional_terminal_report(
            provisional_attribution,
            provisional_journal,
        ),
        encoding="utf-8",
    )
    legacy_monthly = investable_monthly.filter(pl.col("strategy") == "Legacy")
    spy = reference_monthly_series(
        legacy_monthly,
        strategy="SPY total return",
        return_column="benchmark_return",
    )
    monthly = pl.concat([investable_monthly, spy], how="diagonal_relaxed")
    artifacts = write_common_portfolio_artifacts(
        output_dir=destination,
        holdings=holdings,
        monthly_returns=monthly,
        prefix="common_v2",
        benchmark_metadata={
            "id": "spy_next_session_open_total_return_v1",
            "label": "SPY total return",
            "price_column": "adjusted_open_to_adjusted_close",
            "includes_distributions": True,
        },
    )
    calendar = (
        monthly.group_by("strategy")
        .agg(
            pl.col("holding_month").min().alias("start"),
            pl.col("holding_month").max().alias("end"),
            pl.len().alias("months"),
        )
        .sort("strategy")
    )
    if calendar["months"].n_unique() != 1:
        raise RuntimeError("Common v2 strategies do not share one calendar")
    manifest = {
        "contract_version": 1,
        "scope": "alpharank_common_v2_replay",
        "status": (
            "provisional_manual_terminal_review"
            if provisional_journal.height
            else "canonical_common_strategy_replay"
        ),
        "comparison_eligible": provisional_journal.is_empty(),
        "methodology_version": "v2-causal",
        "composition_id": composition_id,
        "execution_policy_id": "next_session_open_v1",
        "missing_return_policy": "raise",
        "transaction_cost_model": asdict(cost_model),
        "benchmark_cost_policy": "no simulated trading cost",
        "top_n_values": list(top_n_values),
        "holding_month_membership_gate": {
            "policy_id": HOLDING_MONTH_MEMBERSHIP_POLICY_ID,
            "uses_holding_prices_or_returns": False,
            "candidate_rows_before": predictions_before_membership_gate,
            "candidate_rows_after": predictions_after_membership_gate,
            "candidate_rows_removed": membership_rows_removed,
        },
        "first_session_execution_gate": {
            "policy_id": "first_holding_session_open_before_ranking_v1",
            "uses_holding_return_or_month_end_price": False,
            "candidate_rows_before": predictions_before_execution_gate,
            "candidate_rows_after": predictions_after_execution_gate,
            "candidate_rows_removed": execution_rows_removed,
        },
        "reviewed_pre_execution_block_gate": {
            "policy_id": "reviewed_pre_execution_block_before_ranking_v1",
            "uses_holding_return_or_month_end_price": False,
            "terminal_event_registry_sha256": terminal_registry.sha256,
            "candidate_rows_before": predictions_before_pre_execution_block_gate,
            "candidate_rows_after": predictions_after_pre_execution_block_gate,
            "candidate_rows_removed": pre_execution_block_rows_removed,
        },
        "calendar": calendar.to_dicts(),
        "provisional_terminal_observations": {
            "policy_id": "provisional_last_observation_v1",
            "holding_rows": provisional_journal.height,
            "tickers": provisional_journal["ticker"].n_unique(),
            "manual_review_status": (
                "pending_manual_terminal_event_review"
                if provisional_journal.height
                else "not_applicable"
            ),
        },
        "source_validation": {
            "causal_snapshot": causal,
            "legacy": legacy_validation,
            "boosting": boosting_validation,
        },
        "sources": {
            "legacy_manifest": _file_record(
                legacy_run_dir / "legacy_v2_replay_manifest.json"
            ),
            "boosting_manifest": _file_record(boosting_run_dir / "manifest.json"),
            "terminal_event_registry": _file_record(terminal_registry.path),
        },
        "artifacts": {
            label: _file_record(path)
            for label, path in {
                **artifacts,
                "provisional_holding_journal": provisional_journal_path,
                "provisional_holding_journal_csv": provisional_journal_csv_path,
                "provisional_cagr_attribution": provisional_attribution_path,
                "provisional_terminal_impact_report": provisional_report_path,
            }.items()
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return validate_common_v2_replay(
        destination,
        expected_composition_id=composition_id,
        allow_provisional=not provisional_journal.is_empty(),
    )


def validate_common_v2_replay(
    output_dir: Path,
    *,
    expected_composition_id: str,
    tolerance: float = COMMON_V2_TOLERANCE,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    """Recalculate monthly rows and require one exact common calendar."""

    root = output_dir.resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("scope") != "alpharank_common_v2_replay":
        raise RuntimeError("Invalid common v2 replay scope")
    comparison_eligible = manifest.get("comparison_eligible") is True
    if not comparison_eligible and not allow_provisional:
        raise RuntimeError("Common v2 replay is not comparison eligible")
    if not comparison_eligible and manifest.get("status") != (
        "provisional_manual_terminal_review"
    ):
        raise RuntimeError("Common v2 replay has an unsupported provisional status")
    if manifest.get("composition_id") != expected_composition_id:
        raise RuntimeError("Common v2 composition differs from the causal snapshot")
    if manifest.get("execution_policy_id") != "next_session_open_v1":
        raise RuntimeError("Common v2 execution policy drifted")
    if manifest.get("missing_return_policy") != "raise":
        raise RuntimeError("Common v2 missing returns do not fail closed")
    for source_report in manifest.get("source_validation", {}).values():
        if source_report.get("passed") is not True:
            raise RuntimeError("A common v2 source validation did not pass")
    for label, record in manifest.get("artifacts", {}).items():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Common v2 artifact hash mismatch: {label}")

    holdings_path = Path(manifest["artifacts"]["holdings"]["path"])
    monthly_path = Path(manifest["artifacts"]["monthly_parquet"]["path"])
    holdings = pl.read_parquet(holdings_path)
    monthly = pl.read_parquet(monthly_path)
    validate_holdings(holdings)
    validate_causal_timing(holdings)
    cost_model = TransactionCostModel(**manifest["transaction_cost_model"])
    rebuilt_parts = [
        simulate_weighted_portfolio(
            frame,
            transaction_cost_model=cost_model,
            missing_return_policy="raise",
            causal_timing_policy="require_explicit",
        )
        for frame in holdings.partition_by("strategy", maintain_order=True)
    ]
    rebuilt = pl.concat(rebuilt_parts)
    legacy = rebuilt.filter(pl.col("strategy") == "Legacy")
    rebuilt = pl.concat(
        [
            rebuilt,
            reference_monthly_series(
                legacy,
                strategy="SPY total return",
                return_column="benchmark_return",
            ),
        ],
        how="diagonal_relaxed",
    )
    keys = ["strategy", "decision_month", "holding_month"]
    numeric = [
        "gross_return",
        "turnover",
        "transaction_cost",
        "net_return",
        "benchmark_return",
    ]
    joined = rebuilt.join(
        monthly.select(
            *keys,
            *[pl.col(column).alias(f"saved_{column}") for column in numeric],
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != rebuilt.height or joined.height != monthly.height:
        raise RuntimeError("Common v2 monthly calendar is not exactly reproducible")
    errors = {
        column: float(
            joined.select(
                (pl.col(column) - pl.col(f"saved_{column}")).abs().max()
            ).item()
        )
        for column in numeric
    }
    if max(errors.values()) > tolerance:
        raise RuntimeError(f"Common v2 monthly reconciliation failed: {errors}")
    calendar_counts = monthly.group_by("strategy").len()
    if calendar_counts["len"].n_unique() != 1:
        raise RuntimeError("Common v2 strategies have different month counts")
    month_sets = [
        set(frame["holding_month"].to_list())
        for frame in monthly.partition_by("strategy")
    ]
    if any(months != month_sets[0] for months in month_sets[1:]):
        raise RuntimeError("Common v2 strategies have different holding calendars")
    return {
        "passed": True,
        "comparison_eligible": comparison_eligible,
        "status": manifest.get("status", "canonical_common_strategy_replay"),
        "provisional_holding_rows": manifest.get(
            "provisional_terminal_observations", {}
        ).get("holding_rows", 0),
        "composition_id": expected_composition_id,
        "strategy_count": monthly["strategy"].n_unique(),
        "months_per_strategy": int(calendar_counts["len"][0]),
        "holdings_rows": holdings.height,
        "monthly_rows": monthly.height,
        "maximum_absolute_reconciliation_error": max(errors.values()),
        "holdings_sha256": _sha256(holdings_path),
        "monthly_sha256": _sha256(monthly_path),
    }


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _render_provisional_terminal_report(
    attribution: pl.DataFrame,
    journal: pl.DataFrame,
) -> str:
    lines = [
        "# Impact des fins de cotation provisoires",
        "",
        "Les positions ci-dessous restent dans le portefeuille. Leur rendement est "
        "mesuré de la première ouverture du mois à la dernière cotation observée, "
        "puis la valeur est portée à rendement nul jusqu'à la fin prévue. Chaque "
        "cas reste en revue manuelle jusqu'à résolution de l'événement terminal.",
        "",
        "Les contributions en log annualisé sont additives. L'impact marginal sur "
        "le CAGR est exact pour ce groupe, mais ne doit pas être additionné à "
        "d'autres impacts marginaux indépendants.",
        "",
        "| Stratégie | CAGR provisoire | Contribution log ann. | CAGR sans ce groupe | Impact CAGR marginal | Positions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in attribution.to_dicts():
        lines.append(
            f"| {row['strategy']} | {100 * row['cagr']:.4f} % | "
            f"{100 * row['provisional_annualized_log_contribution']:.4f} pt | "
            f"{100 * row['cagr_without_provisional_component']:.4f} % | "
            f"{100 * row['provisional_marginal_cagr_impact']:.4f} pt | "
            f"{row['provisional_holding_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Journal à résoudre",
            "",
            "| Stratégie | Achat | Titre | Dernière cotation | Fin prévue | Rendement retenu |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in journal.to_dicts():
        lines.append(
            f"| {row['strategy']} | {row['holding_month']} | {row['ticker']} | "
            f"{row['holding_return_end_at'].date()} | "
            f"{row['scheduled_holding_end_at'].date()} | "
            f"{100 * row['realized_return']:.4f} % |"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
