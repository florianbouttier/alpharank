"""Maintained builder for the same-snapshot Legacy/boosting comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from alpharank.data.terminal_eligibility import (
    TERMINAL_ENTRY_POLICY_ID,
    apply_terminal_entry_gate_to_decisions,
)
from alpharank.governance import capture_runtime_provenance, reserve_run_directory
from alpharank.multihorizon.config import validate_latest_common_comparison_profile
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.lineage import (
    load_manifest,
    require_matching_data_contexts,
    require_matching_price_eligibility,
    require_matching_ticker_exclusions,
)
from alpharank.portfolio.maturity import PortfolioMaturitySplit, split_completed_portfolio_months
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.terminal_event_registry import (
    TerminalEventRegistry,
    load_terminal_event_registry,
)
from alpharank.strategy.legacy_valuation import (
    build_legacy_valuation_registry,
    filter_predictions_to_legacy_valuation_universe,
)

DEFAULT_TOP_N_VALUES = (5, 10, 15, 20)


@dataclass(frozen=True, slots=True)
class CommonStrategyReplayConfig:
    """Explicit application inputs for one publishable common replay."""

    legacy_run_dir: Path
    boosting_run_dir: Path
    output_dir: Path
    project_root: Path
    command_argv: tuple[str, ...]
    transaction_cost_bps: float = 10.0
    top_n_values: tuple[int, ...] = DEFAULT_TOP_N_VALUES
    include_legacy_valuation_universe: bool = True


@dataclass(frozen=True, slots=True)
class CommonReplayInputs:
    """Resolved files required by the common replay contract."""

    legacy_manifest: Path
    boosting_manifest: Path
    predictions: Path
    legacy_monthly: Path
    legacy_holdings: Path
    snapshot_dir: Path


@dataclass(frozen=True, slots=True)
class CommonReplayChecks:
    """Validated lineage and comparison-profile evidence."""

    boosting_manifest: dict[str, Any]
    comparison_profile: dict[str, Any]
    lineage: dict[str, Any]
    ticker_exclusions: dict[str, Any]
    price_eligibility: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GatedPredictionFrames:
    """Completed/live predictions and explicit terminal-entry audit."""

    maturity: PortfolioMaturitySplit
    completed: pl.DataFrame
    live: pl.DataFrame
    journal: pl.DataFrame
    completed_block_count: int
    live_block_count: int


@dataclass(frozen=True, slots=True)
class PortfolioFrames:
    """Portfolio frames sharing the exact realized calendar."""

    holdings: pl.DataFrame
    monthly: pl.DataFrame
    live_holdings: pl.DataFrame
    valuation_registry: pl.DataFrame
    universe_summary: tuple[dict[str, object], ...]
    expected_months: pl.DataFrame
    common_start: object
    common_end: object


def build_common_strategy_replay(config: CommonStrategyReplayConfig) -> Path:
    """Build the canonical native-universe replay and its manifest."""

    top_n_values = _validated_top_n(config.top_n_values)
    inputs = _resolve_inputs(config)
    destination = reserve_run_directory(config.output_dir)
    checks = _validate_common_context(config, inputs)
    terminal_registry = load_terminal_event_registry()
    provenance = _capture_provenance(
        config,
        inputs=inputs,
        checks=checks,
        destination=destination,
        top_n_values=top_n_values,
    )
    predictions = pl.read_parquet(inputs.predictions)
    gated = _gate_predictions(predictions, terminal_registry=terminal_registry)
    portfolios = _build_portfolio_frames(
        inputs,
        gated=gated,
        top_n_values=top_n_values,
        transaction_cost_bps=config.transaction_cost_bps,
        include_legacy_valuation_universe=config.include_legacy_valuation_universe,
    )
    artifacts = _write_artifacts(
        destination,
        portfolios=portfolios,
        terminal_journal=gated.journal,
    )
    return _write_manifest(
        destination,
        config=config,
        inputs=inputs,
        checks=checks,
        provenance=provenance,
        terminal_registry=terminal_registry,
        gated=gated,
        portfolios=portfolios,
        artifacts=artifacts,
        top_n_values=top_n_values,
    )


def build_native_boosting_holdings(
    predictions: pl.DataFrame,
    *,
    top_n_values: Sequence[int] = DEFAULT_TOP_N_VALUES,
) -> pl.DataFrame:
    """Build the characterized native Top-N holdings without universe filtering."""

    values = _validated_top_n(top_n_values)
    return _build_boosting_holdings(
        (("native", predictions),),
        top_n_values=values,
    )


def _build_boosting_holdings(
    prediction_sets: Sequence[tuple[str, pl.DataFrame]],
    *,
    top_n_values: Sequence[int],
) -> pl.DataFrame:
    return pl.concat(
        [
            boosting_predictions_to_holdings(
                prediction_frame,
                strategy=(
                    f"Boosting Top {top_n}"
                    if universe == "native"
                    else f"Boosting Top {top_n} | Legacy PE universe"
                ),
                top_n=top_n,
            )
            for universe, prediction_frame in prediction_sets
            for top_n in top_n_values
        ],
        how="diagonal_relaxed",
    )


def _resolve_inputs(config: CommonStrategyReplayConfig) -> CommonReplayInputs:
    paths = CommonReplayInputs(
        legacy_manifest=config.legacy_run_dir / "data_input_manifest.json",
        boosting_manifest=config.boosting_run_dir / "manifest.json",
        predictions=config.boosting_run_dir / "classification_h06/predictions.parquet",
        legacy_monthly=config.legacy_run_dir / "legacy_common_monthly.parquet",
        legacy_holdings=config.legacy_run_dir / "legacy_common_holdings.parquet",
        snapshot_dir=(config.legacy_run_dir / "input_snapshot").resolve(),
    )
    for path in (
        paths.legacy_manifest,
        paths.boosting_manifest,
        paths.predictions,
        paths.legacy_monthly,
        paths.legacy_holdings,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def _validate_common_context(
    config: CommonStrategyReplayConfig,
    inputs: CommonReplayInputs,
) -> CommonReplayChecks:
    boosting_manifest = load_manifest(inputs.boosting_manifest)
    comparison_profile = validate_latest_common_comparison_profile(boosting_manifest["config"])
    if not comparison_profile["passed"]:
        raise ValueError(
            "Boosting run does not satisfy the public latest-common comparison "
            f"profile: {comparison_profile['mismatches']}"
        )
    lineage = require_matching_data_contexts(
        inputs.legacy_manifest,
        inputs.boosting_manifest,
        required_keys=set(boosting_manifest["input_paths"]),
    )
    ticker_exclusions = require_matching_ticker_exclusions(
        inputs.legacy_manifest, inputs.boosting_manifest
    )
    price_eligibility = require_matching_price_eligibility(
        inputs.legacy_manifest, inputs.boosting_manifest
    )
    if not all(
        check.get("passed", False) for check in (lineage, ticker_exclusions, price_eligibility)
    ):
        raise ValueError("Common replay lineage gates did not all pass.")
    _validate_declared_snapshot(config, inputs, boosting_manifest)
    _validate_legacy_return_hashes(inputs, boosting_manifest)
    return CommonReplayChecks(
        boosting_manifest=boosting_manifest,
        comparison_profile=comparison_profile,
        lineage=lineage,
        ticker_exclusions=ticker_exclusions,
        price_eligibility=price_eligibility,
    )


def _validate_declared_snapshot(
    config: CommonStrategyReplayConfig,
    inputs: CommonReplayInputs,
    boosting_manifest: Mapping[str, Any],
) -> None:
    declared = Path(str(boosting_manifest["config"]["data_dir"]))
    if not declared.is_absolute():
        declared = config.project_root / declared
    if declared.resolve() != inputs.snapshot_dir:
        raise ValueError(
            "Legacy and boosting do not declare the same input snapshot: "
            f"{inputs.snapshot_dir} != {declared.resolve()}"
        )


def _validate_legacy_return_hashes(
    inputs: CommonReplayInputs,
    boosting_manifest: Mapping[str, Any],
) -> None:
    for key, filename in (
        ("legacy_detailed_returns", "legacy_detailed_returns_polars.parquet"),
        ("legacy_monthly_returns", "legacy_monthly_returns_polars.parquet"),
    ):
        source = inputs.legacy_manifest.parent / filename
        if _hash(source) != boosting_manifest[f"{key}_sha256"]:
            raise ValueError(f"Boosting does not match {source.name} from the Legacy run.")


def _capture_provenance(
    config: CommonStrategyReplayConfig,
    *,
    inputs: CommonReplayInputs,
    checks: CommonReplayChecks,
    destination: Path,
    top_n_values: tuple[int, ...],
) -> dict[str, Any]:
    provenance = capture_runtime_provenance(
        project_root=config.project_root,
        entrypoint="scripts/build_common_legacy_boosting_replay.py",
        command_argv=config.command_argv,
        resolved_config={
            "transaction_cost_bps_times_turnover": config.transaction_cost_bps,
            "top_n_values": list(top_n_values),
            "timing_contract": "decision_month=t; holding_month=t+1",
            "missing_return_policy": "raise_after_selection",
            "benchmark": "SPY total return from adjusted_close",
        },
        seeds={"replay_randomness": "none_deterministic_inputs_only"},
        critical_files=_critical_files(),
        data_identifiers={
            "legacy_manifest_sha256": _hash(inputs.legacy_manifest),
            "boosting_manifest_sha256": _hash(inputs.boosting_manifest),
            "boosting_predictions_sha256": _hash(inputs.predictions),
            "input_snapshot_dir": str(inputs.snapshot_dir),
            "matching_input_keys": checks.lineage["matching_keys"],
        },
        patch_path=destination / "runtime_git_patch.json",
    )
    if provenance["git"]["dirty"]:
        raise ValueError(
            "A publishable common replay must run from a clean Git worktree. "
            f"Patch evidence was retained in {destination}."
        )
    return provenance


def _critical_files() -> tuple[str, ...]:
    return (
        "scripts/build_common_legacy_boosting_replay.py",
        "src/alpharank/data/terminal_eligibility.py",
        "src/alpharank/governance.py",
        "src/alpharank/governance_contracts/common.py",
        "src/alpharank/governance_contracts/contracts.py",
        "src/alpharank/governance_contracts/runtime_provenance.py",
        "src/alpharank/multihorizon/config.py",
        "src/alpharank/portfolio/adapters/boosting.py",
        "src/alpharank/portfolio/artifacts.py",
        "src/alpharank/portfolio/lineage.py",
        "src/alpharank/portfolio/maturity.py",
        "src/alpharank/portfolio/simulation.py",
        "src/alpharank/portfolio/terminal_event_registry.py",
        "src/alpharank/replay/common_strategy.py",
        "src/alpharank/strategy/legacy_valuation.py",
        "configs/data_quality/terminal_shareholder_events_v1.json",
    )


def _gate_predictions(
    predictions: pl.DataFrame,
    *,
    terminal_registry: TerminalEventRegistry,
) -> GatedPredictionFrames:
    maturity = split_completed_portfolio_months(predictions)
    completed, completed_blocks = _gate_terminal_entries(
        maturity.completed_predictions,
        entry_blocks=terminal_registry.terminal_entry_blocks(),
        prediction_partition="completed_replay",
    )
    live, live_blocks = _gate_terminal_entries(
        maturity.score_only_predictions,
        entry_blocks=terminal_registry.terminal_entry_blocks(),
        prediction_partition="score_only_live",
    )
    journal = pl.concat([completed_blocks, live_blocks], how="diagonal_relaxed").sort(
        ["holding_month", "prediction_partition", "ticker"]
    )
    return GatedPredictionFrames(
        maturity=maturity,
        completed=completed,
        live=live,
        journal=journal,
        completed_block_count=completed_blocks.height,
        live_block_count=live_blocks.height,
    )


def _gate_terminal_entries(
    predictions: pl.DataFrame,
    *,
    entry_blocks: pl.DataFrame,
    prediction_partition: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if predictions.is_empty():
        return predictions.clone(), _empty_terminal_journal()
    gate = apply_terminal_entry_gate_to_decisions(predictions, entry_blocks)
    journal = gate.blocked.select(
        pl.lit(prediction_partition).alias("prediction_partition"),
        "decision_month",
        "holding_month",
        "ticker",
        *[column for column in ("score", "target_status_1m") if column in gate.blocked.columns],
        "terminal_event_id",
        pl.col("_terminal_blocked_from_month").alias("blocked_from_holding_month"),
        "entry_block_rule",
    )
    return gate.eligible.drop("holding_month"), journal


def _empty_terminal_journal() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "prediction_partition": pl.String,
            "decision_month": pl.Date,
            "holding_month": pl.Date,
            "ticker": pl.String,
            "score": pl.Float64,
            "target_status_1m": pl.String,
            "terminal_event_id": pl.String,
            "blocked_from_holding_month": pl.Date,
            "entry_block_rule": pl.String,
        }
    )


def _build_portfolio_frames(
    inputs: CommonReplayInputs,
    *,
    gated: GatedPredictionFrames,
    top_n_values: tuple[int, ...],
    transaction_cost_bps: float,
    include_legacy_valuation_universe: bool,
) -> PortfolioFrames:
    prediction_sets, valuation_registry = _prediction_universes(
        inputs,
        predictions=gated.completed,
        include_legacy_valuation_universe=include_legacy_valuation_universe,
    )
    boosting_holdings = _build_boosting_holdings(
        prediction_sets,
        top_n_values=top_n_values,
    )
    _reject_censored_holdings(boosting_holdings)
    live_holdings = (
        build_native_boosting_holdings(gated.live, top_n_values=top_n_values)
        if not gated.live.is_empty()
        else pl.DataFrame()
    )
    boosting_monthly = _simulate_strategies(
        boosting_holdings, transaction_cost_bps=transaction_cost_bps
    )
    if boosting_monthly.is_empty():
        raise ValueError("Boosting produced no realized portfolio month.")
    common_start = boosting_monthly["holding_month"].min()
    common_end = boosting_monthly["holding_month"].max()
    expected_months = _expected_months(boosting_monthly, top_n_values[0])
    legacy_holdings, references = _build_legacy_references(
        inputs,
        common_start=common_start,
        common_end=common_end,
        transaction_cost_bps=transaction_cost_bps,
    )
    _require_exact_calendar(references, expected_months)
    return PortfolioFrames(
        holdings=pl.concat([boosting_holdings, legacy_holdings], how="diagonal_relaxed"),
        monthly=pl.concat([boosting_monthly, references], how="diagonal_relaxed"),
        live_holdings=live_holdings,
        valuation_registry=valuation_registry,
        universe_summary=tuple(
            {
                "universe": universe,
                "rows": frame.height,
                "tickers": frame["ticker"].n_unique(),
                "months": frame["decision_month"].n_unique(),
            }
            for universe, frame in prediction_sets
        ),
        expected_months=expected_months,
        common_start=common_start,
        common_end=common_end,
    )


def _prediction_universes(
    inputs: CommonReplayInputs,
    *,
    predictions: pl.DataFrame,
    include_legacy_valuation_universe: bool,
) -> tuple[tuple[tuple[str, pl.DataFrame], ...], pl.DataFrame]:
    if not include_legacy_valuation_universe:
        return (("native", predictions),), pl.DataFrame()
    registry = build_legacy_valuation_registry(
        snapshot_dir=inputs.snapshot_dir,
        candidates=predictions,
    )
    matched = filter_predictions_to_legacy_valuation_universe(predictions, registry)
    return (("native", predictions), ("legacy_valuation", matched)), registry


def _build_legacy_references(
    inputs: CommonReplayInputs,
    *,
    common_start: object,
    common_end: object,
    transaction_cost_bps: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    source = pl.read_parquet(inputs.legacy_monthly).filter(
        pl.col("holding_month").is_between(common_start, common_end)
    )
    holdings = (
        pl.read_parquet(inputs.legacy_holdings)
        .filter(
            (pl.col("strategy") == "Combined_Frequency")
            & pl.col("holding_month").is_between(common_start, common_end)
        )
        .with_columns(pl.lit("Legacy").alias("strategy"))
    )
    legacy_monthly = simulate_weighted_portfolio(
        holdings,
        transaction_cost_bps=transaction_cost_bps,
        causal_timing_policy="legacy_month_only",
    )
    spy = reference_monthly_series(
        source.filter(pl.col("strategy") == "SPY total return"),
        strategy="SPY total return",
        return_column="net_return",
    )
    return holdings, pl.concat([legacy_monthly, spy], how="diagonal_relaxed")


def _simulate_strategies(
    holdings: pl.DataFrame,
    *,
    transaction_cost_bps: float,
) -> pl.DataFrame:
    return pl.concat(
        [
            simulate_weighted_portfolio(
                frame,
                transaction_cost_bps=transaction_cost_bps,
                causal_timing_policy="legacy_month_only",
            )
            for frame in holdings.partition_by("strategy", maintain_order=True)
        ],
        how="diagonal_relaxed",
    )


def _reject_censored_holdings(holdings: pl.DataFrame) -> None:
    censored = holdings.filter(pl.col("target_status_1m") == "approved_censored_last_observation")
    if censored.height:
        examples = (
            censored.select("strategy", "decision_month", "holding_month", "ticker")
            .head(10)
            .to_dicts()
        )
        raise ValueError(
            "Selected Boosting holdings still use censored zero-return targets "
            f"after the terminal entry gate: {examples}"
        )


def _expected_months(monthly: pl.DataFrame, first_top_n: int) -> pl.DataFrame:
    return (
        monthly.filter(pl.col("strategy") == f"Boosting Top {first_top_n}")
        .select("holding_month")
        .unique()
        .sort("holding_month")
    )


def _require_exact_calendar(references: pl.DataFrame, expected: pl.DataFrame) -> None:
    for strategy in ("Legacy", "SPY total return"):
        observed = references.filter(pl.col("strategy") == strategy).select("holding_month")
        if (
            observed.join(expected, on="holding_month", how="anti").height
            or expected.join(observed, on="holding_month", how="anti").height
        ):
            raise ValueError(f"{strategy} does not cover the exact boosting calendar.")


def _write_artifacts(
    output_dir: Path,
    *,
    portfolios: PortfolioFrames,
    terminal_journal: pl.DataFrame,
) -> dict[str, Path]:
    artifacts = write_common_portfolio_artifacts(
        output_dir=output_dir,
        holdings=portfolios.holdings,
        monthly_returns=portfolios.monthly,
        prefix="comparison_common",
        benchmark_metadata={
            "id": "spy_total_return_adjusted_close",
            "label": "SPY total return",
            "price_column": "adjusted_close",
            "includes_distributions": True,
        },
    )
    artifacts.update(_write_terminal_journal(output_dir, terminal_journal))
    if not portfolios.live_holdings.is_empty():
        artifacts.update(_write_live_holdings(output_dir, portfolios.live_holdings))
    if not portfolios.valuation_registry.is_empty():
        artifacts.update(_write_valuation_registry(output_dir, portfolios.valuation_registry))
    return artifacts


def _write_terminal_journal(output_dir: Path, journal: pl.DataFrame) -> dict[str, Path]:
    parquet_path = output_dir / "terminal_entry_journal.parquet"
    csv_path = output_dir / "terminal_entry_journal.csv"
    journal.write_parquet(parquet_path)
    journal.write_csv(csv_path)
    return {"terminal_entry_journal": parquet_path, "terminal_entry_journal_csv": csv_path}


def _write_live_holdings(output_dir: Path, holdings: pl.DataFrame) -> dict[str, Path]:
    parquet_path = output_dir / "boosting_live_score_holdings.parquet"
    csv_path = output_dir / "boosting_live_score_holdings.csv"
    holdings.write_parquet(parquet_path)
    holdings.write_csv(csv_path)
    return {
        "boosting_live_score_holdings": parquet_path,
        "boosting_live_score_holdings_csv": csv_path,
    }


def _write_valuation_registry(
    output_dir: Path,
    registry: pl.DataFrame,
) -> dict[str, Path]:
    registry_path = output_dir / "legacy_valuation_eligibility.parquet"
    summary_path = output_dir / "legacy_valuation_eligibility_summary.csv"
    registry.write_parquet(registry_path)
    (
        registry.group_by("eligibility_reason")
        .agg(
            pl.len().alias("ticker_months"),
            pl.col("ticker").n_unique().alias("unique_tickers"),
        )
        .sort("ticker_months", descending=True)
        .write_csv(summary_path)
    )
    return {
        "legacy_valuation_eligibility": registry_path,
        "legacy_valuation_eligibility_summary": summary_path,
    }


def _write_manifest(
    output_dir: Path,
    *,
    config: CommonStrategyReplayConfig,
    inputs: CommonReplayInputs,
    checks: CommonReplayChecks,
    provenance: Mapping[str, Any],
    terminal_registry: TerminalEventRegistry,
    gated: GatedPredictionFrames,
    portfolios: PortfolioFrames,
    artifacts: Mapping[str, Path],
    top_n_values: tuple[int, ...],
) -> Path:
    manifest = {
        "status": "canonical_common_strategy_replay",
        "methodology_status": "validated_same_snapshot_common_replay",
        "comparison_eligible": True,
        "publication_eligible": True,
        "timing_contract": "decision_month=t; holding_month=t+1",
        "calendar": {
            "start_holding_month": str(portfolios.common_start),
            "end_holding_month": str(portfolios.common_end),
            "months": portfolios.expected_months.height,
            "latest_month_status": "complete_realized_one_month_return",
        },
        "transaction_cost_policy": {
            "strategies": _costed_strategy_names(
                top_n_values,
                include_legacy_valuation_universe=(config.include_legacy_valuation_universe),
            ),
            "bps_times_turnover": config.transaction_cost_bps,
            "benchmark": "SPY total return has no simulated trading cost",
        },
        "comparison_profile": checks.comparison_profile,
        "lineage_check": checks.lineage,
        "ticker_exclusion_check": checks.ticker_exclusions,
        "price_eligibility_check": checks.price_eligibility,
        "runtime_provenance": provenance,
        "sources": _manifest_sources(inputs),
        "model_calendar": {
            "score_only_tail": checks.boosting_manifest.get("protocol", {}).get("score_only_tail"),
            "portfolio_maturity": gated.maturity.manifest,
            "results": checks.boosting_manifest.get("results"),
        },
        "terminal_entry_gate": _terminal_gate_manifest(gated, terminal_registry),
        "boosting_allocation": {
            "top_n_values": list(top_n_values),
            "universes": list(portfolios.universe_summary),
            "native": "historical membership plus shared monthly price gate",
            "legacy_valuation": (
                "native universe intersected at decision month with Legacy's "
                "point-in-time market-cap and 0 < PE < 100 gate"
                if config.include_legacy_valuation_universe
                else None
            ),
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _hash(path)}
            for name, path in artifacts.items()
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _costed_strategy_names(
    top_n_values: Sequence[int],
    *,
    include_legacy_valuation_universe: bool,
) -> list[str]:
    native = [f"Boosting Top {value}" for value in top_n_values]
    matched = (
        [f"Boosting Top {value} | Legacy PE universe" for value in top_n_values]
        if include_legacy_valuation_universe
        else []
    )
    return [*native, *matched, "Legacy"]


def _manifest_sources(inputs: CommonReplayInputs) -> dict[str, dict[str, str]]:
    return {
        "legacy_run_manifest": {
            "path": str(inputs.legacy_manifest.resolve()),
            "sha256": _hash(inputs.legacy_manifest),
        },
        "boosting_manifest": {
            "path": str(inputs.boosting_manifest.resolve()),
            "sha256": _hash(inputs.boosting_manifest),
        },
        "boosting_predictions": {
            "path": str(inputs.predictions.resolve()),
            "sha256": _hash(inputs.predictions),
        },
    }


def _terminal_gate_manifest(
    gated: GatedPredictionFrames,
    registry: TerminalEventRegistry,
) -> dict[str, object]:
    return {
        "policy_id": TERMINAL_ENTRY_POLICY_ID,
        "registry_id": registry.payload["registry_id"],
        "registry_sha256": registry.sha256,
        "registry_path": str(registry.path),
        "completed_predictions_before_gate": gated.maturity.completed_predictions.height,
        "completed_predictions_after_gate": gated.completed.height,
        "completed_predictions_blocked": gated.completed_block_count,
        "score_only_predictions_before_gate": gated.maturity.score_only_predictions.height,
        "score_only_predictions_after_gate": gated.live.height,
        "score_only_predictions_blocked": gated.live_block_count,
        "selected_censored_zero_return_rows": 0,
    }


def _validated_top_n(values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(value) for value in values))
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("top_n_values must contain positive integers.")
    return normalized


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
