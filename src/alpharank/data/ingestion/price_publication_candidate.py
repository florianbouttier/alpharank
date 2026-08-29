"""Resolve a provider price observation into a publishable canonical candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from alpharank.data.prices import (
    HybridPriceResult,
    PriceCandidateMode,
    PriceGateResult,
    PriceReconciliationContext,
    PriceReconciliationResult,
    PriceTickerTransitionResult,
    apply_price_ticker_transition_overlay,
    audit_price_candidate,
    build_price_revision_diagnostic,
    reconcile_validated_price_history,
)
from alpharank.data.prices.contracts import PriceGatePolicy

RESOLVABLE_PROVIDER_BLOCKS = {"unreviewed_historical_return_revisions"}


@dataclass(frozen=True, slots=True)
class PricePublicationContext:
    active_tickers: Sequence[str]
    preserved_terminal_tickers: Sequence[str]
    expected_eodhd_keys: pl.DataFrame
    expected_through: str
    run_id: str
    policy: PriceGatePolicy
    incomplete_provider_tickers: Sequence[str] = ()
    previous_comparison_prices: pl.DataFrame | None = None
    ticker_transition_registry: pl.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class PricePublicationCandidate:
    hybrid: HybridPriceResult
    gate: PriceGateResult
    provider_gate: PriceGateResult
    reconciliation: PriceReconciliationResult | None
    ticker_transition: PriceTickerTransitionResult
    revision_diagnostic: dict[str, object]


def build_price_publication_candidate(
    provider_hybrid: HybridPriceResult,
    current_yahoo_observation: pl.DataFrame,
    previous_lineage: pl.DataFrame | None,
    *,
    context: PricePublicationContext,
) -> PricePublicationCandidate:
    """Audit the provider, retain validated keys, then audit the canonical result."""

    previous_prices = context.previous_comparison_prices
    if previous_prices is None and previous_lineage is not None:
        previous_prices = previous_lineage.select(provider_hybrid.prices.columns)
    provider_gate = _audit_provider_candidate(
        provider_hybrid=provider_hybrid,
        previous_prices=previous_prices,
        context=context,
    )
    if previous_lineage is None or previous_prices is None:
        return _build_initial_publication_candidate(
            provider_hybrid=provider_hybrid,
            previous_prices=previous_prices,
            provider_gate=provider_gate,
            context=context,
        )
    reconciliation = reconcile_validated_price_history(
        previous_validated_lineage=previous_lineage,
        current_yahoo_observation=current_yahoo_observation,
        context=PriceReconciliationContext(
            active_tickers=context.active_tickers,
            preserved_terminal_tickers=context.preserved_terminal_tickers,
            incomplete_provider_tickers=context.incomplete_provider_tickers,
            run_id=context.run_id,
        ),
    )
    reconciled_hybrid = HybridPriceResult(
        prices=reconciliation.prices,
        lineage=reconciliation.lineage,
        composition_report=provider_hybrid.composition_report,
    )
    transition = apply_price_ticker_transition_overlay(
        reconciliation.lineage,
        registry=context.ticker_transition_registry,
    )
    hybrid = _hybrid_with_transition(
        reconciled_hybrid,
        transition,
        reconciliation_report=reconciliation.report,
    )
    canonical_gate = _audit_reconciled_candidate(
        hybrid=hybrid,
        previous_prices=previous_prices,
        reconciliation=reconciliation,
        context=context,
    )
    combined_gate = _combine_publication_gates(
        provider_gate=provider_gate,
        canonical_gate=canonical_gate,
        reconciliation=reconciliation,
    )
    diagnostic = build_price_revision_diagnostic(
        provider_gate=provider_gate,
        previous_prices=previous_prices,
        provider_prices=provider_hybrid.prices,
        expected_through=context.expected_through,
        policy=context.policy,
    )
    return PricePublicationCandidate(
        hybrid=hybrid,
        gate=combined_gate,
        provider_gate=provider_gate,
        reconciliation=reconciliation,
        ticker_transition=transition,
        revision_diagnostic=diagnostic,
    )


def _build_initial_publication_candidate(
    *,
    provider_hybrid: HybridPriceResult,
    previous_prices: pl.DataFrame | None,
    provider_gate: PriceGateResult,
    context: PricePublicationContext,
) -> PricePublicationCandidate:
    transition = apply_price_ticker_transition_overlay(
        provider_hybrid.lineage,
        registry=context.ticker_transition_registry,
    )
    hybrid = _hybrid_with_transition(provider_hybrid, transition)
    canonical_gate = _audit_provider_candidate(
        provider_hybrid=hybrid,
        previous_prices=previous_prices,
        context=context,
    )
    return PricePublicationCandidate(
        hybrid=hybrid,
        gate=canonical_gate,
        provider_gate=provider_gate,
        reconciliation=None,
        ticker_transition=transition,
        revision_diagnostic={"status": "not_applicable_without_validated_vintage"},
    )


def _hybrid_with_transition(
    base: HybridPriceResult,
    transition: PriceTickerTransitionResult,
    *,
    reconciliation_report: dict[str, object] | None = None,
) -> HybridPriceResult:
    report = dict(base.composition_report)
    if reconciliation_report is not None:
        report["canonical_reconciliation"] = reconciliation_report
    report["price_ticker_transition"] = transition.report
    return HybridPriceResult(
        prices=transition.prices,
        lineage=transition.lineage,
        composition_report=report,
    )


def _audit_provider_candidate(
    *,
    provider_hybrid: HybridPriceResult,
    previous_prices: pl.DataFrame | None,
    context: PricePublicationContext,
) -> PriceGateResult:
    return audit_price_candidate(
        previous_prices=previous_prices,
        candidate_prices=provider_hybrid.prices,
        candidate_lineage=provider_hybrid.lineage,
        active_tickers=_quality_active_tickers(context),
        expected_eodhd_keys=context.expected_eodhd_keys,
        expected_through=context.expected_through,
        policy=context.policy,
        active_resolution_vintage_id=context.run_id,
    )


def _audit_reconciled_candidate(
    *,
    hybrid: HybridPriceResult,
    previous_prices: pl.DataFrame,
    reconciliation: PriceReconciliationResult,
    context: PricePublicationContext,
) -> PriceGateResult:
    return audit_price_candidate(
        previous_prices=previous_prices,
        candidate_prices=hybrid.prices,
        candidate_lineage=hybrid.lineage,
        active_tickers=_quality_active_tickers(context),
        expected_eodhd_keys=context.expected_eodhd_keys,
        expected_through=context.expected_through,
        policy=context.policy,
        active_resolution_vintage_id=context.run_id,
        candidate_mode=PriceCandidateMode.VALIDATED_HISTORY_RETURN_EXTENSION,
        observed_active_tickers=reconciliation.observed_active_tickers,
    )


def _combine_publication_gates(
    *,
    provider_gate: PriceGateResult,
    canonical_gate: PriceGateResult,
    reconciliation: PriceReconciliationResult,
) -> PriceGateResult:
    provider_blocks = set(provider_gate.report["blocking_reasons"])
    unresolved_provider = provider_blocks - RESOLVABLE_PROVIDER_BLOCKS
    canonical_blocks = set(canonical_gate.report["blocking_reasons"])
    reconciliation_blocks = set(reconciliation.report["blocking_reasons"])
    blocking_reasons = sorted(unresolved_provider | canonical_blocks | reconciliation_blocks)
    report = {
        **canonical_gate.report,
        "provider_observation_passed_original_gate": provider_gate.report["passed"],
        "provider_observation_blocking_reasons": sorted(provider_blocks),
        "provider_revision_resolution": reconciliation.report,
        "resolved_provider_blocking_reasons": sorted(provider_blocks & RESOLVABLE_PROVIDER_BLOCKS)
        if reconciliation.report["passed"]
        else [],
        "blocking_reasons": blocking_reasons,
        "passed": not blocking_reasons,
    }
    return PriceGateResult(
        report=report,
        daily_return_revisions=canonical_gate.daily_return_revisions,
        transition_factor_findings=canonical_gate.transition_factor_findings,
        historical_key_removals=canonical_gate.historical_key_removals,
    )


def _quality_active_tickers(context: PricePublicationContext) -> tuple[str, ...]:
    terminal = {_normalize_ticker(ticker) for ticker in context.preserved_terminal_tickers}
    return tuple(
        ticker
        for raw_ticker in context.active_tickers
        if (ticker := _normalize_ticker(raw_ticker)) not in terminal
    )


def resolve_incomplete_provider_tickers(
    source_refresh_contract: dict[str, object],
) -> tuple[str, ...]:
    """Read tickers whose previous validated prefix DEF deliberately retained."""

    semantics = source_refresh_contract.get("source_semantics")
    yahoo = semantics.get("yfinance_prices") if isinstance(semantics, dict) else None
    coverage = yahoo.get("validated_key_coverage") if isinstance(yahoo, dict) else None
    definitive = coverage.get("definitive_resolution") if isinstance(coverage, dict) else None
    tickers = (
        definitive.get("frozen_previous_prefix_tickers") if isinstance(definitive, dict) else None
    )
    if not isinstance(tickers, list):
        return ()
    return tuple(str(ticker) for ticker in tickers)


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).upper()
    return value if value.endswith(".US") else f"{value}.US"
