"""Build a guarded price roll-forward from one immutable provider vintage."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from alpharank.data.ingestion.price_publication_candidate import (
    PricePublicationContext,
    build_price_publication_candidate,
    resolve_incomplete_provider_tickers,
)
from alpharank.data.open_source.price_quality import (
    ExtremePriceMoveGateResult,
    audit_extreme_adjusted_price_moves,
    load_reviewed_extreme_price_moves,
)
from alpharank.data.prices import (
    EodhdSeed,
    HybridPriceResult,
    PriceGateResult,
    build_persistent_price_history_registry,
    load_eodhd_seed,
    persistent_history_summary,
    roll_forward_validated_price_history,
    validate_price_candidate,
    validate_price_gate_report,
)
from alpharank.data.prices.contracts import PRODUCTION_PRICE_GATE_POLICY
from alpharank.data.publishing.price_package_inputs import (
    latest_constituents,
    prepare_benchmark_prices,
    refreshable_active_tickers,
    resolve_active_resolution_vintage_id,
    validated_terminal_tickers,
)
from alpharank.data.publishing.price_package_output import (
    PreparedPricePackage,
    PricePackageRequest,
    write_price_package,
)
from alpharank.data.quality.freshness import (
    build_data_freshness_summary,
    validate_data_freshness,
)
from alpharank.data.security_identity import (
    apply_security_identity_policy,
    load_security_identity_registry,
)


@dataclass(frozen=True, slots=True)
class RollForwardEvidence:
    previous: pl.DataFrame
    active_tickers: tuple[str, ...]
    terminal_tickers: tuple[str, ...]
    result: HybridPriceResult
    revision_gate: PriceGateResult
    extreme_gate: ExtremePriceMoveGateResult
    seed: EodhdSeed
    security_identities: pl.DataFrame
    reviewed_registry_manifest: dict[str, object]


def build_price_roll_forward_package(request: PricePackageRequest) -> dict[str, object]:
    """Build, validate and write one new immutable canonical price package."""

    evidence = _prepare_roll_forward_evidence(request)
    benchmark = prepare_benchmark_prices(
        request.benchmark_path.resolve(),
        expected_run_id=request.expected_benchmark_run_id,
    )
    constituents = apply_security_identity_policy(
        pl.read_csv(request.constituents_path.resolve(), infer_schema_length=0),
        ticker_column="Ticker",
        date_column="Date",
        registry=evidence.security_identities,
    )
    freshness = _resolve_freshness(
        request=request,
        prices=evidence.result.prices,
        benchmark=benchmark,
        constituents=constituents.frame,
        terminal_tickers=evidence.terminal_tickers,
    )
    history_registry = build_persistent_price_history_registry(
        evidence.result.lineage,
        active_tickers=evidence.active_tickers,
        preserved_terminal_tickers=evidence.terminal_tickers,
    )
    prepared = PreparedPricePackage(
        result=evidence.result,
        revision_gate=evidence.revision_gate,
        extreme_gate=evidence.extreme_gate,
        benchmark_prices=benchmark,
        constituents=constituents,
        history_registry=history_registry,
        history_summary=persistent_history_summary(history_registry),
        seed=evidence.seed,
        security_identities=evidence.security_identities,
        reviewed_registry_manifest=evidence.reviewed_registry_manifest,
        data_freshness=freshness,
    )
    manifest = write_price_package(request, prepared)
    validate_price_candidate(evidence.revision_gate)
    validate_price_gate_report(manifest["source_refresh_contract"]["price_publication_guard"])
    return manifest


def _prepare_roll_forward_evidence(request: PricePackageRequest) -> RollForwardEvidence:
    previous = pl.read_parquet(request.previous_lineage_path.resolve())
    fresh_yahoo = pl.read_parquet(request.fresh_yahoo_path.resolve())
    identities = load_security_identity_registry()
    active_resolution_id = resolve_active_resolution_vintage_id(
        run_id=request.run_id,
        fresh_yahoo=fresh_yahoo,
    )
    active_tickers = latest_constituents(request.constituents_path.resolve())
    terminal_tickers = validated_terminal_tickers(
        requested=request.preserve_terminal_tickers,
        registry_path=request.constituent_registry_path.resolve(),
        expected_through=request.expected_through,
    )
    refreshable = refreshable_active_tickers(active_tickers, terminal_tickers)
    provider_result = roll_forward_validated_price_history(
        previous_validated_lineage=previous,
        active_yahoo_vintage=fresh_yahoo,
        active_tickers=active_tickers,
        preserved_terminal_tickers=terminal_tickers,
        active_resolution_vintage_id=active_resolution_id,
        security_identity_registry=identities,
    )
    seed = load_eodhd_seed(request.eodhd_seed_path.resolve(), start_date=request.start_date)
    publication_candidate = build_price_publication_candidate(
        provider_result,
        fresh_yahoo,
        previous,
        context=PricePublicationContext(
            active_tickers=active_tickers,
            preserved_terminal_tickers=terminal_tickers,
            expected_eodhd_keys=seed.frame.select("ticker", "date"),
            expected_through=request.expected_through,
            run_id=active_resolution_id,
            policy=PRODUCTION_PRICE_GATE_POLICY,
            incomplete_provider_tickers=resolve_incomplete_provider_tickers(
                dict(request.source_refresh_contract)
            ),
            previous_comparison_prices=previous.select(provider_result.prices.columns),
        ),
    )
    result = publication_candidate.hybrid
    revision_gate = publication_candidate.gate
    extreme_gate, reviewed_manifest = _review_extreme_moves(
        request=request,
        previous=previous,
        candidate=result.prices,
        refreshable=refreshable,
    )
    return RollForwardEvidence(
        previous=previous,
        active_tickers=active_tickers,
        terminal_tickers=terminal_tickers,
        result=result,
        revision_gate=revision_gate,
        extreme_gate=extreme_gate,
        seed=seed,
        security_identities=identities,
        reviewed_registry_manifest=reviewed_manifest,
    )


def _review_extreme_moves(
    *,
    request: PricePackageRequest,
    previous: pl.DataFrame,
    candidate: pl.DataFrame,
    refreshable: tuple[str, ...],
) -> tuple[object, dict[str, object]]:
    reviewed, registry_manifest = load_reviewed_extreme_price_moves(
        request.reviewed_move_registry_path.resolve()
    )
    review_keys = candidate.select("ticker", "date").join(
        previous.select("ticker", "date"),
        on=["ticker", "date"],
        how="anti",
    )
    gate = audit_extreme_adjusted_price_moves(
        candidate,
        review_keys=review_keys,
        tickers=list(refreshable),
        reviewed_moves=reviewed,
    )
    return gate, registry_manifest


def _resolve_freshness(
    *,
    request: PricePackageRequest,
    prices: pl.DataFrame,
    benchmark: pl.DataFrame,
    constituents: pl.DataFrame,
    terminal_tickers: tuple[str, ...],
) -> dict[str, object]:
    if request.data_freshness is not None:
        return dict(request.data_freshness)
    if request.sec_package_dir is None:
        raise RuntimeError("An acquired run requires a validated SEC package for freshness")
    lineage = request.sec_package_dir.resolve() / "lineage"
    financials = pl.read_parquet(lineage / "financials_sec_consolidated.parquet")
    earnings = pl.read_parquet(lineage / "earnings_sec_consolidated.parquet")
    freshness = build_data_freshness_summary(
        prices=prices,
        benchmark_prices=benchmark,
        financials=financials,
        earnings_sec_calendar=earnings,
        constituents=constituents,
        terminal_tickers=terminal_tickers,
    )
    validate_data_freshness(freshness, expected_through=request.expected_through)
    return freshness
