from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from alpharank.data.prices.contracts import PriceGatePolicy


@dataclass(frozen=True)
class SourceRefreshPolicy:
    """Refresh contract applied before a full open-source snapshot is published."""

    refresh_full_price_history: bool = True
    refresh_full_sec_companyfacts_history: bool = True
    refresh_sec_companyfacts: bool = True
    persist_sec_companyfacts_payloads: bool = False
    refresh_sec_submissions: bool = True
    persist_sec_filing_metadata: bool = False
    persist_sec_filing_documents: bool = False
    refresh_stockanalysis: bool = True
    persist_stockanalysis_payloads: bool = False
    simfin_refresh_days: int = 0
    historical_revision_guard_days: int = 730
    allow_historical_revisions: bool = False
    historical_revision_review_note: str | None = None
    require_eodhd_price_seed: bool = True
    historical_price_return_revision_threshold: float = 0.0001
    price_transition_factor_jump_threshold: float = 0.0001
    price_recent_mutable_calendar_days: int = 7
    allow_historical_price_revisions: bool = False
    allow_historical_price_key_removals: bool = False

    def price_gate_policy(self) -> PriceGatePolicy:
        return PriceGatePolicy(
            historical_return_revision_threshold=self.historical_price_return_revision_threshold,
            transition_factor_jump_threshold=self.price_transition_factor_jump_threshold,
            recent_mutable_calendar_days=self.price_recent_mutable_calendar_days,
            allow_historical_price_revisions=self.allow_historical_price_revisions,
            allow_historical_price_key_removals=self.allow_historical_price_key_removals,
        )

    def to_manifest(
        self,
        *,
        mode: str,
        price_start_date: str,
        price_end_date: str,
        financial_years: tuple[int, ...],
        snapshot_scope: str = "full_ingestion",
    ) -> dict[str, Any]:
        return {
            "contract_version": 1,
            "snapshot_scope": snapshot_scope,
            "policy": asdict(self),
            "source_semantics": {
                "yfinance_prices": {
                    "fetch": "network",
                    "window": {"start_date": price_start_date, "end_date": price_end_date},
                    "history": (
                        "full available active-universe history upsert into retained official raw"
                        if self.refresh_full_price_history
                        else "rolling overlap upsert into retained official raw"
                    ),
                },
                "sec_companyfacts": {
                    "fetch": "network_full_company_payload" if self.refresh_sec_companyfacts else "cache_allowed",
                    "years_applied": list(financial_years),
                    "persistent_payload_cache": self.persist_sec_companyfacts_payloads,
                },
                "sec_submissions": {
                    "fetch": "network_full_company_payload" if self.refresh_sec_submissions else "cache_allowed",
                    "years_applied": list(financial_years),
                },
                "sec_filing_documents": {
                    "fetch": "network_on_demand",
                    "persistent_metadata_cache": self.persist_sec_filing_metadata,
                    "persistent_cache": self.persist_sec_filing_documents,
                },
                "stockanalysis": {
                    "fetch": "network_full_history" if self.refresh_stockanalysis else "cache_allowed",
                    "persistent_payload_cache": self.persist_stockanalysis_payloads,
                },
                "simfin": {"refresh_days": self.simfin_refresh_days},
                "target_build": {
                    "input": "complete retained official raw store",
                    "mode": mode,
                },
            },
        }


PRODUCTION_SOURCE_REFRESH_POLICY = SourceRefreshPolicy()
