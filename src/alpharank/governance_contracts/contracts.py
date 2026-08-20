"""Shared versions, schemas and errors for governance contracts."""

from __future__ import annotations

BASELINE_MANIFEST_NAME = "baseline_manifest.json"
BASELINE_SEAL_NAME = "baseline_manifest.sha256"
BASELINE_CONTRACT_VERSION = 1
ECONOMIC_PREFIX_CONTRACT_VERSION = 1
RUNTIME_PROVENANCE_CONTRACT_VERSION = 1
SEALED_CONFIRMATION_CONTRACT_VERSION = 1
APPROVED_NUMERIC_TOLERANCE = 1e-12

HOLDINGS_PREFIX_KEYS = (
    "strategy",
    "decision_month",
    "holding_month",
    "ticker",
)
HOLDINGS_PREFIX_NUMERIC_COLUMNS = (
    "target_weight",
    "realized_return",
    "benchmark_return",
)
HOLDINGS_PREFIX_EXACT_COLUMNS = (
    "selection_rank",
    "sector",
    "return_resolution",
    "terminal_event_id",
)
MONTHLY_PREFIX_KEYS = ("strategy", "decision_month", "holding_month")
MONTHLY_PREFIX_NUMERIC_COLUMNS = (
    "gross_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "benchmark_return",
    "active_return",
    "relative_return",
)
MONTHLY_PREFIX_EXACT_COLUMNS = ("n_positions", "sector_count")


class BaselineValidationError(RuntimeError):
    """Raised when a sealed methodology baseline is incomplete or modified."""


class EconomicPrefixError(RuntimeError):
    """Raised when a supposedly neutral migration changes published economics."""


class RuntimeProvenanceError(RuntimeError):
    """Raised when a run manifest does not prove its runtime provenance."""


class SealedConfirmationError(RuntimeError):
    """Raised when the final confirmation protocol can no longer be promoted."""
