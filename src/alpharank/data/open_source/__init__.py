"""Lazy compatibility facade for historical open-source imports."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_MODULE_ALIASES = {
    "ingestion": "alpharank.data.ingestion.orchestration",
    "ingestion_frames": "alpharank.data.ingestion.frames",
    "ingestion_prices": "alpharank.data.ingestion.prices",
    "ingestion_reference": "alpharank.data.ingestion.reference",
}
_ATTRIBUTE_ALIASES = {
    "OpenSourceCadrageResult": "alpharank.data.ingestion.cadrage",
    "OpenSourceIngestionResult": "alpharank.data.ingestion.orchestration",
    "OpenSourceReferenceRefreshResult": "alpharank.data.ingestion.orchestration",
    "OpenSourcePriceTransitionResult": "alpharank.data.ingestion.transition",
    "repair_open_source_price_history": "alpharank.data.ingestion.orchestration",
    "refresh_open_source_reference_layers": "alpharank.data.ingestion.orchestration",
    "run_open_source_cadrage": "alpharank.data.ingestion.cadrage",
    "run_open_source_ingestion": "alpharank.data.ingestion.orchestration",
    "run_open_source_price_transition": "alpharank.data.ingestion.transition",
}

__all__ = [
    "OpenSourceCadrageResult",
    "OpenSourceIngestionResult",
    "OpenSourceReferenceRefreshResult",
    "OpenSourcePriceTransitionResult",
    "repair_open_source_price_history",
    "refresh_open_source_reference_layers",
    "run_open_source_cadrage",
    "run_open_source_ingestion",
    "run_open_source_price_transition",
    "ingestion",
    "ingestion_frames",
    "ingestion_prices",
    "ingestion_reference",
]


def __getattr__(name: str) -> object:
    """Resolve one reviewed historical name without eager import cycles."""

    module_name = _MODULE_ALIASES.get(name)
    if module_name is not None:
        module: ModuleType = import_module(module_name)
        globals()[name] = module
        return module
    module_name = _ATTRIBUTE_ALIASES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
