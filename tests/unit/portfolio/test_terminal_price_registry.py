from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from alpharank.portfolio.terminal_price_registry import (
    DEFAULT_TERMINAL_PRICE_REGISTRY,
    load_terminal_price_registry,
)


def test_terminal_successor_prices_match_sealed_snapshot_and_sec_evidence() -> None:
    registry = load_terminal_price_registry()
    base = registry.payload["base_snapshot"]
    prices = pl.DataFrame(
        {
            "ticker": ["KHC.US", "CI.US"],
            "date": ["2015-07-31", "2018-12-31"],
            "close": [79.47000122070312, 189.9199981689453],
        }
    )

    prices = registry.successor_prices(
        prices,
        snapshot_id=base["snapshot_id"],
        composition_id=base["composition_id"],
        price_artifact_sha256=base["price_artifact_sha256"],
    )

    assert prices.height == 3
    assert prices["ticker"].to_list() == ["KHC.US", "CI.US", "ECA.US"]
    assert prices["holding_end_price"].to_list() == pytest.approx(
        [79.47000122070312, 189.9199981689453, 7.25]
    )
    assert prices["price_vintage_id"].n_unique() == 1
    eca = prices.filter(pl.col("ticker") == "ECA.US").row(0, named=True)
    assert eca["price_source_kind"] == "sec_issuer_filing"
    assert eca["price_asof_date"].isoformat() == "2019-02-28"


def test_terminal_successor_prices_reject_snapshot_or_reviewed_value_drift(
    tmp_path: Path,
) -> None:
    registry = load_terminal_price_registry()
    base = registry.payload["base_snapshot"]
    prices = pl.DataFrame(
        {
            "ticker": ["KHC.US", "CI.US"],
            "date": ["2015-07-31", "2018-12-31"],
            "close": [79.47000122070312, 189.9199981689453],
        }
    )
    with pytest.raises(ValueError, match="do not match the sealed base snapshot"):
        registry.successor_prices(
            prices,
            snapshot_id="wrong",
            composition_id=base["composition_id"],
            price_artifact_sha256=base["price_artifact_sha256"],
        )

    payload = deepcopy(registry.payload)
    payload["observations"][0]["holding_end_price"] = 80.0
    path = tmp_path / "terminal_prices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    drifted = load_terminal_price_registry(path)
    with pytest.raises(ValueError, match="differs from its reviewed value"):
        drifted.successor_prices(
            prices,
            snapshot_id=base["snapshot_id"],
            composition_id=base["composition_id"],
            price_artifact_sha256=base["price_artifact_sha256"],
        )


def test_terminal_price_registry_rejects_ambiguous_observation(tmp_path: Path) -> None:
    payload = deepcopy(json.loads(DEFAULT_TERMINAL_PRICE_REGISTRY.read_text(encoding="utf-8")))
    payload["observations"].append(deepcopy(payload["observations"][0]))
    payload["observations"][-1]["observation_id"] = "OTHER"
    path = tmp_path / "terminal_prices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate successor price key"):
        load_terminal_price_registry(path)
