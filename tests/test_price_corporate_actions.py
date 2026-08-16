from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from alpharank.data.prices.corporate_actions import (
    combine_stock_split_evidence,
    load_confirmed_stock_splits,
)


def test_confirmed_split_registry_is_hash_manifested_and_filtered(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps(
            {
                "registry_id": "actions-v1",
                "events": [
                    {
                        "ticker": "MNST.US",
                        "action": "stock_split",
                        "effective_date": "2026-08-11",
                        "split_ratio": 2.0,
                        "source": "issuer",
                        "source_url": "https://example.com/mnst",
                    },
                    {
                        "ticker": "AAA.US",
                        "action": "stock_split",
                        "effective_date": "2026-01-01",
                        "split_ratio": 3.0,
                        "source": "issuer",
                        "source_url": "https://example.com/aaa",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    frame, manifest = load_confirmed_stock_splits(path, tickers=["MNST.US"])

    assert frame.select("ticker", "date", "split_ratio").rows() == [
        ("MNST.US", "2026-08-11", 2.0)
    ]
    assert manifest["registry_id"] == "actions-v1"
    assert len(manifest["sha256"]) == 64


def test_reviewed_split_evidence_replaces_endpoint_duplicate() -> None:
    endpoint = pl.DataFrame(
        {
            "ticker": ["MNST.US"],
            "date": ["2026-08-11"],
            "split_ratio": [2.0],
            "source": ["yahoo_actions"],
        }
    )
    reviewed = pl.DataFrame(
        {
            "ticker": ["MNST.US"],
            "date": ["2026-08-11"],
            "split_ratio": [2.0],
            "source": ["issuer"],
            "source_url": ["https://example.com/mnst"],
        }
    )

    combined = combine_stock_split_evidence(endpoint, reviewed)

    assert combined.height == 1
    assert combined["source"].item() == "issuer"
