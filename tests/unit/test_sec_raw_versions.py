from __future__ import annotations

import polars as pl

from alpharank.data.open_source.sec_raw_versions import (
    rebuild_full_companyfacts_versions,
)


def _frame(rows: list[tuple[str, str, float, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [row[0] for row in rows],
            "statement": ["shares"] * len(rows),
            "metric": ["outstanding_shares"] * len(rows),
            "date": ["2020-12-31"] * len(rows),
            "filing_date": [row[1] for row in rows],
            "value": [row[2] for row in rows],
            "source": ["sec_companyfacts"] * len(rows),
            "ingested_at": [row[3] for row in rows],
        }
    )


def test_full_companyfacts_rebuild_preserves_versions_and_unrefreshed_tickers() -> None:
    retained = _frame(
        [
            ("AAA.US", "2022-02-01", 110.0, "old"),
            ("FAILED.US", "2021-02-01", 50.0, "old"),
        ]
    )
    refresh = _frame(
        [
            ("AAA.US", "2021-02-01", 100.0, "new"),
            ("AAA.US", "2022-02-01", 110.0, "new"),
        ]
    )

    rebuilt = rebuild_full_companyfacts_versions(
        retained=retained,
        full_refresh=refresh,
    )

    assert rebuilt.select("ticker", "filing_date", "value").rows() == [
        ("AAA.US", "2021-02-01", 100.0),
        ("AAA.US", "2022-02-01", 110.0),
        ("FAILED.US", "2021-02-01", 50.0),
    ]
