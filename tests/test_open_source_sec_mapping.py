from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.sec_mapping import resolve_sec_company_mapping


def test_resolve_sec_company_mapping_uses_historical_bridges(tmp_path: Path) -> None:
    reference_dir = tmp_path / "data"
    (reference_dir / "eodhd" / "output").mkdir(parents=True)
    pl.DataFrame(
        {
            "Code": ["IPG", "WBA"],
            "Name": ["Interpublic", "Walgreens"],
            "Exchange": ["NYSE", "NASDAQ"],
            "CIK": ["0000051644", "0001618921"],
        }
    ).write_parquet(reference_dir / "eodhd" / "output" / "US_General.parquet")

    sec_mapping_all = pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "name": ["Apple Inc.", "Microsoft Corp."],
            "exchange": ["NASDAQ", "NASDAQ"],
            "cik": [320193, 789019],
        }
    )
    existing_general_reference_lineage = pl.DataFrame(
        {
            "ticker": ["HES.US"],
            "sec_name": ["HESS CORP"],
            "sec_exchange": ["NYSE"],
            "sec_cik": ["0000448271"],
        }
    )

    mapping = resolve_sec_company_mapping(
        requested_tickers=["AAPL", "IPG", "HES", "WBA"],
        sec_mapping_all=sec_mapping_all,
        reference_data_dir=reference_dir,
        existing_general_reference_lineage=existing_general_reference_lineage,
    ).sort("ticker")

    assert mapping["ticker"].to_list() == ["AAPL", "HES", "IPG", "WBA"]
    assert mapping["mapping_source"].to_list() == [
        "sec_live_mapping",
        "raw_sec_lineage_bridge",
        "eodhd_cik_bridge",
        "eodhd_cik_bridge",
    ]
    assert mapping["cik"].to_list() == ["0000320193", "0000448271", "0000051644", "0001618921"]
