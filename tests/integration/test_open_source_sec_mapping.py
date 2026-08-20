from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.sec_mapping import resolve_sec_company_mapping


def test_resolve_sec_company_mapping_uses_historical_bridges(tmp_path: Path) -> None:
    reference_dir = tmp_path / "data"
    (reference_dir / "eodhd" / "output").mkdir(parents=True)
    (reference_dir / "sec").mkdir(parents=True)
    pl.DataFrame(
        {
            "Code": ["IPG", "WBA"],
            "Name": ["Interpublic", "Walgreens"],
            "Exchange": ["NYSE", "NASDAQ"],
            "CIK": ["0000051644", "0001618921"],
        }
    ).write_parquet(reference_dir / "eodhd" / "output" / "US_General.parquet")
    (reference_dir / "sec" / "manual_historical_ticker_bridge.csv").write_text(
        "\n".join(
            [
                "ticker,name,exchange,cik,start_date,end_date,mapping_source,mapping_priority",
                "DTV,DIRECTV,NYSE,0000944868,1990-01-01,2015-07-31,sec_manual_historical_bridge,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sec_mapping_all = pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "DTV"],
            "name": ["Apple Inc.", "Microsoft Corp.", "DTE Energy Co"],
            "exchange": ["NASDAQ", "NASDAQ", "NYSE"],
            "cik": [320193, 789019, 936340],
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
        requested_tickers=["AAPL", "DTV", "IPG", "HES", "WBA"],
        sec_mapping_all=sec_mapping_all,
        reference_data_dir=reference_dir,
        existing_general_reference_lineage=existing_general_reference_lineage,
    ).sort("ticker")

    assert mapping["ticker"].to_list() == ["AAPL", "DTV", "HES", "IPG", "WBA"]
    assert mapping["mapping_source"].to_list() == [
        "sec_live_mapping",
        "sec_manual_historical_bridge",
        "raw_sec_lineage_bridge",
        "eodhd_cik_bridge",
        "eodhd_cik_bridge",
    ]
    assert mapping["cik"].to_list() == ["0000320193", "0000944868", "0000448271", "0000051644", "0001618921"]


def test_resolve_sec_company_mapping_expands_dot_share_class_aliases() -> None:
    sec_mapping_all = pl.DataFrame(
        {
            "ticker": ["BF-B", "BRK-B"],
            "name": ["Brown-Forman", "Berkshire Hathaway"],
            "exchange": ["NYSE", "NYSE"],
            "cik": [14693, 1067983],
        }
    )

    mapping = resolve_sec_company_mapping(
        requested_tickers=["BF-B", "BF.B", "BRK-B", "BRK.B"],
        sec_mapping_all=sec_mapping_all,
    ).sort("ticker")

    assert mapping["ticker"].to_list() == ["BF-B", "BF.B", "BRK-B", "BRK.B"]
    assert mapping["cik"].to_list() == [
        "0000014693",
        "0000014693",
        "0001067983",
        "0001067983",
    ]
