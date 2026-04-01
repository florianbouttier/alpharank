from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.publishing import publish_open_source_output_package


def test_publish_open_source_output_package_writes_exact_outputs_and_lineage(tmp_path: Path) -> None:
    legacy_source = tmp_path / "legacy_source"
    legacy_source.mkdir(parents=True)
    (legacy_source / "lineage").mkdir(parents=True)
    legacy_paths = {
        "US_Finalprice.parquet": legacy_source / "US_Finalprice.parquet",
        "SP500Price.parquet": legacy_source / "SP500Price.parquet",
    }
    for path in legacy_paths.values():
        pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-01"]}).write_parquet(path)
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-01"], "selected_method": ["reported_period_end"]}).write_parquet(
        legacy_source / "lineage" / "legacy_share_semantics.parquet"
    )

    output_dir = tmp_path / "output"
    history_root = tmp_path / "history"
    frame = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-01"], "value": [1.0]})
    constituents_path = tmp_path / "SP500_Constituents.csv"
    constituents_path.write_text("Date,Ticker,Name\n2025-01-01,AAPL,Apple\n", encoding="utf-8")

    published = publish_open_source_output_package(
        output_dir=output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=constituents_path,
        prices_frame=frame,
        benchmark_prices=frame,
        general_reference=pl.DataFrame({"ticker": ["AAPL.US"], "name": ["Apple"], "exchange": ["NASDAQ"], "cik": ["1"], "source": ["sec_mapping"]}),
        general_reference_lineage=pl.DataFrame({"ticker": ["AAPL.US"], "name": ["Apple"], "exchange": ["NASDAQ"], "cik": ["1"], "source": ["sec_mapping"]}),
        consolidated_financials=frame,
        consolidated_lineage=frame,
        source_summary=pl.DataFrame({"statement": ["income_statement"], "selected_source": ["sec_companyfacts"]}),
        earnings_consolidated=pl.DataFrame({"ticker": ["AAPL.US"], "reportDate": ["2025-01-30"]}),
        earnings_lineage=pl.DataFrame({"ticker": ["AAPL.US"], "reportDate": ["2025-01-30"]}),
        earnings_long_frame=frame,
        manifest={"run_id": "abc"},
        history_root=history_root,
    )

    assert (output_dir / "US_Finalprice.parquet").exists()
    assert (output_dir / "SP500Price.parquet").exists()
    assert (output_dir / "SP500_Constituents.csv").exists()
    assert (output_dir / "lineage" / "financials_open_source_lineage.parquet").exists()
    assert (output_dir / "lineage" / "earnings_open_source_consolidated.parquet").exists()
    assert (output_dir / "lineage" / "general_reference_lineage.parquet").exists()
    assert (output_dir / "lineage" / "legacy_share_semantics.parquet").exists()
    assert (output_dir / "lineage" / "manifest.json").exists()
    assert "lineage/financials_open_source_lineage.parquet" in published.published_paths
    assert "lineage/legacy_share_semantics.parquet" in published.published_paths
    assert published.snapshot_dir is None

    second = publish_open_source_output_package(
        output_dir=output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=constituents_path,
        prices_frame=frame,
        benchmark_prices=frame,
        general_reference=pl.DataFrame({"ticker": ["AAPL.US"], "name": ["Apple"], "exchange": ["NASDAQ"], "cik": ["1"], "source": ["sec_mapping"]}),
        general_reference_lineage=pl.DataFrame({"ticker": ["AAPL.US"], "name": ["Apple"], "exchange": ["NASDAQ"], "cik": ["1"], "source": ["sec_mapping"]}),
        consolidated_financials=frame,
        consolidated_lineage=frame,
        source_summary=pl.DataFrame({"statement": ["income_statement"], "selected_source": ["sec_companyfacts"]}),
        earnings_consolidated=pl.DataFrame({"ticker": ["AAPL.US"], "reportDate": ["2025-01-30"]}),
        earnings_lineage=pl.DataFrame({"ticker": ["AAPL.US"], "reportDate": ["2025-01-30"]}),
        earnings_long_frame=frame,
        manifest={"run_id": "def"},
        history_root=history_root,
    )
    assert second.snapshot_dir is not None
    assert second.snapshot_dir.exists()
