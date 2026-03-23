from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.publishing import publish_open_source_output_package


def test_publish_open_source_output_package_writes_exact_outputs_and_lineage(tmp_path: Path) -> None:
    legacy_source = tmp_path / "legacy_source"
    legacy_source.mkdir(parents=True)
    legacy_paths = {
        "US_Finalprice.parquet": legacy_source / "US_Finalprice.parquet",
        "SP500Price.parquet": legacy_source / "SP500Price.parquet",
    }
    for path in legacy_paths.values():
        pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-01"]}).write_parquet(path)

    output_dir = tmp_path / "output"
    frame = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-01"], "value": [1.0]})

    published = publish_open_source_output_package(
        output_dir=output_dir,
        legacy_paths=legacy_paths,
        prices_frame=frame,
        benchmark_prices=frame,
        general_reference=pl.DataFrame({"ticker": ["AAPL.US"], "name": ["Apple"], "exchange": ["NASDAQ"], "cik": ["1"], "source": ["sec_mapping"]}),
        consolidated_financials=frame,
        consolidated_lineage=frame,
        source_summary=pl.DataFrame({"statement": ["income_statement"], "selected_source": ["sec_companyfacts"]}),
        earnings_frame=pl.DataFrame({"ticker": ["AAPL.US"], "reportDate": ["2025-01-30"]}),
        earnings_long_frame=frame,
        manifest={"run_id": "abc"},
    )

    assert (output_dir / "US_Finalprice.parquet").exists()
    assert (output_dir / "SP500Price.parquet").exists()
    assert (output_dir / "lineage" / "financials_open_source_lineage.parquet").exists()
    assert (output_dir / "lineage" / "manifest.json").exists()
    assert "lineage/financials_open_source_lineage.parquet" in published
