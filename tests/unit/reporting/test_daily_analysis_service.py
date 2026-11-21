from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import polars as pl
import pytest

from alpharank.governance_contracts.common import sha256_path
from alpharank.reporting.strategy_analysis import (
    AnalysisQuery,
    prepare_analysis_dataset,
    query_analysis,
)
from alpharank.reporting.strategy_analysis.coverage import monthly_fallback_window
from alpharank.reporting.strategy_analysis.dataset import read_performance_payload
from alpharank.reporting.strategy_analysis.report import (
    load_analysis_report,
    write_daily_analysis_report,
)
from alpharank.reporting.strategy_analysis.server import create_analysis_server

BENCHMARK = "SPY · Total return"


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "snapshot"
    root.mkdir()
    days = ["2026-01-30", "2026-02-02", "2026-02-03", "2026-02-27"]
    for filename, ticker, values in [
        ("US_Finalprice.parquet", "A", [100.0, 90.0, 105.0, 110.0]),
        ("SP500Price.parquet", "SPY.US", [100.0, 99.0, 103.0, 105.0]),
    ]:
        pl.DataFrame(
            {"ticker": [ticker] * 4, "date": days, "adjusted_close": values}
        ).write_parquet(root / filename)
    manifest = root / "snapshot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "composition_id": "fixture",
                "output_sha256": {
                    name: sha256_path(root / name)
                    for name in ["US_Finalprice.parquet", "SP500Price.parquet"]
                },
            }
        )
    )
    return root, manifest


def _payload(tmp_path: Path) -> dict:
    root, manifest = _snapshot(tmp_path)
    names = ["Alpha", "Beta", "Gamma", BENCHMARK]
    return {
        "schema_version": 5,
        "generated_at_utc": "2026-03-01T00:00:00+00:00",
        "title": "Fixture",
        "strategy_order": names,
        "strategies": [
            {
                "label": name,
                "family": "Benchmark" if name == BENCHMARK else "Boosting",
                "color": "#2563eb",
                "status": "rd",
            }
            for name in names
        ],
        "monthly": [
            {
                "strategy": name,
                "holding_month": "2026-02-01",
                "decision_month": "2026-01-01",
                "net_return": 0.05 if name == BENCHMARK else 0.099,
                "benchmark_return": 0.05,
                "transaction_cost": 0.0 if name == BENCHMARK else 0.001,
                "turnover": 0.0 if name == BENCHMARK else 1.0,
                "n_positions": 1,
            }
            for name in names
        ],
        "holdings": [
            {
                "strategy": name,
                "ticker": "A",
                "holding_month": "2026-02-01",
                "decision_month": "2026-01-01",
                "target_weight": 1.0,
                "realized_return": 0.1,
            }
            for name in names[:-1]
        ],
        "calendar": {"start": "2026-02-01", "end": "2026-02-01", "months": 1},
        "metric_fields": ["cagr", "max_drawdown"],
        "metric_windows": {"2026-02-01|2026-02-01": [[0.1, -0.1]] * 4},
        "portfolio_composer": {"strategy_order": names[:-1]},
        "strategy_frontiers": {"strategy_order": names[:-1]},
        "current_portfolio": {"holding_month": "2026-03-01", "holdings": []},
        "methodologies": [{"title": "Fixture method", "pseudo_code": ["Do not drop this"]}],
        "lineage": {
            "composition_id": "fixture",
            "snapshot_dir": str(root),
            "source_files": [
                {
                    "role": "snapshot_manifest",
                    "path": str(manifest),
                    "sha256": sha256_path(manifest),
                }
            ],
        },
    }


def test_service_uses_requested_daily_dates_and_preserves_all_source_fields(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    before = deepcopy(payload)
    dataset = prepare_analysis_dataset(payload)
    view = query_analysis(dataset, AnalysisQuery(date(2026, 2, 3), date(2026, 2, 28)))
    assert payload == before
    assert view["anchor_date"] == "2026-02-02"
    assert view["end_date"] == "2026-02-27"
    assert view["series"]["Alpha"]["metrics"]["total_return"] == pytest.approx(1.099 / 0.899 - 1)


def test_preparation_rejects_price_hash_drift_instead_of_using_revised_quotes(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    path = tmp_path / "snapshot" / "US_Finalprice.parquet"
    frame = pl.read_parquet(path).with_columns(
        (pl.col("adjusted_close") * 2).alias("adjusted_close")
    )
    frame.write_parquet(path)
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_analysis_dataset(payload)


def test_monthly_reference_is_explicit_and_does_not_fabricate_daily_prices(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    source = tmp_path / "snapshot" / "US_Finalprice.parquet"
    pl.read_parquet(source).filter(pl.col("date") != "2026-02-03").write_parquet(source)
    manifest_path = tmp_path / "snapshot" / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][source.name] = sha256_path(source)
    manifest_path.write_text(json.dumps(manifest))
    payload["lineage"]["source_files"][0]["sha256"] = sha256_path(manifest_path)
    dataset = prepare_analysis_dataset(payload)
    strict = query_analysis(dataset, AnalysisQuery(date(2026, 2, 1), date(2026, 2, 28)))
    fallback = query_analysis(
        dataset, AnalysisQuery(date(2026, 2, 1), date(2026, 2, 28), "monthly_reference")
    )
    assert strict["series"]["Alpha"]["metrics"]["max_drawdown"] is None
    assert strict["price_gaps"][0]["ticker"] == "A"
    assert strict["price_gaps"][0]["dates"] == ["2026-02-03"]
    assert fallback["frequency"] == "monthly"
    assert fallback["series"]["Alpha"]["metrics"]["total_return"] == pytest.approx(0.099)
    assert len(fallback["series"]["Alpha"]["points"]) == 2
    assert (
        dataset.ledgers["history"].filter(pl.col("strategy") == "Alpha")["wealth"].null_count() == 1
    )
    assert fallback["price_gaps"] == strict["price_gaps"]


def test_monthly_reference_never_rounds_partial_month_dates_silently(tmp_path: Path) -> None:
    dataset = prepare_analysis_dataset(_payload(tmp_path))
    strict = query_analysis(dataset, AnalysisQuery(date(2026, 2, 3), date(2026, 2, 28)))
    assert strict["monthly_fallback"]["available"] is False
    with pytest.raises(ValueError, match="mois complets"):
        query_analysis(
            dataset, AnalysisQuery(date(2026, 2, 3), date(2026, 2, 28), "monthly_reference")
        )


def test_monthly_fallback_proposes_only_contained_complete_months() -> None:
    view = monthly_fallback_window(date(2020, 1, 10), date(2026, 8, 17), enabled=True)
    assert view["start"] == "2020-02-01" and view["end"] == "2026-07-31"
    assert view["available"] is True and view["period_changed"] is True
    exact = monthly_fallback_window(date(2024, 2, 1), date(2024, 2, 29), enabled=True)
    assert exact["period_changed"] is False
    assert (
        monthly_fallback_window(date(2026, 2, 3), date(2026, 2, 28), enabled=True)["available"]
        is False
    )


def test_daily_writer_keeps_complete_payload_and_refuses_to_overwrite(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    paths = write_daily_analysis_report(
        payload,
        output_dir=tmp_path / "reports" / "first",
        generated_at_utc=datetime.now(timezone.utc),
    )
    combined = read_performance_payload(paths["report"])
    assert combined["analysis"]["initial_view"]["series"]["Alpha"]["points"][0]["wealth"] == 1.0
    for key in payload:
        if key != "generated_at_utc":
            assert combined[key] == payload[key]
    assert (
        load_analysis_report(paths["report"].parent).sources["history"]["max_date"] == "2026-02-28"
    )
    with pytest.raises(ValueError, match="overwrite"):
        write_daily_analysis_report(
            payload, output_dir=paths["report"].parent, generated_at_utc=datetime.now(timezone.utc)
        )


def test_http_viewer_returns_exact_queries_and_rejects_unsafe_or_invalid_requests(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    paths = write_daily_analysis_report(
        payload,
        output_dir=tmp_path / "reports" / "first",
        generated_at_utc=datetime.now(timezone.utc),
    )
    server = create_analysis_server(paths["report"].parent, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/analysis?start=2026-02-03&end=2026-02-28") as response:
            view = json.load(response)
        assert view["anchor_date"] == "2026-02-02"
        for route in [
            "/api/analysis?start=2026-02-28&end=2026-02-03",
            "/api/analysis?start=2026-02-03&start=2026-02-04&end=2026-02-28",
            "/api/analysis?start=2026-02-03&end=2026-02-28&source=unknown",
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + route)
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/", headers={"Host": "untrusted.example"}))
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../../.env")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
