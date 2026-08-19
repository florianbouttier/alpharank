import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from scripts.run_legacy import (
    DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
    INPUT_PACKAGE_FILENAMES,
    _copy_snapshot_file,
    _parse_args,
    _default_log_stem,
    _input_files,
    _manifest_extra_context,
    _resolve_open_source_output_by_run_id,
    _simulate_historical_legacy_common,
    _snapshot_input_package,
    normalize_year_month_to_timestamp,
)
from scripts.validate_legacy_replay_package import _resolve_project_root
from scripts.validate_legacy_replay_package import validate_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_legacy_exposes_common_holdings_month_normalizer() -> None:
    frame = pd.DataFrame({"year_month": [pd.Period("2026-06", freq="M")]})
    normalized = normalize_year_month_to_timestamp(frame)
    assert normalized["year_month"].iloc[0] == pd.Timestamp("2026-06-01")


def test_historical_legacy_common_explicitly_preserves_missing_return_compatibility() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["Combined_Equal", "Combined_Equal"],
            "decision_month": [date(2018, 12, 1), date(2018, 12, 1)],
            "holding_month": [date(2019, 1, 1), date(2019, 1, 1)],
            "ticker": ["AAA.US", "BBB.US"],
            "target_weight": [0.5, 0.5],
            "realized_return": [0.10, None],
            "benchmark_return": [0.01, 0.01],
        }
    )

    monthly = _simulate_historical_legacy_common(holdings)

    assert monthly["gross_return"].item() == 0.10
    assert monthly["n_positions"].item() == 2


def test_run_legacy_cli_enables_versioned_ticker_registry_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_legacy.py"])
    args = _parse_args()
    assert Path(args.ticker_exclusion_registry) == (
        DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY
    )
    assert args.no_ticker_exclusion_registry is False
    assert args.price_eligibility_policy_id == "monthly_price_eligibility_v1"
    assert args.minimum_monthly_price_observations == 10
    assert args.minimum_monthly_median_dollar_volume == 1_000_000.0
    assert args.maximum_monthly_ohlc_violation_rate == 0.05
    assert args.no_checkpoints is False


def test_run_legacy_cli_can_disable_registry_for_compatibility(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_legacy.py", "--no-ticker-exclusion-registry"],
    )
    args = _parse_args()
    assert args.no_ticker_exclusion_registry is True


def test_run_legacy_cli_can_skip_optional_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_legacy.py", "--no-checkpoints"])
    args = _parse_args()
    assert args.no_checkpoints is True


def test_manifest_extra_context_links_open_source_output_to_latest_run(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "open_source" / "output"
    official_dir = tmp_path / "data" / "open_source" / "official"

    _write_json(
        data_dir / "lineage" / "manifest.json",
        {
            "run_id": "20260531_001503",
            "official_dir": str(official_dir),
            "target_dir": str(official_dir / "target"),
            "output_dir": str(data_dir),
            "legacy_dir": str(official_dir / "target" / "legacy_compatible"),
            "source_refresh_contract": {"snapshot_scope": "full_ingestion"},
        },
    )
    _write_json(
        official_dir / "manifests" / "latest_run.json",
        {
            "run_id": "20260531_001503",
            "mode": "daily",
            "ingested_at": "2026-05-31T00:15:03+00:00",
            "price_window": {"start_date": "2026-05-22", "end_date": "2026-05-31"},
            "financial_years_refreshed": [2025, 2026],
            "ticker_count": 725,
        },
    )
    _write_json(
        official_dir / "runs" / "20260531_001503" / "manifest.json",
        {
            "run_id": "20260531_001503",
            "mode": "daily",
            "ingested_at": "2026-05-31T00:15:03+00:00",
            "price_window": {"start_date": "2026-05-22", "end_date": "2026-05-31"},
            "financial_years_refreshed": [2025, 2026],
            "ticker_count": 725,
        },
    )

    context = _manifest_extra_context(data_dir=data_dir, latest_snapshot=None)

    assert context["consumer"] == "scripts.run_legacy"
    assert context["open_source_output_run_id"] == "20260531_001503"
    assert context["open_source_output_lineage_run_id"] == "20260531_001503"
    assert context["open_source_ingestion_run_id"] == "20260531_001503"
    assert context["open_source_run_id_match"] is True
    assert context["open_source_price_window"] == {"start_date": "2026-05-22", "end_date": "2026-05-31"}
    assert context["open_source_financial_years_refreshed"] == [2025, 2026]
    assert context["open_source_ticker_count"] == 725
    assert context["open_source_source_refresh_scope"] == "full_ingestion"
    assert context["open_source_output_manifest_path"].endswith("output/lineage/manifest.json")
    assert context["open_source_ingestion_manifest_path"].endswith("official/runs/20260531_001503/manifest.json")


def test_manifest_extra_context_prefers_snapshot_run_id_and_flags_lineage_mismatch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "open_source" / "output"
    official_dir = tmp_path / "data" / "open_source" / "official"

    _write_json(data_dir / "snapshot_manifest.json", {"run_id": "snapshot_run"})
    _write_json(data_dir / "lineage" / "manifest.json", {"run_id": "lineage_run", "official_dir": str(official_dir)})
    _write_json(official_dir / "runs" / "snapshot_run" / "manifest.json", {"run_id": "snapshot_run"})

    context = _manifest_extra_context(data_dir=data_dir, latest_snapshot=None)

    assert context["open_source_output_run_id"] == "snapshot_run"
    assert context["open_source_output_snapshot_run_id"] == "snapshot_run"
    assert context["open_source_output_lineage_run_id"] == "lineage_run"
    assert context["open_source_output_manifest_run_id_match"] is False
    assert context["open_source_ingestion_run_id"] == "snapshot_run"
    assert context["open_source_run_id_match"] is True


def test_manifest_extra_context_flags_active_output_drift_from_published_snapshot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "open_source" / "output"
    official_dir = tmp_path / "data" / "open_source" / "official"
    published_snapshot = tmp_path / "data" / "open_source" / "history" / "output" / "open_source_output_20260610_013924"

    for directory, price_value in ((data_dir, "active-price"), (published_snapshot, "published-price")):
        for file_name in INPUT_PACKAGE_FILENAMES.values():
            path = directory / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("same", encoding="utf-8")
        (directory / "US_Finalprice.parquet").write_text(price_value, encoding="utf-8")
    _write_json(
        data_dir / "lineage" / "manifest.json",
        {"run_id": "run_a", "official_dir": str(official_dir)},
    )
    _write_json(
        official_dir / "runs" / "run_a" / "manifest.json",
        {"run_id": "run_a", "published_output_snapshot": "history/output/open_source_output_20260610_013924"},
    )

    context = _manifest_extra_context(data_dir=data_dir, latest_snapshot=None)

    assert context["open_source_output_matches_published_snapshot"] is False
    assert context["open_source_output_published_snapshot_differing_files"] == ["US_Finalprice.parquet"]


def test_resolve_open_source_output_by_run_id_prefers_history_snapshot(tmp_path: Path) -> None:
    current_output = tmp_path / "data" / "open_source" / "output"
    history_output = tmp_path / "data" / "open_source" / "history" / "output" / "open_source_output_20260531_014133"
    _write_json(current_output / "lineage" / "manifest.json", {"run_id": "current_run"})
    _write_json(history_output / "snapshot_manifest.json", {"run_id": "current_run"})

    resolved = _resolve_open_source_output_by_run_id(tmp_path, "current_run")

    assert resolved == history_output


def test_resolve_open_source_output_by_run_id_finds_history_snapshot(tmp_path: Path) -> None:
    current_output = tmp_path / "data" / "open_source" / "output"
    history_output = tmp_path / "data" / "open_source" / "history" / "output" / "open_source_output_20260531_014133"
    _write_json(current_output / "lineage" / "manifest.json", {"run_id": "current_run"})
    _write_json(history_output / "snapshot_manifest.json", {"run_id": "history_run"})
    _write_json(history_output / "lineage" / "manifest.json", {"run_id": "stale_lineage_run"})

    resolved = _resolve_open_source_output_by_run_id(tmp_path, "history_run")

    assert resolved == history_output


def test_snapshot_input_package_copies_canonical_inputs_and_lineage(tmp_path: Path) -> None:
    source_data_dir = tmp_path / "source"
    for filename in INPUT_PACKAGE_FILENAMES.values():
        path = source_data_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content for {filename}", encoding="utf-8")
    _write_json(source_data_dir / "lineage" / "manifest.json", {"run_id": "lineage_run"})
    _write_json(source_data_dir / "snapshot_manifest.json", {"run_id": "snapshot_run"})

    snapshot_dir = _snapshot_input_package(
        source_data_dir=source_data_dir,
        input_files=_input_files(source_data_dir),
        run_day_dir=tmp_path / "outputs" / "2026-06-08",
    )

    assert snapshot_dir.name == "input_snapshot"
    for filename in INPUT_PACKAGE_FILENAMES.values():
        assert (snapshot_dir / filename).read_text(encoding="utf-8") == f"content for {filename}"
    assert json.loads((snapshot_dir / "lineage" / "manifest.json").read_text(encoding="utf-8")) == {"run_id": "lineage_run"}
    assert json.loads((snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")) == {"run_id": "snapshot_run"}
    storage = json.loads(
        (snapshot_dir / "storage_manifest.json").read_text(encoding="utf-8")
    )
    assert storage["strategy"] == "copy_on_write_with_physical_copy_fallback"
    assert storage["file_count"] == len(INPUT_PACKAGE_FILENAMES) + 2
    assert sum(storage["storage_mode_counts"].values()) == storage["file_count"]


def test_copy_snapshot_file_falls_back_to_physical_copy(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("immutable snapshot", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")

    mode = _copy_snapshot_file(source, destination)

    assert mode == "physical_copy"
    assert destination.read_text(encoding="utf-8") == "immutable snapshot"


def test_validate_legacy_replay_package_requires_retained_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "outputs" / "2026-06-08" / "input_snapshot"
    data_file = snapshot_dir / "US_Finalprice.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("price data", encoding="utf-8")
    digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "outputs" / "2026-06-08" / "data_input_manifest.json"
    _write_json(
        manifest_path,
        {
            "input_snapshot_dir": str(snapshot_dir),
            "datasets": {
                "final_price": {
                    "canonical_path": str(data_file),
                    "sha256": digest,
                }
            },
            "run_config": {"source_input_sha256": {"final_price": digest}},
            "code_context": {"critical_file_sha256": {}},
        },
    )

    errors, warnings = validate_manifest(manifest_path)

    assert errors == []
    assert warnings == []


def test_validate_legacy_replay_package_rejects_open_source_lineage_mismatch(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "outputs" / "2026-06-08" / "input_snapshot"
    data_file = snapshot_dir / "US_Finalprice.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("price data", encoding="utf-8")
    digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    manifest_path = tmp_path / "outputs" / "2026-06-08" / "data_input_manifest.json"
    _write_json(
        manifest_path,
        {
            "input_snapshot_dir": str(snapshot_dir),
            "datasets": {
                "final_price": {
                    "canonical_path": str(data_file),
                    "sha256": digest,
                }
            },
            "run_config": {"source_input_sha256": {"final_price": digest}},
            "code_context": {"critical_file_sha256": {}},
            "open_source_output_manifest_run_id_match": False,
            "open_source_output_matches_published_snapshot": False,
            "open_source_output_published_snapshot_differing_files": ["US_Finalprice.parquet"],
        },
    )

    errors, _ = validate_manifest(manifest_path)

    assert "open_source_output_manifest_run_id_match is false" in errors
    assert (
        "open_source_output_matches_published_snapshot is false: ['US_Finalprice.parquet']"
        in errors
    )


def test_validate_legacy_replay_package_rejects_missing_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "outputs" / "2026-06-08" / "data_input_manifest.json"
    _write_json(
        manifest_path,
        {
            "datasets": {},
            "run_config": {"source_input_sha256": {}},
            "code_context": {"critical_file_sha256": {}},
        },
    )

    errors, _ = validate_manifest(manifest_path)

    assert "missing input_snapshot_dir" in errors


def test_default_log_stem_identifies_open_source_runs() -> None:
    assert _default_log_stem(data_dir="data/open_source/output", open_source_run_id=None) == "run_legacy_open_source"
    assert (
        _default_log_stem(data_dir=None, open_source_run_id="2026/05/31 00:15")
        == "run_legacy_open_source_2026_05_31_00_15"
    )


def test_validate_project_root_resolution_finds_repo_from_nested_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested_manifest = repo_root / "outputs" / "2026-06-10" / "runs" / "20260610_133845" / "data_input_manifest.json"
    code_file = repo_root / "scripts" / "run_legacy.py"
    code_file.parent.mkdir(parents=True, exist_ok=True)
    code_file.write_text("# code", encoding="utf-8")
    nested_manifest.parent.mkdir(parents=True, exist_ok=True)
    nested_manifest.write_text("{}", encoding="utf-8")

    assert _resolve_project_root(nested_manifest, ["scripts/run_legacy.py"]) == repo_root
