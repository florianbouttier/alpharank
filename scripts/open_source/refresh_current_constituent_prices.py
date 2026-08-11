#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
import html
from pathlib import Path
import json

import polars as pl

from alpharank.data.open_source.constituents import (
    current_constituent_price_coverage,
    sha256_file,
)
from alpharank.data.open_source.ingestion import (
    _canonicalize_price_tickers,
    _consolidate_price_sources,
    _with_price_ingestion_metadata,
)
from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs
from alpharank.data.open_source.publishing import publish_open_source_output_package
from alpharank.data.open_source.price_quality import (
    EXTREME_ADJUSTED_RETURN_THRESHOLD,
    assert_no_extreme_adjusted_price_moves,
)
from alpharank.data.open_source.storage import (
    OpenSourceLivePaths,
    acquire_process_json_lock,
    append_run_delta,
    merge_upsert_frames,
    new_run_id,
    read_json,
    upsert_parquet,
    utc_now_iso,
    write_run_manifest,
)
from alpharank.data.open_source.yahoo import YahooFinanceClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_DIR = PROJECT_ROOT / "data" / "open_source" / "official"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data"
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "data_quality" / "sp500_constituent_changes_2026.json"
DEFAULT_TICKERS = ("BNY", "ECHO", "FDXF", "FLEX", "HONA", "MRVL", "VEEV")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill newly active S&P 500 ticker prices and republish the complete "
            "open-source package without refreshing every historical ticker."
        )
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--maximum-absolute-daily-return",
        type=float,
        default=EXTREME_ADJUSTED_RETURN_THRESHOLD,
        help="Fail before writing when a refreshed adjusted close exceeds this move.",
    )
    parser.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--finalize-current-package",
        action="store_true",
        help="Finalize the already-published targeted run without creating another snapshot.",
    )
    args = parser.parse_args()

    paths = OpenSourceLivePaths(args.official_dir.resolve())
    paths.ensure()
    acquire_process_json_lock(
        paths.manifests_dir / "nightly.lock.json",
        operation="current_constituent_price_refresh",
    )
    if args.finalize_current_package:
        _finalize_current_package(paths)
        return
    run_id = new_run_id()
    ingested_at = utc_now_iso()
    end_date = args.end_date or date.today().isoformat()
    requested = tuple(sorted({str(ticker).upper() for ticker in args.tickers}))
    prior_manifest = read_json(paths.latest_manifest_path)
    prior_run_id = prior_manifest.get("run_id") if isinstance(prior_manifest, dict) else None

    yahoo = YahooFinanceClient(
        cache_dir=PROJECT_ROOT / "data" / "open_source" / "_cache" / "yfinance"
    )
    delta = _with_price_ingestion_metadata(
        yahoo.download_prices(requested, args.start_date, end_date),
        dataset="prices_yfinance_current_constituent_backfill",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    benchmark_delta = _with_price_ingestion_metadata(
        yahoo.download_prices(["SPY"], args.start_date, end_date),
        dataset="prices_spy_yfinance_current_refresh",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    covered = tuple(
        delta.filter(pl.col("adjusted_close").is_not_null())
        .select(pl.col("ticker").str.replace(r"\.US$", "").unique().sort())
        .to_series()
        .to_list()
    )
    missing = tuple(sorted(set(requested) - set(covered)))
    if missing:
        raise RuntimeError(
            "The targeted price refresh did not return every required current ticker: "
            f"{missing}. The official package was not republished."
        )

    raw_yahoo_path = paths.raw_dir / "prices_yfinance.parquet"
    existing_yahoo = pl.read_parquet(raw_yahoo_path)
    prospective_yahoo = merge_upsert_frames(
        existing_yahoo,
        delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )
    all_constituent_tickers = (
        pl.read_csv(args.reference_dir / "SP500_Constituents.csv")
        .select(pl.col("Ticker").drop_nulls().unique().sort())
        .to_series()
        .to_list()
    )
    prospective_yahoo = _canonicalize_price_tickers(
        prospective_yahoo,
        ticker_list=all_constituent_tickers,
    )
    raw_simfin = pl.read_parquet(paths.raw_dir / "prices_simfin.parquet")
    raw_stockanalysis = pl.read_parquet(paths.raw_dir / "prices_stockanalysis.parquet")
    prospective_prices, prospective_lineage = _consolidate_price_sources(
        [prospective_yahoo, raw_simfin, raw_stockanalysis],
        ticker_list=all_constituent_tickers,
    )
    assert_no_extreme_adjusted_price_moves(
        prospective_prices,
        event_since=args.start_date,
        tickers=[f"{ticker}.US" for ticker in requested],
        threshold=args.maximum_absolute_daily_return,
    )

    append_run_delta(
        paths.run_dir(run_id) / "raw" / "prices_yfinance.parquet",
        delta,
    )
    append_run_delta(
        paths.run_dir(run_id) / "raw" / "prices_spy_yfinance.parquet",
        benchmark_delta,
    )
    raw_yahoo = prospective_yahoo
    raw_yahoo.write_parquet(raw_yahoo_path)
    raw_benchmark = upsert_parquet(
        paths.raw_dir / "prices_spy_yfinance.parquet",
        benchmark_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )
    benchmark = raw_benchmark.select(
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
            "ticker",
        ]
    ).sort(["ticker", "date"])
    benchmark.write_parquet(
        paths.target_dir / "benchmark_prices_open_source.parquet"
    )
    clean_prices, clean_lineage = prospective_prices, prospective_lineage
    clean_prices.write_parquet(paths.target_dir / "prices_open_source.parquet")
    clean_lineage.write_parquet(
        paths.target_dir / "prices_open_source_lineage.parquet"
    )

    general = pl.read_parquet(paths.target_dir / "general_reference.parquet")
    general_lineage = pl.read_parquet(
        paths.target_dir / "general_reference_lineage.parquet"
    )
    financials = pl.read_parquet(
        paths.target_dir / "financials_open_source_consolidated.parquet"
    )
    financial_lineage = pl.read_parquet(
        paths.target_dir / "financials_open_source_lineage.parquet"
    )
    source_summary = pl.read_parquet(
        paths.target_dir / "financials_open_source_source_summary.parquet"
    )
    earnings = pl.read_parquet(
        paths.target_dir / "earnings_open_source_consolidated.parquet"
    )
    earnings_lineage = pl.read_parquet(
        paths.target_dir / "earnings_open_source_lineage.parquet"
    )
    earnings_long = pl.read_parquet(
        paths.target_dir / "earnings_open_source_long.parquet"
    )
    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark,
        general_reference=general,
        consolidated_financials=financials,
        consolidated_lineage=financial_lineage,
        earnings_frame=earnings,
        reference_data_dir=args.reference_dir,
        output_dir=paths.legacy_dir,
    )
    package_manifest = {
        "run_id": run_id,
        "mode": "current_constituent_price_refresh",
        "generated_at": ingested_at,
        "ingested_at": ingested_at,
        "prior_run_id": prior_run_id,
        "official_dir": str(paths.base_dir),
        "target_dir": str(paths.target_dir),
        "output_dir": str(paths.output_dir),
        "legacy_dir": str(paths.legacy_dir),
        "refresh_tickers": requested,
        "price_quality": {
            "maximum_absolute_daily_return": args.maximum_absolute_daily_return,
            "event_since": args.start_date,
            "status": "passed",
        },
        "constituent_registry": str(args.registry.resolve()),
        "constituent_registry_sha256": sha256_file(args.registry),
        "constituents_sha256": sha256_file(
            args.reference_dir / "SP500_Constituents.csv"
        ),
    }
    published = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=args.reference_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        prices_lineage=clean_lineage,
        benchmark_prices=benchmark,
        general_reference=general,
        general_reference_lineage=general_lineage,
        consolidated_financials=financials,
        consolidated_lineage=financial_lineage,
        source_summary=source_summary,
        earnings_consolidated=earnings,
        earnings_lineage=earnings_lineage,
        earnings_long_frame=earnings_long,
        manifest=package_manifest,
        history_root=paths.root_dir / "history" / "output",
    )
    max_price_date = clean_prices.select(
        pl.col("date")
        .cast(pl.Date)
        .filter(pl.col("adjusted_close").is_not_null())
        .max()
    ).item()
    max_benchmark_date = benchmark.select(
        pl.col("date")
        .cast(pl.Date)
        .filter(pl.col("adjusted_close").is_not_null())
        .max()
    ).item()
    ticker_coverage = (
        clean_prices.filter(
            pl.col("ticker")
            .str.replace(r"\.US$", "")
            .is_in(list(requested))
        )
        .group_by(
            pl.col("ticker").str.replace(r"\.US$", "").alias("ticker")
        )
        .agg(
            pl.col("date")
            .cast(pl.Date)
            .filter(pl.col("adjusted_close").is_not_null())
            .min()
            .alias("min_date"),
            pl.col("date")
            .cast(pl.Date)
            .filter(pl.col("adjusted_close").is_not_null())
            .max()
            .alias("max_date"),
            pl.col("adjusted_close").is_not_null().sum().alias("price_rows"),
        )
        .sort("ticker")
    )
    current_coverage, current_member_dates = current_constituent_price_coverage(
        clean_prices,
        constituents_path=args.reference_dir / "SP500_Constituents.csv",
    )
    paths.run_dir(run_id).mkdir(parents=True, exist_ok=True)
    ticker_coverage.write_csv(
        paths.run_dir(run_id) / "current_constituent_price_coverage.csv"
    )
    current_member_dates.write_csv(
        paths.run_dir(run_id) / "all_current_constituent_price_coverage.csv"
    )
    _write_price_coverage_html(
        paths.run_dir(run_id) / "current_constituent_price_coverage.html",
        run_id=run_id,
        published_snapshot=str(published.snapshot_dir),
        current_coverage=current_coverage,
        targeted_coverage=ticker_coverage,
    )
    manifest = {
        **package_manifest,
        "price_window": {
            "start_date": args.start_date,
            "end_date": end_date,
            "max_published_price_date": str(max_price_date),
            "max_published_benchmark_date": str(max_benchmark_date),
        },
        "ticker_count": clean_prices.select(
            pl.col("ticker").n_unique()
        ).item(),
        "price_rows": clean_prices.height,
        "refresh_ticker_coverage": _json_safe_rows(ticker_coverage),
        "current_constituent_coverage": current_coverage,
        "published_output_snapshot": (
            str(published.snapshot_dir.relative_to(paths.root_dir))
            if published.snapshot_dir is not None
            else None
        ),
        "published_output": {
            name: str(path.relative_to(paths.root_dir))
            for name, path in published.published_paths.items()
        },
    }
    write_run_manifest(paths, run_id, manifest)

    print(f"Run id: {run_id}")
    print(f"Prior run id: {prior_run_id}")
    print(f"Refreshed tickers: {', '.join(requested)}")
    print(ticker_coverage)
    print(f"Published price max: {max_price_date}")
    print(f"Published benchmark max: {max_benchmark_date}")
    print(f"Output snapshot: {published.snapshot_dir}")
    print(f"Official manifest: {paths.latest_manifest_path}")


def _finalize_current_package(paths: OpenSourceLivePaths) -> None:
    package_manifest_path = paths.output_lineage_dir / "manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    run_id = str(package_manifest["run_id"])
    coverage_path = paths.run_dir(run_id) / "current_constituent_price_coverage.csv"
    if not coverage_path.exists():
        raise FileNotFoundError(
            f"Cannot finalize {run_id}: missing coverage audit {coverage_path}."
        )
    matching_snapshots = []
    for snapshot_dir in (paths.root_dir / "history" / "output").glob(
        "open_source_output_*"
    ):
        snapshot_manifest = snapshot_dir / "snapshot_manifest.json"
        if not snapshot_manifest.exists():
            continue
        payload = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        if str(payload.get("run_id")) == run_id:
            matching_snapshots.append(snapshot_dir)
    if not matching_snapshots:
        raise FileNotFoundError(f"No published output snapshot matches run_id={run_id}.")
    snapshot_dir = sorted(matching_snapshots)[-1]
    clean_prices = pl.read_parquet(paths.target_dir / "prices_open_source.parquet")
    max_price_date = clean_prices.select(
        pl.col("date")
        .cast(pl.Date)
        .filter(pl.col("adjusted_close").is_not_null())
        .max()
    ).item()
    constituents_path = paths.output_dir / "SP500_Constituents.csv"
    current_coverage, current_member_dates = current_constituent_price_coverage(
        clean_prices,
        constituents_path=constituents_path,
    )
    current_member_dates.write_csv(
        paths.run_dir(run_id) / "all_current_constituent_price_coverage.csv"
    )
    targeted_coverage = pl.read_csv(coverage_path, try_parse_dates=False)
    _write_price_coverage_html(
        paths.run_dir(run_id) / "current_constituent_price_coverage.html",
        run_id=run_id,
        published_snapshot=str(snapshot_dir),
        current_coverage=current_coverage,
        targeted_coverage=targeted_coverage,
    )
    manifest = {
        **package_manifest,
        "price_window": {
            "start_date": "2005-01-01",
            "end_date": date.today().isoformat(),
            "max_published_price_date": str(max_price_date),
        },
        "ticker_count": clean_prices.select(
            pl.col("ticker").n_unique()
        ).item(),
        "price_rows": clean_prices.height,
        "refresh_ticker_coverage": _json_safe_rows(targeted_coverage),
        "current_constituent_coverage": current_coverage,
        "published_output_snapshot": str(snapshot_dir.relative_to(paths.root_dir)),
        "published_output": {
            str(path.relative_to(paths.output_dir)): str(
                path.relative_to(paths.root_dir)
            )
            for path in paths.output_dir.rglob("*")
            if path.is_file()
        },
        "finalized_at": utc_now_iso(),
    }
    write_run_manifest(paths, run_id, manifest)
    print(f"Finalized run id: {run_id}")
    print(f"Output snapshot: {snapshot_dir}")
    print(f"Official manifest: {paths.latest_manifest_path}")
    print(
        "Current members: "
        f"{current_coverage['member_count']} covered, "
        f"{current_coverage['missing_price_count']} missing, "
        f"common through {current_coverage['latest_common_price_date']}"
    )


def _write_price_coverage_html(
    path: Path,
    *,
    run_id: str,
    published_snapshot: str,
    current_coverage: dict[str, object],
    targeted_coverage: pl.DataFrame,
) -> None:
    distribution_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['max_price_date']))}</td>"
        f"<td>{int(row['ticker_count'])}</td>"
        "</tr>"
        for row in current_coverage["max_date_distribution"]  # type: ignore[union-attr]
    )
    targeted_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['ticker']))}</td>"
        f"<td>{html.escape(str(row['min_date']))}</td>"
        f"<td>{html.escape(str(row['max_date']))}</td>"
        f"<td>{int(row['price_rows'])}</td>"
        "</tr>"
        for row in targeted_coverage.to_dicts()
    )
    path.write_text(
        f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit des prix des constituants courants</title><style>
body{{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:24px}}
main{{max-width:1050px;margin:auto}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #dde3ec;border-radius:14px;padding:18px;margin:14px 0}}
.metric{{font-size:28px;font-weight:700}} table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:9px;border-bottom:1px solid #e7ebf1;text-align:left}}
.warn{{background:#fff4d8;color:#704600;padding:12px;border-radius:8px}} code{{word-break:break-all}}
</style></head><body><main>
<h1>Audit des prix — univers S&amp;P 500 courant</h1>
<p><strong>Run :</strong> {html.escape(run_id)}<br>
<strong>Snapshot publié :</strong> <code>{html.escape(published_snapshot)}</code><br>
<strong>Mois des constituants :</strong> {html.escape(str(current_coverage['constituent_month']))}</p>
<div class="grid">
<div class="card"><div>Membres</div><div class="metric">{int(current_coverage['member_count'])}</div></div>
<div class="card"><div>Prix manquants</div><div class="metric">{int(current_coverage['missing_price_count'])}</div></div>
<div class="card"><div>Date commune</div><div class="metric">{html.escape(str(current_coverage['latest_common_price_date']))}</div></div>
<div class="card"><div>Date maximale</div><div class="metric">{html.escape(str(current_coverage['latest_any_price_date']))}</div></div>
</div>
<p class="warn">Une date de juillet confirme la fraîcheur des données, mais juillet reste un mois incomplet. La décision de production utilise le dernier mois calendaire terminé.</p>
<div class="card"><h2>Distribution de la dernière date par ticker</h2>
<table><thead><tr><th>Dernière date</th><th>Tickers</th></tr></thead><tbody>{distribution_rows}</tbody></table></div>
<div class="card"><h2>Tickers ajoutés ou renommés contrôlés</h2>
<table><thead><tr><th>Ticker</th><th>Début</th><th>Fin</th><th>Observations</th></tr></thead><tbody>{targeted_rows}</tbody></table></div>
</main></body></html>""",
        encoding="utf-8",
    )


def _json_safe_rows(frame: pl.DataFrame) -> list[dict[str, object]]:
    return [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in row.items()
        }
        for row in frame.to_dicts()
    ]


if __name__ == "__main__":
    main()
