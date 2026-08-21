#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime
import html
import json
from pathlib import Path
import tempfile

import polars as pl

from alpharank.data.sources.constituents import (
    load_constituent_change_registry,
    refresh_monthly_constituents,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONSTITUENTS = PROJECT_ROOT / "data" / "SP500_Constituents.csv"
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "data_quality" / "sp500_constituent_changes_2026.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend monthly S&P 500 membership using a versioned, sourced event registry."
    )
    parser.add_argument("--constituents", type=Path, default=DEFAULT_CONSTITUENTS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--target-month", default=None, help="Calendar month YYYY-MM-01; defaults to the current month.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_month = date.fromisoformat(args.target_month) if args.target_month else date.today().replace(day=1)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "data_refresh" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"
    html_dir.mkdir(exist_ok=True)

    source_sha256 = sha256_file(args.constituents)
    registry_sha256 = sha256_file(args.registry)
    registry = load_constituent_change_registry(args.registry)
    source = pl.read_csv(args.constituents, try_parse_dates=True)
    result = refresh_monthly_constituents(source, registry=registry, target_month=target_month)

    if not args.dry_run:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="sp500_constituents_",
            dir=args.constituents.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        try:
            result.frame.write_csv(temporary_path)
            temporary_path.replace(args.constituents)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    output_sha256 = sha256_file(args.constituents) if not args.dry_run else None
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "constituents_path": str(args.constituents.resolve()),
        "registry_path": str(args.registry.resolve()),
        "source_sha256": source_sha256,
        "registry_sha256": registry_sha256,
        "output_sha256": output_sha256,
        "base_month": result.base_month.isoformat(),
        "target_month": result.target_month.isoformat(),
        "snapshot_semantics": registry["snapshot_semantics"],
        "membership_event_lineage_contract": {
            "required_fields": [
                "event_id",
                "source_url",
                "observed_at",
                "effective_at",
                "effective_date",
                "confidence",
            ],
            "observation_time_policy": "known publication dates without a precise time use 23:59:59 America/New_York",
        },
        "monthly_summary": list(result.monthly_summary),
        "operation_audit": list(result.operation_audit),
    }
    (output_dir / "constituent_refresh_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_html_report(html_dir / "constituent_refresh_audit.html", manifest)

    print(f"Run id: {run_id}")
    print(f"Constituent source: {args.constituents}")
    print(f"Registry: {args.registry}")
    print(f"Target month: {result.target_month}")
    print(f"Rows: {result.frame.height}")
    for row in result.monthly_summary:
        print(f"  {row['month']}: {row['constituent_count']} constituents ({row['status']})")
    print(f"Manifest: {output_dir / 'constituent_refresh_manifest.json'}")
    print(f"HTML audit: {html_dir / 'constituent_refresh_audit.html'}")


def _write_html_report(path: Path, manifest: dict[str, object]) -> None:
    monthly_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['month']))}</td>"
        f"<td>{html.escape(str(row['constituent_count']))}</td>"
        f"<td>{html.escape(str(row['event_count']))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        "</tr>"
        for row in manifest["monthly_summary"]  # type: ignore[index]
    )
    operation_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(row['event_id']))}</code></td>"
        f"<td>{html.escape(str(row['observed_at']))}</td>"
        f"<td>{html.escape(str(row['effective_at']))}</td>"
        f"<td>{html.escape(str(row['effective_date']))}</td>"
        f"<td>{html.escape(str(row['snapshot_month']))}</td>"
        f"<td>{html.escape(str(row['action']))}</td>"
        f"<td>{html.escape(str(row['ticker']))}</td>"
        f"<td>{html.escape(str(row.get('new_ticker') or ''))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{html.escape(str(row['confidence']))}</td>"
        f"<td><a href=\"{html.escape(str(row['source_url']), quote=True)}\">source</a></td>"
        "</tr>"
        for row in manifest["operation_audit"]  # type: ignore[index]
    )
    path.write_text(
        f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit de fraîcheur des constituants S&amp;P 500</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:24px}}
main{{max-width:1180px;margin:auto}} .card{{background:white;border:1px solid #dde3ec;border-radius:14px;padding:20px;margin:16px 0}}
h1,h2{{margin-top:0}} table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{padding:9px;border-bottom:1px solid #e7ebf1;text-align:left}}
code{{word-break:break-all}} .warn{{color:#8a5200;background:#fff4d8;padding:12px;border-radius:8px}}
</style></head><body><main>
<h1>Audit de fraîcheur des constituants S&amp;P 500</h1>
<div class="card"><p><strong>Run :</strong> {html.escape(str(manifest['run_id']))}</p>
<p><strong>Sémantique :</strong> {html.escape(str(manifest['snapshot_semantics']))}</p>
<p><strong>Fenêtre reconstruite :</strong> {html.escape(str(manifest['base_month']))} → {html.escape(str(manifest['target_month']))}</p>
<p class="warn">Les deux no-op CASY/HOLX sont conservés dans l'audit : le snapshot hérité d'avril reflétait déjà ces deux côtés du remplacement. Aucun historique antérieur n'a été réécrit.</p>
<p><strong>SHA source :</strong> <code>{html.escape(str(manifest['source_sha256']))}</code><br>
<strong>SHA registre :</strong> <code>{html.escape(str(manifest['registry_sha256']))}</code><br>
<strong>SHA sortie :</strong> <code>{html.escape(str(manifest.get('output_sha256') or 'dry-run'))}</code></p></div>
<div class="card"><h2>Univers par mois</h2><table><thead><tr><th>Mois</th><th>Nombre</th><th>Événements</th><th>Statut</th></tr></thead><tbody>{monthly_rows}</tbody></table></div>
<div class="card"><h2>Journal des changements</h2><table><thead><tr><th>Événement</th><th>Observé</th><th>Effectif à</th><th>Date</th><th>Snapshot</th><th>Action</th><th>Ticker</th><th>Nouveau</th><th>Statut</th><th>Confiance</th><th>Justification</th></tr></thead><tbody>{operation_rows}</tbody></table></div>
</main></body></html>""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
