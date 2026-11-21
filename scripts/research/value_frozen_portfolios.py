#!/usr/bin/env python3
"""Value sealed dashboard baskets from an explicitly archived Yahoo download."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import platform
import re
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio import ValuationWindow, value_frozen_holdings


def _window(text: str) -> ValuationWindow:
    name, start, end = text.split(":")
    return ValuationWindow(name, date.fromisoformat(start), date.fromisoformat(end))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--entry-date", type=date.fromisoformat, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--window", type=_window, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[dict, pl.DataFrame]:
    receipt = json.loads(args.raw_receipt.read_text(encoding="utf-8"))
    if receipt["source_name"] != "yfinance" or receipt["response_status"] != 200:
        raise ValueError("A successful yfinance receipt is required.")
    report_hash = sha256_path(args.source_report)
    if receipt["requested_scope"]["report_sha256"] != report_hash:
        raise ValueError("Source report hash differs from the download scope.")
    raw_root = args.raw_receipt.resolve().parent.parent
    raw_path = (raw_root / receipt["payload_object_path"]).resolve()
    if not raw_path.is_relative_to(raw_root) or sha256_path(raw_path) != receipt["payload_sha256"]:
        raise ValueError("Archived price payload path or SHA-256 is invalid.")
    match = re.search(
        r'const PAYLOAD_GZIP_BASE64="([^"]+)"', args.source_report.read_text(encoding="utf-8")
    )
    if match is None:
        raise ValueError("Source report does not contain the sealed performance payload.")
    payload = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    prices = pl.from_dicts(json.loads(raw_path.read_text(encoding="utf-8"))["prices"]).with_columns(
        pl.col("date").str.to_date()
    )
    if prices.filter(pl.col("date") > args.as_of).height:
        raise ValueError("Downloaded prices contain observations after the explicit as-of date.")
    if any(window.end_date > args.as_of for window in args.window):
        raise ValueError("Valuation window ends after the explicit as-of date.")
    args.report_hash = report_hash
    args.receipt = receipt
    return payload, prices


def _markdown_report(payload: dict, results: pl.DataFrame, args: argparse.Namespace) -> str:
    portfolio = payload["current_portfolio"]
    lines = [
        f"# Valorisation au {args.as_of.isoformat()} — paniers figés",
        "",
        f"Prix Yahoo nouvellement téléchargés ; dernière clôture : {results['end_date'].max()}.",
        f"Panier prévu pour {portfolio['holding_month']} ; entrée au close du {args.entry_date}.",
        "Aucun nouveau signal ni rééquilibrage : au-delà du mois prévu, maintien hypothétique.",
        "Rendements bruts de frais d'entrée/sortie, ajustés des opérations sur titres, sans change.",
        "Les poids dérivent depuis l'entrée ; aucun retour à l'équipondération entre les fenêtres.",
        "Les KPI historiques du rapport source ne sont ni prolongés ni recalculés.",
        "",
    ]
    for window in args.window:
        selected = results.filter(pl.col("window") == window.name)
        lines.extend(
            [
                f"## {window.name} — {window.start_date} → {window.end_date}",
                "",
                "| Stratégie | Performance | Écart SPY (points) |",
                "| --- | ---: | ---: |",
            ]
        )
        for strategy in payload["strategy_order"]:
            row = selected.filter(pl.col("strategy") == strategy)
            if strategy == "SPY · Total return":
                total = selected["benchmark_return"][0]
                excess = 0.0
            elif row.height == 1:
                total = row["total_return"][0]
                excess = row["excess_return"][0]
            else:
                raise ValueError(f"Expected one valuation for {strategy}, {window.name}.")
            lines.append(f"| {strategy} | {total:+.2%} | {100 * excess:+.2f} |")
        lines.append("")
    lines.extend(
        [
            "## Preuve",
            "",
            f"- Rapport source SHA-256 : `{args.report_hash}`.",
            f"- Prix archivés SHA-256 : `{args.receipt['payload_sha256']}`.",
            f"- Téléchargement UTC : `{args.receipt['retrieved_at']}`.",
            "- Identité des paniers vérifiée par hash avant/après ; aucun modèle réexécuté.",
            "- Ce contrôle n'est pas un replay inter-vintages ni une preuve de causalité des modèles.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    payload, prices = _load_inputs(args)
    original_baskets = json.dumps(payload["current_portfolio"], sort_keys=True).encode()
    basket_hash = hashlib.sha256(original_baskets).hexdigest()
    holdings = pl.from_dicts(payload["current_portfolio"]["holdings"], infer_schema_length=None)
    results = value_frozen_holdings(
        holdings, prices, entry_date=args.entry_date, windows=args.window
    )
    if json.dumps(payload["current_portfolio"], sort_keys=True).encode() != original_baskets:
        raise ValueError("Frozen source baskets changed during valuation.")
    if sha256_path(args.source_report) != args.report_hash:
        raise ValueError("Source report changed during valuation.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    holdings.write_parquet(args.output_dir / "frozen_holdings.parquet")
    prices.write_parquet(args.output_dir / "downloaded_prices.parquet")
    results.write_csv(args.output_dir / "valuation.csv")
    report = args.output_dir / "valuation_report.md"
    report.write_text(_markdown_report(payload, results, args), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated_valuation_only_not_promotable",
        "as_of_date": args.as_of.isoformat(),
        "entry_date": args.entry_date.isoformat(),
        "latest_common_close": results["end_date"].max().isoformat(),
        "source_report": {"path": str(args.source_report), "sha256": args.report_hash},
        "raw_receipt": {"path": str(args.raw_receipt), "sha256": sha256_path(args.raw_receipt)},
        "raw_payload_sha256": args.receipt["payload_sha256"],
        "retrieved_at": args.receipt["retrieved_at"],
        "frozen_baskets_sha256_before": basket_hash,
        "frozen_baskets_sha256_after": basket_hash,
        "frozen_holding_month": payload["current_portfolio"]["holding_month"],
        "new_signal_generated": False,
        "historical_model_replay": "not_rerun_valuation_only",
        "return_convention": "adjusted_close_buy_and_hold_gross_no_rebalance_no_fx",
        "code": {
            "python_version": platform.python_version(),
            "script_sha256": sha256_path(Path(__file__)),
            "valuation_sha256": sha256_path(
                Path(__file__).resolve().parents[2] / "src/alpharank/portfolio/valuation.py"
            ),
            "allocation_sha256": sha256_path(
                Path(__file__).resolve().parents[2] / "src/alpharank/portfolio/allocation.py"
            ),
        },
        "coverage": {
            "holdings": holdings.height,
            "strategies": holdings["strategy"].n_unique(),
            "price_tickers_including_spy": prices["ticker"].n_unique(),
            "price_rows": prices.height,
            "sessions": prices["date"].n_unique(),
        },
        "artifacts": {
            name: {"sha256": sha256_path(args.output_dir / name)}
            for name in (
                "frozen_holdings.parquet",
                "downloaded_prices.parquet",
                "valuation.csv",
                report.name,
            )
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(results.select("strategy", "window", "total_return", "benchmark_return"))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
