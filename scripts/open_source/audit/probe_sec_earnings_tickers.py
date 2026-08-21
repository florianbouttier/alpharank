from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from alpharank.data.sources.earnings import (
    align_sec_actuals_to_calendar,
    build_sec_companyfacts_earnings_actuals,
    empty_earnings_actuals_frame,
    empty_earnings_calendar_frame,
)
from alpharank.data.ingestion.cadrage import _combine_sec_earnings_actuals
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir or (
        project_root / "outputs" / f"sec_earnings_ticker_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.start_year, args.end_year + 1))
    tickers = [_normalize_ticker(ticker) for ticker in args.tickers]
    ticker_roots = [ticker.removesuffix(".US") for ticker in tickers]

    cache_dir = project_root / "data" / "open_source" / "_cache"
    sec_client = SecCompanyFactsClient(
        user_agent=args.user_agent,
        cache_dir=cache_dir / "sec_companyfacts",
        max_retries=args.max_retries,
        request_pause_seconds=args.request_pause_seconds,
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=args.user_agent,
        cache_dir=None,
        max_retries=args.max_retries,
        request_pause_seconds=args.request_pause_seconds,
    )

    mapping = sec_client.fetch_company_mapping().filter(pl.col("ticker").is_in(ticker_roots))
    missing_from_mapping = sorted(set(ticker_roots) - set(mapping.get_column("ticker").to_list()))

    calendar_frames: list[pl.DataFrame] = []
    companyfacts_frames: list[pl.DataFrame] = []
    filing_frames: list[pl.DataFrame] = []
    errors: list[dict[str, str]] = []

    for row in mapping.select(["ticker", "cik"]).iter_rows(named=True):
        ticker_root = str(row["ticker"])
        ticker = f"{ticker_root}.US"
        cik = row["cik"]

        try:
            calendar = sec_filing_client.extract_earnings_calendar(ticker_root, cik, years)
            calendar_frames.append(_filter_years(calendar, args.start_year, args.end_year))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append({"ticker": ticker, "stage": "calendar", "error": str(exc)})

        try:
            facts_payload = sec_client.fetch_company_facts(cik)
            companyfacts = build_sec_companyfacts_earnings_actuals(ticker=ticker_root, facts_payload=facts_payload)
            companyfacts_frames.append(_filter_years(companyfacts, args.start_year, args.end_year))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append({"ticker": ticker, "stage": "companyfacts", "error": str(exc)})

        try:
            filing = sec_filing_client.extract_earnings_actuals(ticker_root, cik, years)
            filing_frames.append(_filter_years(filing, args.start_year, args.end_year))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append({"ticker": ticker, "stage": "filing", "error": str(exc)})

    calendar = _concat_or_empty(calendar_frames, empty_earnings_calendar_frame())
    companyfacts = _concat_or_empty(companyfacts_frames, empty_earnings_actuals_frame())
    filing = _concat_or_empty(filing_frames, empty_earnings_actuals_frame())
    combined = _combine_sec_earnings_actuals(sec_companyfacts=companyfacts, sec_filing=filing)
    aligned = align_sec_actuals_to_calendar(sec_calendar=calendar, sec_actuals=combined)

    probe = (
        calendar.select(["ticker", "period_end", "reportDate", "accession_number", "form", "fiscal_period", "fiscal_year"])
        .join(
            companyfacts.select(["ticker", "period_end", "reportDate", "epsActual", "source_label"]).rename(
                {
                    "reportDate": "companyfacts_reportDate",
                    "epsActual": "companyfacts_epsActual",
                    "source_label": "companyfacts_label",
                }
            ),
            on=["ticker", "period_end"],
            how="left",
        )
        .join(
            filing.select(["ticker", "period_end", "reportDate", "epsActual", "source_label"]).rename(
                {
                    "reportDate": "filing_reportDate",
                    "epsActual": "filing_epsActual",
                    "source_label": "filing_label",
                }
            ),
            on=["ticker", "period_end"],
            how="left",
        )
        .join(
            aligned.select(["ticker", "period_end", "reportDate", "epsActual", "source", "source_label"]).rename(
                {
                    "reportDate": "aligned_reportDate",
                    "epsActual": "aligned_epsActual",
                    "source": "aligned_source",
                    "source_label": "aligned_label",
                }
            ),
            on=["ticker", "period_end"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("aligned_epsActual").is_not_null())
            .then(pl.lit("covered"))
            .otherwise(pl.lit("missing"))
            .alias("status")
        )
        .sort(["ticker", "period_end"])
    )

    summary = (
        probe.group_by("ticker")
        .agg(
            [
                pl.len().alias("calendar_rows"),
                pl.col("aligned_epsActual").is_not_null().sum().alias("covered_rows"),
                pl.col("companyfacts_epsActual").is_not_null().sum().alias("companyfacts_rows"),
                pl.col("filing_epsActual").is_not_null().sum().alias("filing_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("calendar_rows") - pl.col("covered_rows")).alias("missing_rows"),
                ((pl.col("calendar_rows") - pl.col("covered_rows")) / pl.col("calendar_rows") * 100.0).alias("missing_pct"),
                (pl.col("covered_rows") / pl.col("calendar_rows") * 100.0).alias("coverage_pct"),
            ]
        )
        .sort(["missing_rows", "ticker"], descending=[True, False])
    )

    probe.write_parquet(output_dir / "probe.parquet")
    probe.write_csv(output_dir / "probe.csv")
    summary.write_parquet(output_dir / "summary.parquet")
    summary.write_csv(output_dir / "summary.csv")
    if errors:
        pl.DataFrame(errors).write_csv(output_dir / "errors.csv")

    (output_dir / "report.html").write_text(
        _render_html(
            tickers=tickers,
            start_year=args.start_year,
            end_year=args.end_year,
            summary=summary,
            probe=probe,
            errors=errors,
            missing_from_mapping=missing_from_mapping,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.txt").write_text(_render_text(summary=summary, errors=errors, missing_from_mapping=missing_from_mapping), encoding="utf-8")

    print(output_dir)
    print(summary)
    if errors:
        print(pl.DataFrame(errors))
    if missing_from_mapping:
        print(f"Missing from SEC mapping: {missing_from_mapping}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe SEC earnings coverage for selected tickers.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-pause-seconds", type=float, default=0.5)
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Florian Bouttier florianbouttier@example.com",
    )
    return parser.parse_args()


def _normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    return value if value.endswith(".US") else f"{value}.US"


def _filter_years(frame: pl.DataFrame, start_year: int, end_year: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(pl.col("period_end").str.slice(0, 4).cast(pl.Int64).is_between(start_year, end_year, closed="both"))


def _concat_or_empty(frames: list[pl.DataFrame], empty: pl.DataFrame) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return empty
    return pl.concat(non_empty, how="vertical")


def _render_text(*, summary: pl.DataFrame, errors: list[dict[str, str]], missing_from_mapping: list[str]) -> str:
    lines = ["# SEC earnings ticker probe", "", summary.write_csv()]
    if missing_from_mapping:
        lines.extend(["", "## Missing from SEC mapping", *[f"- {ticker}" for ticker in missing_from_mapping]])
    if errors:
        lines.extend(["", "## Errors", pl.DataFrame(errors).write_csv()])
    return "\n".join(lines)


def _render_html(
    *,
    tickers: list[str],
    start_year: int,
    end_year: int,
    summary: pl.DataFrame,
    probe: pl.DataFrame,
    errors: list[dict[str, str]],
    missing_from_mapping: list[str],
) -> str:
    summary_rows = "".join(
        f"<tr><td>{row['ticker']}</td><td>{row['calendar_rows']}</td><td>{row['covered_rows']}</td>"
        f"<td>{row['companyfacts_rows']}</td><td>{row['filing_rows']}</td><td>{row['missing_rows']}</td>"
        f"<td>{row['missing_pct']:.2f}%</td><td>{row['coverage_pct']:.2f}%</td></tr>"
        for row in summary.to_dicts()
    )

    missing_sections: list[str] = []
    for ticker in summary.filter(pl.col("missing_rows") > 0).get_column("ticker").to_list():
        rows = probe.filter((pl.col("ticker") == ticker) & (pl.col("status") == "missing")).to_dicts()
        row_html = "".join(
            f"<tr><td>{row['period_end']}</td><td>{row['reportDate']}</td><td>{row['form']}</td>"
            f"<td>{row.get('companyfacts_epsActual')}</td><td>{row.get('filing_epsActual')}</td>"
            f"<td>{row.get('aligned_epsActual')}</td><td>{row.get('companyfacts_label')}</td>"
            f"<td>{row.get('filing_label')}</td><td>{row.get('aligned_label')}</td></tr>"
            for row in rows
        )
        missing_sections.append(
            "<section>"
            f"<h2>{ticker}</h2>"
            "<table><thead><tr><th>period_end</th><th>reportDate</th><th>form</th><th>companyfacts</th>"
            "<th>filing</th><th>aligned</th><th>companyfacts_label</th><th>filing_label</th><th>aligned_label</th>"
            f"</tr></thead><tbody>{row_html}</tbody></table></section>"
        )

    error_html = ""
    if errors:
        error_rows = "".join(
            f"<tr><td>{row['ticker']}</td><td>{row['stage']}</td><td>{row['error']}</td></tr>"
            for row in errors
        )
        error_html = (
            "<section><h2>Errors</h2><table><thead><tr><th>Ticker</th><th>Stage</th><th>Error</th></tr></thead>"
            f"<tbody>{error_rows}</tbody></table></section>"
        )

    mapping_html = ""
    if missing_from_mapping:
        mapping_items = "".join(f"<li>{ticker}</li>" for ticker in missing_from_mapping)
        mapping_html = f"<section><h2>Missing from SEC mapping</h2><ul>{mapping_items}</ul></section>"

    return f"""
<html>
<head>
  <meta charset="utf-8">
  <title>SEC earnings ticker probe</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f4f4; }}
    code {{ background: #f6f6f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>SEC earnings ticker probe</h1>
  <p>Tickers: <code>{", ".join(tickers)}</code></p>
  <p>Years: <code>{start_year}</code> to <code>{end_year}</code></p>
  <section>
    <h2>Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>Calendar</th><th>Covered</th><th>Companyfacts</th><th>Filing</th><th>Missing</th><th>Missing %</th><th>Coverage %</th>
        </tr>
      </thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </section>
  {mapping_html}
  {error_html}
  <section>
    <h2>Missing details</h2>
    {"".join(missing_sections) if missing_sections else "<p>No missing rows in this probe.</p>"}
  </section>
</body>
</html>
"""


if __name__ == "__main__":
    main()
