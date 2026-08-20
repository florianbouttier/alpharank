from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl

from alpharank.data.open_source.earnings import empty_earnings_actuals_frame, empty_earnings_calendar_frame
from alpharank.data.open_source.earnings import align_sec_actuals_to_calendar
from alpharank.data.open_source.pipeline import (
    _combine_sec_earnings_actuals,
    _concat_or_empty,
    _fetch_sec_earnings_actuals,
    _fetch_sec_filing_earnings_actuals,
    _ticker_roots,
)
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or project_root / "outputs" / f"sec_earnings_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.start_year, args.end_year + 1))
    cache_dir = project_root / "data" / "open_source" / "_cache"
    sec_client = SecCompanyFactsClient(
        user_agent=args.user_agent,
        cache_dir=cache_dir / "sec_companyfacts",
        max_retries=8,
        request_pause_seconds=0.5,
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=args.user_agent,
        # Historical audits are read-heavy and can otherwise create a massive
        # on-disk filing cache. Keep the long-lived app cache for companyfacts,
        # but disable filing cache materialization for this coverage script.
        cache_dir=None,
        max_retries=8,
        request_pause_seconds=0.5,
    )
    sec_mapping_all = sec_client.fetch_company_mapping()
    universe = _load_universe(project_root=project_root, limit=args.limit)
    sec_mapping = sec_mapping_all.join(universe, on="ticker", how="inner")
    print(f"Universe tickers: {sec_mapping.height}")

    companyfacts_frames, companyfacts_failures = _fetch_sec_earnings_actuals(sec_client, sec_mapping)
    companyfacts = _concat_or_empty(companyfacts_frames, empty=empty_earnings_actuals_frame())
    companyfacts = _filter_period_range(companyfacts, args.start_year, args.end_year)
    print(f"Companyfacts actual rows: {companyfacts.height}")

    calendar_frames, calendar_failures = _fetch_sec_earnings_calendar_range(
        sec_client=sec_filing_client,
        sec_mapping=sec_mapping,
        years=years,
        max_workers=args.max_workers,
    )
    calendar = _concat_or_empty(calendar_frames, empty=empty_earnings_calendar_frame())
    calendar = _filter_period_range(calendar, args.start_year, args.end_year, date_column="period_end")
    print(f"Calendar rows: {calendar.height}")

    fallback_tickers = _identify_filing_gap_tickers(calendar=calendar, sec_companyfacts_actuals=companyfacts)
    print(f"Fallback tickers: {len(fallback_tickers)}")
    filing = empty_earnings_actuals_frame()
    filing_failures: list[dict[str, str]] = []
    if fallback_tickers:
        filing_mapping = sec_mapping.filter(pl.col("ticker").is_in(_ticker_roots(fallback_tickers)))
        filing_frames, filing_failures = _fetch_sec_filing_earnings_actuals_range(
            sec_client=sec_filing_client,
            sec_mapping=filing_mapping,
            years=years,
            max_workers=args.max_workers,
        )
        filing = _concat_or_empty(filing_frames, empty=empty_earnings_actuals_frame())
        filing = _filter_period_range(filing, args.start_year, args.end_year)
    print(f"Filing actual rows: {filing.height}")

    combined = _combine_sec_earnings_actuals(sec_companyfacts=companyfacts, sec_filing=filing)
    combined = align_sec_actuals_to_calendar(sec_calendar=calendar, sec_actuals=combined)

    expected = calendar.select(["ticker", "period_end"]).with_columns(pl.lit(True).alias("has_calendar"))
    actual = combined.select(["ticker", "period_end"]).with_columns(pl.lit(True).alias("has_sec_actual"))
    gap = (
        expected.join(actual, on=["ticker", "period_end"], how="left")
        .with_columns(pl.col("has_sec_actual").fill_null(False))
        .sort(["ticker", "period_end"])
    )

    yearly_summary = (
        gap.with_columns(pl.col("period_end").str.slice(0, 4).cast(pl.Int64).alias("year"))
        .group_by("year")
        .agg(
            [
                pl.len().alias("calendar_rows"),
                pl.col("has_sec_actual").sum().alias("sec_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("calendar_rows") - pl.col("sec_rows")).alias("missing_rows"),
                (pl.col("sec_rows") / pl.col("calendar_rows") * 100.0).alias("coverage_pct"),
            ]
        )
        .sort("year")
    )

    ticker_summary = (
        gap.group_by("ticker")
        .agg(
            [
                pl.len().alias("calendar_rows"),
                pl.col("has_sec_actual").sum().alias("sec_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("calendar_rows") - pl.col("sec_rows")).alias("missing_rows"),
                ((pl.col("calendar_rows") - pl.col("sec_rows")) / pl.col("calendar_rows") * 100.0).alias("missing_pct"),
                (pl.col("sec_rows") / pl.col("calendar_rows") * 100.0).alias("coverage_pct"),
            ]
        )
        .sort(["missing_rows", "ticker"], descending=[True, False])
    )

    global_summary = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "tickers": int(sec_mapping.height),
        "calendar_rows": int(calendar.height),
        "companyfacts_rows": int(companyfacts.height),
        "filing_rows": int(filing.height),
        "combined_rows": int(combined.height),
        "coverage_pct_rows": float(combined.height / calendar.height * 100.0) if calendar.height else None,
        "coverage_pct_tickers": float((ticker_summary.filter(pl.col("missing_rows") == 0).height / ticker_summary.height) * 100.0)
        if ticker_summary.height
        else None,
        "fallback_tickers": len(fallback_tickers),
        "companyfacts_failures": len(companyfacts_failures),
        "calendar_failures": len(calendar_failures),
        "filing_failures": len(filing_failures),
    }

    companyfacts.write_parquet(output_dir / "earnings_sec_companyfacts_actuals.parquet")
    filing.write_parquet(output_dir / "earnings_sec_filing_actuals.parquet")
    combined.write_parquet(output_dir / "earnings_sec_combined_actuals.parquet")
    calendar.write_parquet(output_dir / "earnings_sec_calendar.parquet")
    gap.write_parquet(output_dir / "earnings_sec_gap.parquet")
    yearly_summary.write_parquet(output_dir / "yearly_summary.parquet")
    ticker_summary.write_parquet(output_dir / "ticker_summary.parquet")
    yearly_summary.write_csv(output_dir / "yearly_summary.csv")
    ticker_summary.write_csv(output_dir / "ticker_summary.csv")
    (output_dir / "summary.txt").write_text(_render_summary_text(global_summary, yearly_summary, ticker_summary), encoding="utf-8")
    (output_dir / "report.html").write_text(_render_html(global_summary, yearly_summary, ticker_summary), encoding="utf-8")

    print(f"Coverage report written to {output_dir}")
    print(global_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SEC earnings EPS coverage across calendar/reporting periods.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Florian Bouttier florianbouttier@example.com",
    )
    return parser.parse_args()


def _load_universe(*, project_root: Path, limit: int | None) -> pl.DataFrame:
    lineage = pl.read_parquet(project_root / "data" / "open_source" / "output" / "lineage" / "earnings_open_source_lineage.parquet")
    universe = lineage.select(pl.col("ticker").str.replace(".US", "", literal=True).alias("ticker")).unique().sort("ticker")
    if limit is not None:
        universe = universe.head(limit)
    return universe


def _filter_period_range(frame: pl.DataFrame, start_year: int, end_year: int, *, date_column: str = "period_end") -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(
        pl.col(date_column).str.slice(0, 4).cast(pl.Int64).is_between(start_year, end_year, closed="both")
    )


def _fetch_sec_earnings_calendar_range(
    *,
    sec_client: SecFilingFactsClient,
    sec_mapping: pl.DataFrame,
    years: list[int],
    max_workers: int,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = list(sec_mapping.select(["ticker", "cik"]).iter_rows(named=True))
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    print(f"Fetching SEC earnings calendar for {len(rows)} tickers with {max_workers} workers")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_earnings_calendar, str(row["ticker"]), str(row["cik"]), years): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"SEC earnings calendar fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_calendar"})
            completed += 1
            if completed % 50 == 0 or completed == len(rows):
                print(f"Calendar progress: {completed}/{len(rows)}")
    return frames, failures


def _fetch_sec_filing_earnings_actuals_range(
    *,
    sec_client: SecFilingFactsClient,
    sec_mapping: pl.DataFrame,
    years: list[int],
    max_workers: int,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = list(sec_mapping.select(["ticker", "cik"]).iter_rows(named=True))
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    print(f"Fetching SEC filing earnings actuals for {len(rows)} tickers with {max_workers} workers")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_earnings_actuals, str(row["ticker"]), str(row["cik"]), years): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"SEC filing earnings actual fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_actuals"})
            completed += 1
            if completed % 25 == 0 or completed == len(rows):
                print(f"Filing progress: {completed}/{len(rows)}")
    return frames, failures


def _identify_filing_gap_tickers(*, calendar: pl.DataFrame, sec_companyfacts_actuals: pl.DataFrame) -> tuple[str, ...]:
    if calendar.is_empty():
        return ()
    expected = calendar.group_by("ticker").agg(pl.len().alias("calendar_rows"))
    actual = sec_companyfacts_actuals.group_by("ticker").agg(pl.len().alias("actual_rows"))
    coverage = (
        expected.join(actual, on="ticker", how="left")
        .with_columns(pl.col("actual_rows").fill_null(0))
        .filter(pl.col("calendar_rows") > pl.col("actual_rows"))
        .sort("ticker")
    )
    return tuple(coverage.get_column("ticker").to_list())


def _render_summary_text(global_summary: dict[str, object], yearly_summary: pl.DataFrame, ticker_summary: pl.DataFrame) -> str:
    top_missing = ticker_summary.head(25).select(["ticker", "missing_rows", "missing_pct", "coverage_pct"]).to_dicts()
    lines = [
        "# SEC Earnings Coverage",
        "",
        f"Period: {global_summary['start_year']} to {global_summary['end_year']}",
        f"Tickers: {global_summary['tickers']}",
        f"Calendar rows: {global_summary['calendar_rows']}",
        f"SEC rows: {global_summary['combined_rows']}",
        f"Coverage rows: {global_summary['coverage_pct_rows']:.2f}%" if global_summary["coverage_pct_rows"] is not None else "Coverage rows: n/a",
        "",
        "## Yearly coverage",
        yearly_summary.write_csv(),
        "",
        "## Top missing tickers",
    ]
    for row in top_missing:
        lines.append(
            f"- {row['ticker']}: missing={row['missing_rows']}, missing_pct={row['missing_pct']:.2f}%, coverage_pct={row['coverage_pct']:.2f}%"
        )
    return "\n".join(lines)


def _render_html(global_summary: dict[str, object], yearly_summary: pl.DataFrame, ticker_summary: pl.DataFrame) -> str:
    def table(df: pl.DataFrame, limit: int | None = None) -> str:
        if limit is not None:
            df = df.head(limit)
        cols = df.columns
        header = "".join(f"<th>{col}</th>" for col in cols)
        rows = []
        for row in df.iter_rows():
            rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    coverage = global_summary.get("coverage_pct_rows")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SEC Earnings Coverage</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .card {{ padding: 12px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>SEC Earnings Coverage Audit</h1>
  <div class="meta">
    <div class="card"><strong>Period</strong><br>{global_summary['start_year']} - {global_summary['end_year']}</div>
    <div class="card"><strong>Tickers</strong><br>{global_summary['tickers']}</div>
    <div class="card"><strong>Coverage</strong><br>{coverage:.2f}%</div>
  </div>
  <h2>Yearly Coverage</h2>
  {table(yearly_summary)}
  <h2>Top Missing Tickers</h2>
  {table(ticker_summary.select(['ticker', 'calendar_rows', 'sec_rows', 'missing_rows', 'missing_pct', 'coverage_pct']), limit=50)}
</body>
</html>"""


if __name__ == "__main__":
    main()
