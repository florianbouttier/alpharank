from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import random

import polars as pl

from alpharank.data.open_source.benchmark import load_sp500_tickers_for_year
from alpharank.data.open_source.consolidation import FinancialSourceInput, consolidate_financial_sources
from alpharank.data.open_source.earnings import (
    align_sec_actuals_to_calendar,
    build_sec_companyfacts_earnings_actuals,
    empty_earnings_actuals_frame,
    empty_earnings_calendar_frame,
)
from alpharank.data.open_source.pipeline import _combine_sec_earnings_actuals
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient


@dataclass(frozen=True)
class TickerSecHistory:
    companyfacts_financials: pl.DataFrame
    filing_financials: pl.DataFrame
    combined_financials: pl.DataFrame
    financial_lineage: pl.DataFrame
    calendar: pl.DataFrame
    companyfacts_earnings: pl.DataFrame
    filing_earnings: pl.DataFrame
    combined_earnings: pl.DataFrame


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (
        project_root / "outputs" / f"sec_sp500_history_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sp500_2025 = load_sp500_tickers_for_year(project_root / "data", 2025)
    sample = _pick_sample(
        universe=sp500_2025,
        explicit_tickers=args.tickers,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    years = list(range(args.start_year, args.end_year + 1))

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

    mapping = sec_client.fetch_company_mapping().filter(pl.col("ticker").is_in(sample))
    missing_from_mapping = sorted(set(sample) - set(mapping.get_column("ticker").to_list()))

    histories: dict[str, TickerSecHistory] = {}
    error_rows: list[dict[str, str]] = []
    rows = list(mapping.select(["ticker", "cik", "name"]).iter_rows(named=True))
    print(f"Processing {len(rows)} sampled tickers with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_ticker_history_worker,
                ticker_root=str(row["ticker"]),
                cik=row["cik"],
                years=years,
                user_agent=args.user_agent,
                cache_dir=cache_dir,
                max_retries=args.max_retries,
                request_pause_seconds=args.request_pause_seconds,
            ): str(row["ticker"])
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            ticker_root = futures[future]
            completed += 1
            try:
                histories[ticker_root] = future.result()
                print(
                    f"[{completed}/{len(rows)}] Completed {ticker_root}: financial_rows={histories[ticker_root].combined_financials.height}, "
                    f"earnings_rows={histories[ticker_root].combined_earnings.height}"
                )
            except Exception as exc:
                print(f"[{completed}/{len(rows)}] Failed {ticker_root}: {exc}")
                error_rows.append({"ticker": ticker_root, "stage": "ticker_history", "error": str(exc)})

    companyfacts_financials = _concat([item.companyfacts_financials for item in histories.values()], empty=_empty_financials())
    filing_financials = _concat([item.filing_financials for item in histories.values()], empty=_empty_financials())
    combined_financials = _concat([item.combined_financials for item in histories.values()], empty=_empty_consolidated_financials())
    financial_lineage = _concat([item.financial_lineage for item in histories.values()], empty=_empty_consolidated_financials())
    earnings_calendar = _concat([item.calendar for item in histories.values()], empty=empty_earnings_calendar_frame())
    companyfacts_earnings = _concat([item.companyfacts_earnings for item in histories.values()], empty=empty_earnings_actuals_frame())
    filing_earnings = _concat([item.filing_earnings for item in histories.values()], empty=empty_earnings_actuals_frame())
    combined_earnings = _concat([item.combined_earnings for item in histories.values()], empty=empty_earnings_actuals_frame())

    ticker_summary = _build_ticker_summary(sample=sample, mapping=mapping, histories=histories)
    statement_summary = _build_statement_summary(histories=histories)
    metric_summary = _build_metric_summary(histories=histories)
    earnings_summary = _build_earnings_summary(histories=histories)

    companyfacts_financials.write_parquet(output_dir / "sec_companyfacts_financials.parquet")
    filing_financials.write_parquet(output_dir / "sec_filing_financials.parquet")
    combined_financials.write_parquet(output_dir / "sec_combined_financials.parquet")
    financial_lineage.write_parquet(output_dir / "sec_combined_financial_lineage.parquet")
    earnings_calendar.write_parquet(output_dir / "sec_earnings_calendar.parquet")
    companyfacts_earnings.write_parquet(output_dir / "sec_companyfacts_earnings.parquet")
    filing_earnings.write_parquet(output_dir / "sec_filing_earnings.parquet")
    combined_earnings.write_parquet(output_dir / "sec_combined_earnings.parquet")

    ticker_summary.write_csv(output_dir / "ticker_summary.csv")
    statement_summary.write_csv(output_dir / "statement_summary.csv")
    metric_summary.write_csv(output_dir / "metric_summary.csv")
    earnings_summary.write_csv(output_dir / "earnings_summary.csv")
    ticker_summary.write_parquet(output_dir / "ticker_summary.parquet")
    statement_summary.write_parquet(output_dir / "statement_summary.parquet")
    metric_summary.write_parquet(output_dir / "metric_summary.parquet")
    earnings_summary.write_parquet(output_dir / "earnings_summary.parquet")

    if error_rows or missing_from_mapping:
        error_frame = pl.DataFrame(error_rows, schema={"ticker": pl.String, "stage": pl.String, "error": pl.String}) if error_rows else pl.DataFrame(
            schema={"ticker": pl.String, "stage": pl.String, "error": pl.String}
        )
        if missing_from_mapping:
            missing_frame = pl.DataFrame(
                [{"ticker": ticker, "stage": "mapping", "error": "missing from SEC company mapping"} for ticker in missing_from_mapping]
            )
            error_frame = _concat([error_frame, missing_frame], empty=pl.DataFrame(schema=error_frame.schema))
        error_frame.write_csv(output_dir / "errors.csv")
        error_frame.write_parquet(output_dir / "errors.parquet")

    (output_dir / "report.html").write_text(
        _render_html(
            sample=sample,
            ticker_summary=ticker_summary,
            statement_summary=statement_summary,
            metric_summary=metric_summary,
            earnings_summary=earnings_summary,
            errors=error_rows,
            missing_from_mapping=missing_from_mapping,
            start_year=args.start_year,
            end_year=args.end_year,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_markdown(
            sample=sample,
            ticker_summary=ticker_summary,
            statement_summary=statement_summary,
            earnings_summary=earnings_summary,
            start_year=args.start_year,
            end_year=args.end_year,
        ),
        encoding="utf-8",
    )

    print(output_dir)
    print(ticker_summary)
    print(statement_summary)
    print(earnings_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe SEC-only history reconstruction for a random S&P 500 2025 sample.")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-pause-seconds", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Florian Bouttier florianbouttier@example.com",
    )
    return parser.parse_args()


def _pick_sample(*, universe: tuple[str, ...], explicit_tickers: list[str] | None, sample_size: int, seed: int) -> list[str]:
    if explicit_tickers:
        return sorted({ticker.strip().upper().removesuffix(".US") for ticker in explicit_tickers})
    if len(universe) <= sample_size:
        return list(universe)
    return sorted(random.Random(seed).sample(list(universe), sample_size))


def _fetch_ticker_history(
    *,
    ticker_root: str,
    cik: str | int,
    years: list[int],
    sec_client: SecCompanyFactsClient,
    sec_filing_client: SecFilingFactsClient,
) -> TickerSecHistory:
    companyfacts_financials = sec_client.extract_financials(ticker_root, cik)
    companyfacts_payload = sec_client.fetch_company_facts(cik)
    companyfacts_earnings = build_sec_companyfacts_earnings_actuals(ticker=ticker_root, facts_payload=companyfacts_payload)

    filing_financial_frames: list[pl.DataFrame] = []
    for year in years:
        frame = sec_filing_client.extract_financials(ticker_root, cik, year)
        if not frame.is_empty():
            filing_financial_frames.append(frame)
    filing_financials = _concat(filing_financial_frames, empty=_empty_financials())

    combined_financials, lineage, _ = consolidate_financial_sources(
        [
            FinancialSourceInput(source_name="sec_companyfacts", frame=companyfacts_financials, priority=1),
            FinancialSourceInput(source_name="sec_filing", frame=filing_financials, priority=2),
        ]
    )

    calendar = sec_filing_client.extract_earnings_calendar(ticker_root, cik, years)
    filing_earnings = sec_filing_client.extract_earnings_actuals(ticker_root, cik, years)
    combined_earnings = _combine_sec_earnings_actuals(sec_companyfacts=companyfacts_earnings, sec_filing=filing_earnings)
    combined_earnings = align_sec_actuals_to_calendar(sec_calendar=calendar, sec_actuals=combined_earnings)

    return TickerSecHistory(
        companyfacts_financials=companyfacts_financials,
        filing_financials=filing_financials,
        combined_financials=combined_financials,
        financial_lineage=lineage,
        calendar=calendar,
        companyfacts_earnings=companyfacts_earnings,
        filing_earnings=filing_earnings,
        combined_earnings=combined_earnings,
    )


def _fetch_ticker_history_worker(
    *,
    ticker_root: str,
    cik: str | int,
    years: list[int],
    user_agent: str,
    cache_dir: Path,
    max_retries: int,
    request_pause_seconds: float,
) -> TickerSecHistory:
    sec_client = SecCompanyFactsClient(
        user_agent=user_agent,
        cache_dir=cache_dir / "sec_companyfacts",
        max_retries=max_retries,
        request_pause_seconds=request_pause_seconds,
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=user_agent,
        cache_dir=None,
        max_retries=max_retries,
        request_pause_seconds=request_pause_seconds,
    )
    return _fetch_ticker_history(
        ticker_root=ticker_root,
        cik=cik,
        years=years,
        sec_client=sec_client,
        sec_filing_client=sec_filing_client,
    )


def _build_ticker_summary(*, sample: list[str], mapping: pl.DataFrame, histories: dict[str, TickerSecHistory]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    name_map = {row["ticker"]: row["name"] for row in mapping.select(["ticker", "name"]).iter_rows(named=True)}
    cik_map = {row["ticker"]: row["cik"] for row in mapping.select(["ticker", "cik"]).iter_rows(named=True)}
    for ticker_root in sample:
        history = histories.get(ticker_root)
        if history is None:
            rows.append({"ticker": f"{ticker_root}.US", "name": name_map.get(ticker_root), "cik": cik_map.get(ticker_root), "status": "failed"})
            continue
        rows.append(
            {
                "ticker": f"{ticker_root}.US",
                "name": name_map.get(ticker_root),
                "cik": cik_map.get(ticker_root),
                "status": "ok",
                "companyfacts_financial_rows": history.companyfacts_financials.height,
                "filing_financial_rows": history.filing_financials.height,
                "combined_financial_rows": history.combined_financials.height,
                "companyfacts_financial_start": _min_value(history.companyfacts_financials, "date"),
                "companyfacts_financial_end": _max_value(history.companyfacts_financials, "date"),
                "combined_financial_start": _min_value(history.combined_financials, "date"),
                "combined_financial_end": _max_value(history.combined_financials, "date"),
                "earnings_calendar_rows": history.calendar.height,
                "combined_earnings_rows": history.combined_earnings.height,
                "earnings_start": _min_value(history.combined_earnings, "period_end"),
                "earnings_end": _max_value(history.combined_earnings, "period_end"),
                "earnings_coverage_pct": _earnings_coverage_pct(history.calendar, history.combined_earnings),
            }
        )
    return pl.DataFrame(rows).sort("ticker")


def _build_statement_summary(*, histories: dict[str, TickerSecHistory]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker_root, history in histories.items():
        ticker = f"{ticker_root}.US"
        for statement in ("income_statement", "balance_sheet", "cash_flow", "shares"):
            cf = history.companyfacts_financials.filter(pl.col("statement") == statement)
            filing = history.filing_financials.filter(pl.col("statement") == statement)
            combined = history.combined_financials.filter(pl.col("statement") == statement)
            rows.append(
                {
                    "ticker": ticker,
                    "statement": statement,
                    "companyfacts_rows": cf.height,
                    "filing_rows": filing.height,
                    "combined_rows": combined.height,
                    "metric_count": combined.select("metric").n_unique() if not combined.is_empty() else 0,
                    "companyfacts_start": _min_value(cf, "date"),
                    "companyfacts_end": _max_value(cf, "date"),
                    "filing_start": _min_value(filing, "date"),
                    "filing_end": _max_value(filing, "date"),
                    "combined_start": _min_value(combined, "date"),
                    "combined_end": _max_value(combined, "date"),
                    "candidate_sources": _candidate_sources(combined),
                }
            )
    return pl.DataFrame(rows).sort(["ticker", "statement"])


def _build_metric_summary(*, histories: dict[str, TickerSecHistory]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker_root, history in histories.items():
        ticker = f"{ticker_root}.US"
        combined = history.combined_financials
        if combined.is_empty():
            continue
        for group in combined.group_by(["statement", "metric"], maintain_order=True):
            key, frame = group
            statement = key[0]
            metric = key[1]
            rows.append(
                {
                    "ticker": ticker,
                    "statement": statement,
                    "metric": metric,
                    "rows": frame.height,
                    "start": _min_value(frame, "date"),
                    "end": _max_value(frame, "date"),
                    "selected_sources": " | ".join(sorted(set(frame.get_column("selected_source").drop_nulls().to_list()))),
                }
            )
    return pl.DataFrame(rows).sort(["ticker", "statement", "metric"])


def _build_earnings_summary(*, histories: dict[str, TickerSecHistory]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker_root, history in histories.items():
        ticker = f"{ticker_root}.US"
        rows.append(
            {
                "ticker": ticker,
                "calendar_rows": history.calendar.height,
                "companyfacts_actual_rows": history.companyfacts_earnings.height,
                "filing_actual_rows": history.filing_earnings.height,
                "combined_actual_rows": history.combined_earnings.height,
                "calendar_start": _min_value(history.calendar, "period_end"),
                "calendar_end": _max_value(history.calendar, "period_end"),
                "actual_start": _min_value(history.combined_earnings, "period_end"),
                "actual_end": _max_value(history.combined_earnings, "period_end"),
                "coverage_pct": _earnings_coverage_pct(history.calendar, history.combined_earnings),
            }
        )
    return pl.DataFrame(rows).sort("ticker")


def _earnings_coverage_pct(calendar: pl.DataFrame, actuals: pl.DataFrame) -> float | None:
    if calendar.is_empty():
        return None
    return (actuals.height / calendar.height) * 100.0


def _candidate_sources(frame: pl.DataFrame) -> str | None:
    if frame.is_empty() or "selected_source" not in frame.columns:
        return None
    values = sorted(set(frame.get_column("selected_source").drop_nulls().to_list()))
    return " | ".join(values) if values else None


def _min_value(frame: pl.DataFrame, column: str) -> str | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame.select(pl.col(column).min()).item()
    return None if value is None else str(value)


def _max_value(frame: pl.DataFrame, column: str) -> str | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame.select(pl.col(column).max()).item()
    return None if value is None else str(value)


def _concat(frames: list[pl.DataFrame], *, empty: pl.DataFrame) -> pl.DataFrame:
    usable = [frame for frame in frames if not frame.is_empty()]
    if not usable:
        return empty
    return pl.concat(usable, how="vertical")


def _empty_financials() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )


def _empty_consolidated_financials() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
            "fallback_used": pl.Boolean,
            "candidate_source_count": pl.Int64,
            "candidate_sources": pl.String,
            "candidate_source_labels": pl.String,
        }
    )


def _render_markdown(
    *,
    sample: list[str],
    ticker_summary: pl.DataFrame,
    statement_summary: pl.DataFrame,
    earnings_summary: pl.DataFrame,
    start_year: int,
    end_year: int,
) -> str:
    lines = [
        "# SEC-only S&P 500 History Sample",
        "",
        f"Sample (`seed=42` unless explicit): `{', '.join(sample)}`",
        f"Years requested from filings: `{start_year}` to `{end_year}`",
        "",
        "## Ticker summary",
    ]
    for row in ticker_summary.to_dicts():
        lines.append(
            f"- {row['ticker']}: financials `{row.get('combined_financial_start')}` -> `{row.get('combined_financial_end')}`, "
            f"rows `{row.get('combined_financial_rows')}`, earnings coverage `{_fmt_pct(row.get('earnings_coverage_pct'))}`"
        )
    lines.append("")
    lines.append("## Statement summary")
    for row in statement_summary.to_dicts():
        lines.append(
            f"- {row['ticker']} {row['statement']}: combined `{row['combined_rows']}` rows, "
            f"`{row.get('combined_start')}` -> `{row.get('combined_end')}`, sources `{row.get('candidate_sources')}`"
        )
    lines.append("")
    lines.append("## Earnings summary")
    for row in earnings_summary.to_dicts():
        lines.append(
            f"- {row['ticker']}: calendar `{row['calendar_rows']}`, actual `{row['combined_actual_rows']}`, coverage `{_fmt_pct(row.get('coverage_pct'))}`"
        )
    return "\n".join(lines)


def _render_html(
    *,
    sample: list[str],
    ticker_summary: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    earnings_summary: pl.DataFrame,
    errors: list[dict[str, str]],
    missing_from_mapping: list[str],
    start_year: int,
    end_year: int,
) -> str:
    ticker_rows = "".join(
        f"<tr><td>{row['ticker']}</td><td>{row.get('name')}</td><td>{row.get('status')}</td>"
        f"<td>{row.get('combined_financial_start')}</td><td>{row.get('combined_financial_end')}</td>"
        f"<td>{row.get('combined_financial_rows')}</td><td>{row.get('earnings_start')}</td>"
        f"<td>{row.get('earnings_end')}</td><td>{_fmt_pct(row.get('earnings_coverage_pct'))}</td></tr>"
        for row in ticker_summary.to_dicts()
    )
    statement_rows = "".join(
        f"<tr><td>{row['ticker']}</td><td>{row['statement']}</td><td>{row['companyfacts_rows']}</td><td>{row['filing_rows']}</td>"
        f"<td>{row['combined_rows']}</td><td>{row['metric_count']}</td><td>{row.get('combined_start')}</td><td>{row.get('combined_end')}</td><td>{row.get('candidate_sources')}</td></tr>"
        for row in statement_summary.to_dicts()
    )
    metric_rows = "".join(
        f"<tr><td>{row['ticker']}</td><td>{row['statement']}</td><td>{row['metric']}</td><td>{row['rows']}</td>"
        f"<td>{row.get('start')}</td><td>{row.get('end')}</td><td>{row.get('selected_sources')}</td></tr>"
        for row in metric_summary.to_dicts()
    )
    earnings_rows = "".join(
        f"<tr><td>{row['ticker']}</td><td>{row['calendar_rows']}</td><td>{row['companyfacts_actual_rows']}</td><td>{row['filing_actual_rows']}</td>"
        f"<td>{row['combined_actual_rows']}</td><td>{row.get('calendar_start')}</td><td>{row.get('calendar_end')}</td><td>{_fmt_pct(row.get('coverage_pct'))}</td></tr>"
        for row in earnings_summary.to_dicts()
    )
    error_html = ""
    if errors or missing_from_mapping:
        items = [
            *[f"<li>{row['ticker']}: {row['stage']} -> {row['error']}</li>" for row in errors],
            *[f"<li>{ticker}: missing from SEC mapping</li>" for ticker in missing_from_mapping],
        ]
        error_html = f"<section><h2>Errors</h2><ul>{''.join(items)}</ul></section>"

    return f"""
<html>
<head>
  <meta charset="utf-8">
  <title>SEC-only S&P 500 History Sample</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>SEC-only S&P 500 History Sample</h1>
  <p>Sample: <code>{', '.join(sample)}</code></p>
  <p>Filing years requested: <code>{start_year}</code> to <code>{end_year}</code></p>
  {error_html}
  <section>
    <h2>Ticker summary</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>Name</th><th>Status</th><th>Financial Start</th><th>Financial End</th><th>Financial Rows</th>
          <th>Earnings Start</th><th>Earnings End</th><th>Earnings Coverage</th>
        </tr>
      </thead>
      <tbody>{ticker_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Statement summary</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>Statement</th><th>Companyfacts Rows</th><th>Filing Rows</th><th>Combined Rows</th><th>Metrics</th>
          <th>Combined Start</th><th>Combined End</th><th>Sources</th>
        </tr>
      </thead>
      <tbody>{statement_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Metric summary</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>Statement</th><th>Metric</th><th>Rows</th><th>Start</th><th>End</th><th>Selected Sources</th>
        </tr>
      </thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Earnings summary</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>Calendar Rows</th><th>Companyfacts Actuals</th><th>Filing Actuals</th><th>Combined Actuals</th>
          <th>Calendar Start</th><th>Calendar End</th><th>Coverage</th>
        </tr>
      </thead>
      <tbody>{earnings_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


if __name__ == "__main__":
    main()
