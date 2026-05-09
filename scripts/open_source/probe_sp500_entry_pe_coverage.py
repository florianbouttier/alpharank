from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import random

import pandas as pd
import polars as pl

from alpharank.data.processing import FundamentalProcessor, PricesDataPreprocessor


@dataclass(frozen=True)
class WindowSpec:
    label: str
    start: pd.Period
    end: pd.Period


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (
        project_root / "outputs" / f"sp500_entry_pe_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    constituents = pl.read_csv(project_root / "data" / "open_source" / "output" / "SP500_Constituents.csv", try_parse_dates=True)
    membership = _build_membership_table(constituents)

    sampled = _pick_sample(
        membership=membership,
        explicit_tickers=args.tickers,
        min_entry_date=date(args.min_entry_year, 1, 1),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    tickers = [f"{ticker}.US" for ticker in sampled]

    open_payload = _load_dataset(project_root / "data" / "open_source" / "output", tickers)
    eodhd_payload = _load_dataset(project_root / "data" / "eodhd" / "output", tickers)

    open_pe = _compute_pe_frame(**open_payload)
    eodhd_pe = _compute_pe_frame(**eodhd_payload)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for ticker_root in sampled:
        ticker = f"{ticker_root}.US"
        membership_row = membership.filter(pl.col("Ticker") == ticker_root).to_dicts()[0]
        first_month = pd.Period(membership_row["first_month"], freq="M")
        last_month = pd.Period(membership_row["last_month"], freq="M")

        earliest_price_open = _earliest_price_month(open_payload["final_price"], ticker)
        earliest_price_eodhd = _earliest_price_month(eodhd_payload["final_price"], ticker)
        listing_month = _min_period([earliest_price_open, earliest_price_eodhd])
        if listing_month is None:
            continue

        windows = _build_windows(first_month=first_month, last_month=last_month, listing_month=listing_month)
        open_ticker = open_pe[open_pe["ticker"] == ticker].copy()
        eodhd_ticker = eodhd_pe[eodhd_pe["ticker"] == ticker].copy()

        base = {
            "ticker": ticker,
            "sp500_entry_month": str(first_month),
            "sp500_last_month": str(last_month),
            "reference_listing_month": str(listing_month),
            "open_price_start": str(earliest_price_open) if earliest_price_open is not None else None,
            "eodhd_price_start": str(earliest_price_eodhd) if earliest_price_eodhd is not None else None,
        }

        for window in windows:
            open_metrics = _window_metrics(open_ticker, window)
            eodhd_metrics = _window_metrics(eodhd_ticker, window)
            summary_rows.append(
                {
                    **base,
                    "window": window.label,
                    "window_start": str(window.start),
                    "window_end": str(window.end),
                    "expected_months": open_metrics["expected_months"],
                    "open_any_pe_months": open_metrics["any_pe_months"],
                    "open_any_pe_pct": open_metrics["any_pe_pct"],
                    "open_legacy_usable_months": open_metrics["legacy_usable_months"],
                    "open_legacy_usable_pct": open_metrics["legacy_usable_pct"],
                    "eodhd_any_pe_months": eodhd_metrics["any_pe_months"],
                    "eodhd_any_pe_pct": eodhd_metrics["any_pe_pct"],
                    "eodhd_legacy_usable_months": eodhd_metrics["legacy_usable_months"],
                    "eodhd_legacy_usable_pct": eodhd_metrics["legacy_usable_pct"],
                    "open_missing_legacy_months": ", ".join(open_metrics["missing_legacy_months"]),
                    "eodhd_missing_legacy_months": ", ".join(eodhd_metrics["missing_legacy_months"]),
                }
            )

        for row in _per_month_detail(
            ticker=ticker,
            windows=windows,
            open_ticker=open_ticker,
            eodhd_ticker=eodhd_ticker,
        ):
            detail_rows.append(row)

    summary = pl.DataFrame(summary_rows).sort(["sp500_entry_month", "ticker", "window"])
    detail = pl.DataFrame(detail_rows).sort(["ticker", "window", "year_month"])

    summary.write_csv(output_dir / "summary.csv")
    detail.write_csv(output_dir / "detail.csv")
    summary.write_parquet(output_dir / "summary.parquet")
    detail.write_parquet(output_dir / "detail.parquet")
    (output_dir / "report.html").write_text(_render_html(summary=summary, detail=detail), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_markdown(summary=summary), encoding="utf-8")

    print(output_dir)
    print(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe PE coverage around S&P 500 entry for sampled tickers.")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-entry-year", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _build_membership_table(constituents: pl.DataFrame) -> pl.DataFrame:
    return (
        constituents.group_by("Ticker")
        .agg(
            [
                pl.col("Date").min().alias("first_month"),
                pl.col("Date").max().alias("last_month"),
            ]
        )
        .sort("Ticker")
    )


def _pick_sample(
    *,
    membership: pl.DataFrame,
    explicit_tickers: list[str] | None,
    min_entry_date: date,
    sample_size: int,
    seed: int,
) -> list[str]:
    if explicit_tickers:
        return sorted({ticker.strip().upper().removesuffix(".US") for ticker in explicit_tickers})

    eligible = (
        membership.filter(pl.col("first_month") >= min_entry_date)
        .sort("Ticker")
        .get_column("Ticker")
        .to_list()
    )
    if len(eligible) <= sample_size:
        return eligible
    return sorted(random.Random(seed).sample(eligible, sample_size))


def _load_dataset(data_dir: Path, tickers: list[str]) -> dict[str, pl.DataFrame]:
    return {
        "final_price": pl.read_parquet(data_dir / "US_Finalprice.parquet").filter(pl.col("ticker").is_in(tickers)),
        "balance": pl.read_parquet(data_dir / "US_Balance_sheet.parquet").filter(pl.col("ticker").is_in(tickers)),
        "income": pl.read_parquet(data_dir / "US_Income_statement.parquet").filter(pl.col("ticker").is_in(tickers)),
        "cashflow": pl.read_parquet(data_dir / "US_Cash_flow.parquet").filter(pl.col("ticker").is_in(tickers)),
        "earnings": pl.read_parquet(data_dir / "US_Earnings.parquet").filter(pl.col("ticker").is_in(tickers)),
    }


def _compute_pe_frame(
    *,
    final_price: pl.DataFrame,
    balance: pl.DataFrame,
    income: pl.DataFrame,
    cashflow: pl.DataFrame,
    earnings: pl.DataFrame,
) -> pd.DataFrame:
    monthly_return = PricesDataPreprocessor.calculate_monthly_returns(
        final_price,
        column_date="date",
        column_close="adjusted_close",
        backend="polars",
    )
    pe = FundamentalProcessor.calculate_pe_ratios(
        balance=balance,
        earnings=earnings,
        cashflow=cashflow,
        income=income,
        earning_choice="netincome_rolling",
        monthly_return=monthly_return,
        list_date_to_maximise=["filing_date_income", "filing_date_balance"],
        backend="polars",
    )
    pe = pe.copy()
    pe["year_month"] = pd.PeriodIndex(pe["year_month"], freq="M")
    pe["legacy_usable"] = (
        pe["pe"].notna()
        & pe["market_cap"].notna()
        & (pe["pe"] > 0)
        & (pe["pe"] < 100)
    )
    pe["has_any_pe"] = pe["pe"].notna()
    return pe


def _earliest_price_month(final_price: pl.DataFrame, ticker: str) -> pd.Period | None:
    subset = final_price.filter(pl.col("ticker") == ticker)
    if subset.is_empty():
        return None
    first_date = subset.select(pl.col("date").min()).item()
    if first_date is None:
        return None
    return pd.Period(str(first_date)[:7], freq="M")


def _min_period(values: list[pd.Period | None]) -> pd.Period | None:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    return min(non_null)


def _build_windows(*, first_month: pd.Period, last_month: pd.Period, listing_month: pd.Period) -> list[WindowSpec]:
    pre24_start = max(listing_month, first_month - 24)
    pre36_start = max(listing_month, first_month - 36)
    return [
        WindowSpec("pre_24m", pre24_start, first_month - 1),
        WindowSpec("pre_36m", pre36_start, first_month - 1),
        WindowSpec("in_sp500", first_month, last_month),
    ]


def _period_range(start: pd.Period, end: pd.Period) -> list[pd.Period]:
    if end < start:
        return []
    return list(pd.period_range(start=start, end=end, freq="M"))


def _window_metrics(frame: pd.DataFrame, window: WindowSpec) -> dict[str, object]:
    months = _period_range(window.start, window.end)
    expected = len(months)
    if expected == 0:
        return {
            "expected_months": 0,
            "any_pe_months": 0,
            "any_pe_pct": None,
            "legacy_usable_months": 0,
            "legacy_usable_pct": None,
            "missing_legacy_months": [],
        }

    indexed = frame.set_index("year_month") if not frame.empty else frame
    any_count = 0
    legacy_count = 0
    missing_legacy: list[str] = []
    for month in months:
        if frame.empty or month not in indexed.index:
            missing_legacy.append(str(month))
            continue
        row = indexed.loc[month]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if bool(row.get("has_any_pe", False)):
            any_count += 1
        if bool(row.get("legacy_usable", False)):
            legacy_count += 1
        else:
            missing_legacy.append(str(month))

    return {
        "expected_months": expected,
        "any_pe_months": any_count,
        "any_pe_pct": any_count / expected * 100.0,
        "legacy_usable_months": legacy_count,
        "legacy_usable_pct": legacy_count / expected * 100.0,
        "missing_legacy_months": missing_legacy,
    }


def _per_month_detail(
    *,
    ticker: str,
    windows: list[WindowSpec],
    open_ticker: pd.DataFrame,
    eodhd_ticker: pd.DataFrame,
) -> list[dict[str, object]]:
    open_indexed = open_ticker.set_index("year_month") if not open_ticker.empty else open_ticker
    eodhd_indexed = eodhd_ticker.set_index("year_month") if not eodhd_ticker.empty else eodhd_ticker
    rows: list[dict[str, object]] = []
    for window in windows:
        for month in _period_range(window.start, window.end):
            open_row = None if open_ticker.empty or month not in open_indexed.index else open_indexed.loc[month]
            eodhd_row = None if eodhd_ticker.empty or month not in eodhd_indexed.index else eodhd_indexed.loc[month]
            if isinstance(open_row, pd.DataFrame):
                open_row = open_row.iloc[-1]
            if isinstance(eodhd_row, pd.DataFrame):
                eodhd_row = eodhd_row.iloc[-1]
            rows.append(
                {
                    "ticker": ticker,
                    "window": window.label,
                    "year_month": str(month),
                    "open_pe": None if open_row is None else _safe_float(open_row.get("pe")),
                    "open_market_cap": None if open_row is None else _safe_float(open_row.get("market_cap")),
                    "open_legacy_usable": False if open_row is None else bool(open_row.get("legacy_usable", False)),
                    "eodhd_pe": None if eodhd_row is None else _safe_float(eodhd_row.get("pe")),
                    "eodhd_market_cap": None if eodhd_row is None else _safe_float(eodhd_row.get("market_cap")),
                    "eodhd_legacy_usable": False if eodhd_row is None else bool(eodhd_row.get("legacy_usable", False)),
                }
            )
    return rows


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _render_markdown(summary: pl.DataFrame) -> str:
    lines = ["# S&P 500 Entry PE Coverage", ""]
    for row in summary.to_dicts():
        open_pct = "n/a" if row["open_legacy_usable_pct"] is None else f"{row['open_legacy_usable_pct']:.1f}%"
        eodhd_pct = "n/a" if row["eodhd_legacy_usable_pct"] is None else f"{row['eodhd_legacy_usable_pct']:.1f}%"
        lines.append(
            f"- {row['ticker']} {row['window']}: open legacy {row['open_legacy_usable_months']}/{row['expected_months']} "
            f"({open_pct}), "
            f"eodhd legacy {row['eodhd_legacy_usable_months']}/{row['expected_months']} "
            f"({eodhd_pct})"
        )
    return "\n".join(lines)


def _render_html(*, summary: pl.DataFrame, detail: pl.DataFrame) -> str:
    summary_rows = "".join(
        f"<tr><td>{row['ticker']}</td><td>{row['sp500_entry_month']}</td><td>{row['window']}</td>"
        f"<td>{row['expected_months']}</td><td>{row['open_any_pe_months']}</td><td>{_fmt_pct(row['open_any_pe_pct'])}</td>"
        f"<td>{row['open_legacy_usable_months']}</td><td>{_fmt_pct(row['open_legacy_usable_pct'])}</td>"
        f"<td>{row['eodhd_any_pe_months']}</td><td>{_fmt_pct(row['eodhd_any_pe_pct'])}</td>"
        f"<td>{row['eodhd_legacy_usable_months']}</td><td>{_fmt_pct(row['eodhd_legacy_usable_pct'])}</td></tr>"
        for row in summary.to_dicts()
    )

    detail_sections: list[str] = []
    for ticker in detail.get_column("ticker").unique().to_list():
        rows = detail.filter(pl.col("ticker") == ticker).to_dicts()
        table_rows = "".join(
            f"<tr><td>{row['window']}</td><td>{row['year_month']}</td><td>{row['open_pe']}</td><td>{row['open_market_cap']}</td>"
            f"<td>{row['open_legacy_usable']}</td><td>{row['eodhd_pe']}</td><td>{row['eodhd_market_cap']}</td>"
            f"<td>{row['eodhd_legacy_usable']}</td></tr>"
            for row in rows
        )
        detail_sections.append(
            f"<h2>{ticker}</h2><table><thead><tr><th>Window</th><th>Month</th><th>Open PE</th><th>Open Market Cap</th>"
            f"<th>Open Legacy</th><th>EODHD PE</th><th>EODHD Market Cap</th><th>EODHD Legacy</th></tr></thead><tbody>{table_rows}</tbody></table>"
        )

    return f"""
<html>
<head>
  <meta charset="utf-8">
  <title>S&P 500 entry PE coverage</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f4f4; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>S&P 500 entry PE coverage</h1>
  <table>
    <thead>
      <tr>
        <th>Ticker</th><th>Entry</th><th>Window</th><th>Expected Months</th><th>Open Any PE</th><th>Open Any %</th>
        <th>Open Legacy</th><th>Open Legacy %</th><th>EODHD Any PE</th><th>EODHD Any %</th><th>EODHD Legacy</th><th>EODHD Legacy %</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>
  {''.join(detail_sections)}
</body>
</html>
"""


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


if __name__ == "__main__":
    main()
