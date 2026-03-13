from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.data.open_source.benchmark import (
    build_financial_alignment,
    build_price_alignment,
    load_eodhd_prices,
    normalize_eodhd_financials,
    summarize_alignment,
)
from alpharank.data.open_source.config import PILOT_TICKERS
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.yahoo import YahooFinanceClient


@dataclass(frozen=True)
class OpenSourceCadrageResult:
    output_dir: Path
    tickers: tuple[str, ...]
    price_rows: int
    sec_rows: int
    yfinance_financial_rows: int
    price_alignment_rows: int
    financial_alignment_rows: int


def run_open_source_cadrage(
    *,
    year: int = 2025,
    tickers: Sequence[str] | None = None,
    reference_data_dir: Path | None = None,
    output_dir: Path | None = None,
    user_agent: str = "Florian Bouttier florianbouttier@example.com",
) -> OpenSourceCadrageResult:
    project_root = Path(__file__).resolve().parents[4]
    reference_data_dir = reference_data_dir or project_root / "data"
    output_dir = output_dir or project_root / "data" / "open_source" / f"pilot_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_list = tuple(tickers or PILOT_TICKERS)
    yahoo_client = YahooFinanceClient()
    sec_client = SecCompanyFactsClient(user_agent=user_agent)

    sec_mapping = sec_client.fetch_company_mapping().filter(pl.col("ticker").is_in(ticker_list))
    general_reference = yahoo_client.fetch_general_reference(ticker_list, sec_mapping)
    yahoo_prices = yahoo_client.download_prices(ticker_list, f"{year}-01-01", f"{year + 1}-01-01")
    yahoo_earnings = yahoo_client.fetch_earnings_dates(ticker_list)
    yahoo_financials = yahoo_client.fetch_quarterly_financials(ticker_list).filter(pl.col("date").str.starts_with(f"{year}"))

    sec_frames = []
    for row in sec_mapping.select(["ticker", "cik"]).iter_rows(named=True):
        sec_frames.append(sec_client.extract_financials(str(row["ticker"]), str(row["cik"])))
    sec_financials = pl.concat(sec_frames, how="vertical") if sec_frames else _empty_financials()
    sec_financials = sec_financials.filter(pl.col("date").str.starts_with(f"{year}"))

    eodhd_prices = load_eodhd_prices(reference_data_dir, ticker_list, year)
    eodhd_financials = normalize_eodhd_financials(reference_data_dir, ticker_list, year)
    price_alignment = build_price_alignment(eodhd_prices, yahoo_prices)
    financial_alignment = pl.concat(
        [
            build_financial_alignment(eodhd_financials, sec_financials, "sec_companyfacts"),
            build_financial_alignment(eodhd_financials, yahoo_financials, "yfinance"),
        ],
        how="vertical",
    )

    _write_outputs(
        output_dir=output_dir,
        general_reference=general_reference,
        yahoo_prices=yahoo_prices,
        yahoo_earnings=yahoo_earnings,
        yahoo_financials=yahoo_financials,
        sec_financials=sec_financials,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        tickers=ticker_list,
        year=year,
    )

    return OpenSourceCadrageResult(
        output_dir=output_dir,
        tickers=ticker_list,
        price_rows=yahoo_prices.height,
        sec_rows=sec_financials.height,
        yfinance_financial_rows=yahoo_financials.height,
        price_alignment_rows=price_alignment.height,
        financial_alignment_rows=financial_alignment.height,
    )


def _write_outputs(
    *,
    output_dir: Path,
    general_reference: pl.DataFrame,
    yahoo_prices: pl.DataFrame,
    yahoo_earnings: pl.DataFrame,
    yahoo_financials: pl.DataFrame,
    sec_financials: pl.DataFrame,
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    tickers: tuple[str, ...],
    year: int,
) -> None:
    general_reference.write_parquet(output_dir / "general_reference.parquet")
    yahoo_prices.write_parquet(output_dir / "prices_yfinance.parquet")
    yahoo_earnings.write_parquet(output_dir / "earnings_yfinance.parquet")
    yahoo_financials.write_parquet(output_dir / "financials_yfinance.parquet")
    sec_financials.write_parquet(output_dir / "financials_sec_companyfacts.parquet")
    price_alignment.write_parquet(output_dir / f"price_alignment_{year}.parquet")
    financial_alignment.write_parquet(output_dir / f"financial_alignment_{year}.parquet")
    summarize_alignment(
        tickers=tickers,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        output_path=output_dir / "summary.json",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "year": year,
                "tickers": list(tickers),
                "files": {
                    "general_reference": "general_reference.parquet",
                    "prices_yfinance": "prices_yfinance.parquet",
                    "earnings_yfinance": "earnings_yfinance.parquet",
                    "financials_yfinance": "financials_yfinance.parquet",
                    "financials_sec_companyfacts": "financials_sec_companyfacts.parquet",
                    "price_alignment": f"price_alignment_{year}.parquet",
                    "financial_alignment": f"financial_alignment_{year}.parquet",
                    "summary": "summary.json",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
