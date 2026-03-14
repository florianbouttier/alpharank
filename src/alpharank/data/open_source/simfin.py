from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import pandas as pd
import polars as pl
import simfin as sf

from alpharank.data.open_source.config import METRIC_SPECS, MetricSpec


class SimFinClient:
    """Thin adapter around SimFin's pandas-based API.

    The rest of the project remains Polars-first; pandas is only used at the
    vendor boundary because the SimFin client returns pandas DataFrames.
    """

    def __init__(
        self,
        api_key: str | None = None,
        data_dir: Path | None = None,
        refresh_days: int = 30,
    ) -> None:
        load_dotenv()
        self.api_key = (api_key or os.getenv("SIMFIN_API_KEY", "")).strip()
        self.data_dir = data_dir
        self.refresh_days = refresh_days
        self.enabled = bool(self.api_key)
        self._configured = False

    def configure(self) -> None:
        if self._configured or not self.enabled:
            return
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            try:
                sf.set_data_dir(str(self.data_dir))
            except FileExistsError:
                pass
        sf.set_api_key(self.api_key)
        self._configured = True

    def fetch_quarterly_financials(self, tickers: Iterable[str], year: int) -> pl.DataFrame:
        if not self.enabled:
            return _empty_financials()

        self.configure()
        ticker_set = set(tickers)
        dataset_frames = {
            "income_statement": sf.load_income(
                market="us",
                variant="quarterly",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                refresh_days=self.refresh_days,
            ),
            "balance_sheet": sf.load_balance(
                market="us",
                variant="quarterly",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                refresh_days=self.refresh_days,
            ),
            "cash_flow": sf.load_cashflow(
                market="us",
                variant="quarterly",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                refresh_days=self.refresh_days,
            ),
            "derived": sf.load_derived(
                market="us",
                variant="quarterly",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                refresh_days=self.refresh_days,
            ),
        }

        normalized = {
            dataset_name: _prepare_dataset_frame(frame, ticker_set)
            for dataset_name, frame in dataset_frames.items()
        }
        frames: list[pl.DataFrame] = []
        for spec in METRIC_SPECS:
            if spec.statement == "earnings" or not spec.simfin_datasets or not spec.simfin_columns:
                continue
            extracted = _extract_metric_frames(spec=spec, dataset_frames=normalized, year=year)
            if not extracted.is_empty():
                frames.append(extracted)
        return pl.concat(frames, how="vertical").sort(["ticker", "statement", "metric", "date"]) if frames else _empty_financials()


def _prepare_dataset_frame(frame: pd.DataFrame, ticker_set: set[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    prepared = frame.reset_index()
    if "Ticker" not in prepared.columns:
        return pd.DataFrame()
    prepared = prepared.loc[prepared["Ticker"].isin(ticker_set)].copy()
    return prepared


def _extract_metric_frames(*, spec: MetricSpec, dataset_frames: dict[str, pd.DataFrame], year: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in spec.simfin_datasets:
        frame = dataset_frames.get(dataset_name)
        if frame is None or frame.empty:
            continue
        column_name = next((column for column in spec.simfin_columns if column in frame.columns), None)
        if column_name is None:
            continue
        for record in frame.to_dict(orient="records"):
            report_date = _format_date(record.get("Report Date"))
            if report_date is None or not report_date.startswith(str(year)):
                continue
            value = _as_float(record.get(column_name))
            if value is None:
                continue
            if spec.metric == "capital_expenditures":
                value = abs(value)
            rows.append(
                {
                    "ticker": f"{record['Ticker']}.US",
                    "statement": spec.statement,
                    "metric": spec.metric,
                    "date": report_date,
                    "filing_date": _format_date(record.get("Publish Date")),
                    "value": value,
                    "source": "simfin",
                    "source_label": column_name,
                    "form": None,
                    "fiscal_period": _as_str(record.get("Fiscal Period")),
                    "fiscal_year": _as_int(record.get("Fiscal Year")),
                }
            )
        break
    return pl.DataFrame(rows) if rows else _empty_financials()


def _format_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _as_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _as_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _as_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


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
