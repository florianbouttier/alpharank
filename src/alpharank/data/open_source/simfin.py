from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import polars as pl
import simfin as sf
from simfin.download import _maybe_download_dataset
from simfin.paths import _path_dataset

from alpharank.data.open_source.config import METRIC_SPECS, MetricSpec


class SimFinClient:
    """Polars-first SimFin adapter using the official bulk-download endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        data_dir: Path | None = None,
        refresh_days: int = 30,
    ) -> None:
        _load_local_dotenv()
        self.api_key = (api_key or os.getenv("SIMFIN_API_KEY", "")).strip()
        self.data_dir = data_dir or (Path.cwd() / "data" / "open_source" / "_cache" / "simfin")
        self.refresh_days = refresh_days
        self.enabled = bool(self.api_key)
        self._configured = False
        self.last_fetch_failures: list[dict[str, str]] = []

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
        self.last_fetch_failures = []
        ticker_set = set(tickers)
        dataset_frames = {
            "income_statement": self._load_dataset_frame_safe("income", ticker_set, year),
            "balance_sheet": self._load_dataset_frame_safe("balance", ticker_set, year),
            "cash_flow": self._load_dataset_frame_safe("cashflow", ticker_set, year),
            "derived": self._load_dataset_frame_safe("derived", ticker_set, year),
        }

        frames: list[pl.DataFrame] = []
        for spec in METRIC_SPECS:
            if spec.statement == "earnings" or not spec.simfin_datasets or not spec.simfin_columns:
                continue
            extracted = _extract_metric_frames(spec=spec, dataset_frames=dataset_frames, year=year)
            if not extracted.is_empty():
                frames.append(extracted)
        combined = pl.concat(frames, how="vertical").sort(["ticker", "statement", "metric", "date"]) if frames else _empty_financials()
        return _derive_missing_metrics(_standardize_financials(combined))

    def fetch_daily_prices(self, tickers: Iterable[str], start_date: str, end_date: str) -> pl.DataFrame:
        if not self.enabled:
            return _empty_prices()

        self.configure()
        self.last_fetch_failures = []
        try:
            return _load_shareprices_frame(
                tickers=tuple(tickers),
                start_date=start_date,
                end_date=end_date,
                refresh_days=self.refresh_days,
            )
        except Exception as exc:
            self.last_fetch_failures.append({"dataset": "shareprices_daily", "error": str(exc)})
            return _empty_prices()

    def _load_dataset_frame_safe(self, dataset: str, ticker_set: set[str], year: int) -> pl.DataFrame:
        try:
            return _load_dataset_frame(dataset, ticker_set, year, self.refresh_days)
        except Exception as exc:
            self.last_fetch_failures.append({"dataset": dataset, "error": str(exc)})
            return pl.DataFrame()


def _load_dataset_frame(dataset: str, ticker_set: set[str], year: int, refresh_days: int) -> pl.DataFrame:
    _maybe_download_dataset(refresh_days=refresh_days, dataset=dataset, market="us", variant="quarterly")
    path = Path(_path_dataset(dataset=dataset, market="us", variant="quarterly"))
    if not path.exists():
        return pl.DataFrame()

    frame = pl.read_csv(
        path,
        separator=";",
        try_parse_dates=True,
        null_values=["", "null", "None", "N/A"],
        infer_schema_length=10000,
    )
    if "Ticker" not in frame.columns:
        return pl.DataFrame()
    if "Report Date" not in frame.columns:
        return pl.DataFrame()
    return (
        frame.filter(pl.col("Ticker").is_in(list(ticker_set)))
        .with_columns(pl.col("Report Date").cast(pl.Utf8, strict=False).alias("_report_date_str"))
        .filter(pl.col("_report_date_str").str.starts_with(str(year)))
        .drop("_report_date_str")
    )


def _load_shareprices_frame(
    *,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    refresh_days: int,
) -> pl.DataFrame:
    _maybe_download_dataset(refresh_days=refresh_days, dataset="shareprices", market="us", variant="daily")
    path = Path(_path_dataset(dataset="shareprices", market="us", variant="daily"))
    if not path.exists() or not tickers:
        return _empty_prices()

    alias_lookup: dict[str, str] = {}
    for ticker in tickers:
        alias_lookup[ticker] = ticker
        dashed = ticker.replace(".", "-")
        alias_lookup[dashed] = ticker

    frame = (
        pl.scan_csv(
            path,
            separator=";",
            try_parse_dates=True,
            null_values=["", "null", "None", "N/A"],
            infer_schema_length=10000,
        )
        .filter(pl.col("Ticker").is_in(list(alias_lookup)))
        .with_columns(
            [
                pl.col("Ticker").replace(alias_lookup, default=None).alias("_ticker_root"),
                pl.col("Date").cast(pl.Utf8, strict=False).alias("_date"),
            ]
        )
        .filter(pl.col("_ticker_root").is_not_null())
        .filter(pl.col("_date").is_between(pl.lit(start_date), pl.lit(end_date)))
        .select(
            [
                (pl.col("_ticker_root") + pl.lit(".US")).alias("ticker"),
                pl.col("_date").alias("date"),
                pl.col("Open").cast(pl.Float64, strict=False).alias("open"),
                pl.col("High").cast(pl.Float64, strict=False).alias("high"),
                pl.col("Low").cast(pl.Float64, strict=False).alias("low"),
                pl.col("Close").cast(pl.Float64, strict=False).alias("close"),
                pl.col("Adj. Close").cast(pl.Float64, strict=False).alias("adjusted_close"),
                pl.col("Volume").cast(pl.Float64, strict=False).alias("volume"),
            ]
        )
        .collect()
    )
    if frame.is_empty():
        return _empty_prices()
    return frame.sort(["ticker", "date"])


def _extract_metric_frames(*, spec: MetricSpec, dataset_frames: dict[str, pl.DataFrame], year: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in spec.simfin_datasets:
        frame = dataset_frames.get(dataset_name)
        if frame is None or frame.is_empty():
            continue
        column_name = next((column for column in spec.simfin_columns if column in frame.columns), None)
        if column_name is None:
            continue
        for record in frame.select(_required_columns(frame, column_name)).iter_rows(named=True):
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


def _required_columns(frame: pl.DataFrame, metric_column: str) -> list[str]:
    optional = ["Publish Date", "Fiscal Period", "Fiscal Year"]
    columns = ["Ticker", "Report Date", metric_column]
    columns.extend(column for column in optional if column in frame.columns)
    return columns


def _format_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _empty_prices() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "adjusted_close": pl.Float64,
            "ticker": pl.String,
        }
    )


def _derive_missing_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    free_cash_flow = _derive_free_cash_flow(frame)
    if free_cash_flow.is_empty():
        return frame
    without_existing = _standardize_financials(
        frame.filter(~((pl.col("statement") == "cash_flow") & (pl.col("metric") == "free_cash_flow")))
    )
    return pl.concat([without_existing, free_cash_flow], how="vertical").sort(["ticker", "statement", "metric", "date"])


def _derive_free_cash_flow(frame: pl.DataFrame) -> pl.DataFrame:
    operating = (
        frame.filter((pl.col("statement") == "cash_flow") & (pl.col("metric") == "operating_cash_flow"))
        .rename(
            {
                "value": "operating_cash_flow",
                "filing_date": "operating_filing_date",
                "source_label": "operating_source_label",
            }
        )
        .select(["ticker", "date", "operating_cash_flow", "operating_filing_date", "operating_source_label", "fiscal_period", "fiscal_year"])
    )
    capex = (
        frame.filter((pl.col("statement") == "cash_flow") & (pl.col("metric") == "capital_expenditures"))
        .rename(
            {
                "value": "capital_expenditures",
                "filing_date": "capex_filing_date",
                "source_label": "capex_source_label",
            }
        )
        .select(["ticker", "date", "capital_expenditures", "capex_filing_date", "capex_source_label"])
    )
    joined = operating.join(capex, on=["ticker", "date"], how="inner")
    if joined.is_empty():
        return _empty_financials()
    derived = joined.select(
        [
            pl.col("ticker"),
            pl.lit("cash_flow").alias("statement"),
            pl.lit("free_cash_flow").alias("metric"),
            pl.col("date"),
            pl.coalesce([pl.col("operating_filing_date"), pl.col("capex_filing_date")]).alias("filing_date"),
            (pl.col("operating_cash_flow") - pl.col("capital_expenditures")).alias("value"),
            pl.lit("simfin").alias("source"),
            (
                pl.lit("derived:")
                + pl.col("operating_source_label")
                + pl.lit(" - ")
                + pl.col("capex_source_label")
            ).alias("source_label"),
            pl.lit(None).cast(pl.Utf8).alias("form"),
            pl.col("fiscal_period"),
            pl.col("fiscal_year"),
        ]
    )
    return _standardize_financials(derived.select(
        [
            pl.col("ticker").cast(pl.String),
            pl.col("statement").cast(pl.String),
            pl.col("metric").cast(pl.String),
            pl.col("date").cast(pl.String),
            pl.col("filing_date").cast(pl.String),
            pl.col("value").cast(pl.Float64),
            pl.col("source").cast(pl.String),
            pl.col("source_label").cast(pl.String),
            pl.col("form").cast(pl.String),
            pl.col("fiscal_period").cast(pl.String),
            pl.col("fiscal_year").cast(pl.Int64),
        ]
    ))


def _standardize_financials(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_financials()
    return frame.select(
        [
            pl.col("ticker").cast(pl.String),
            pl.col("statement").cast(pl.String),
            pl.col("metric").cast(pl.String),
            pl.col("date").cast(pl.String),
            pl.col("filing_date").cast(pl.String),
            pl.col("value").cast(pl.Float64),
            pl.col("source").cast(pl.String),
            pl.col("source_label").cast(pl.String),
            pl.col("form").cast(pl.String),
            pl.col("fiscal_period").cast(pl.String),
            pl.col("fiscal_year").cast(pl.Int64),
        ]
    )


def _load_local_dotenv() -> None:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        dotenv_path = candidate / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            return
    load_dotenv()
