from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import polars as pl
import requests

from alpharank.data.open_source.config import METRIC_SPECS


class SecCompanyFactsClient:
    def __init__(self, user_agent: str, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def fetch_company_mapping(self) -> pl.DataFrame:
        response = self.session.get("https://www.sec.gov/files/company_tickers_exchange.json", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        fields = payload["fields"]
        return pl.DataFrame(payload["data"], schema=fields, orient="row").with_columns(
            [
                pl.col("cik").cast(pl.Int64, strict=False),
                pl.col("ticker").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8),
                pl.col("exchange").cast(pl.Utf8),
            ]
        )

    def fetch_company_facts(self, cik: str | int) -> dict[str, Any]:
        cik_str = str(cik).zfill(10)
        response = self.session.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def extract_financials(self, ticker: str, cik: str | int) -> pl.DataFrame:
        payload = self.fetch_company_facts(cik)
        facts = payload.get("facts", {}).get("us-gaap", {})
        rows: list[dict[str, object]] = []

        for spec in METRIC_SPECS:
            selected = _select_best_facts(spec.statement, spec.sec_tags, facts)
            if spec.metric == "free_cash_flow":
                continue
            for fact in selected:
                numeric_value = float(fact["val"])
                if spec.metric == "capital_expenditures":
                    numeric_value = abs(numeric_value)
                rows.append(
                    {
                        "ticker": f"{ticker}.US",
                        "statement": spec.statement,
                        "metric": spec.metric,
                        "date": fact["end"],
                        "filing_date": fact.get("filed"),
                        "value": numeric_value,
                        "source": "sec_companyfacts",
                        "source_label": fact["tag"],
                        "form": fact.get("form"),
                        "fiscal_period": fact.get("fp"),
                        "fiscal_year": fact.get("fy"),
                    }
                )

        if not rows:
            return _empty_sec_frame()

        frame = pl.DataFrame(rows)
        free_cash_flow = _derive_free_cash_flow(frame)
        if not free_cash_flow.is_empty():
            frame = pl.concat([frame, free_cash_flow], how="vertical")

        return frame.sort(["ticker", "statement", "metric", "date"])


def _select_best_facts(statement: str, tags: Iterable[str], facts: dict[str, Any]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for tag in tags:
        unit_payload = facts.get(tag, {}).get("units", {})
        records = unit_payload.get("USD", [])
        cleaned = [_clean_fact(statement, tag, record) for record in records]
        cleaned = [record for record in cleaned if record is not None]
        cleaned.sort(key=lambda record: record["filed"] or "", reverse=True)
        for record in cleaned:
            end = record["end"]
            if end not in chosen:
                chosen[end] = record
    return list(chosen.values())


def _clean_fact(statement: str, tag: str, fact: dict[str, Any]) -> dict[str, Any] | None:
    end = fact.get("end")
    filed = fact.get("filed")
    value = fact.get("val")
    if end is None or filed is None or value is None:
        return None

    form = str(fact.get("form", ""))
    if form and form not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
        return None

    start = fact.get("start")
    if statement in {"income_statement", "cash_flow"}:
        if start is None:
            return None
        try:
            duration = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        except ValueError:
            return None
        if duration < 70 or duration > 110:
            return None

    return {
        "tag": tag,
        "start": start,
        "end": end,
        "filed": filed,
        "val": value,
        "fy": fact.get("fy"),
        "fp": fact.get("fp"),
        "form": form,
    }


def _derive_free_cash_flow(frame: pl.DataFrame) -> pl.DataFrame:
    base = frame.filter(pl.col("statement") == "cash_flow")
    if base.is_empty():
        return _empty_sec_frame()

    operating = base.filter(pl.col("metric") == "operating_cash_flow").rename(
        {"value": "operating_cash_flow", "source_label": "operating_tag"}
    )
    capex = base.filter(pl.col("metric") == "capital_expenditures").rename(
        {"value": "capital_expenditures", "source_label": "capex_tag"}
    )
    joined = operating.join(
        capex.select(["ticker", "statement", "date", "capital_expenditures", "capex_tag"]),
        on=["ticker", "statement", "date"],
        how="inner",
    )
    if joined.is_empty():
        return _empty_sec_frame()

    return joined.select(
        [
            pl.col("ticker"),
            pl.col("statement"),
            pl.lit("free_cash_flow").alias("metric"),
            pl.col("date"),
            pl.col("filing_date"),
            (pl.col("operating_cash_flow") - pl.col("capital_expenditures")).alias("value"),
            pl.lit("sec_companyfacts").alias("source"),
            pl.lit("derived_from_operating_cash_flow_minus_capex").alias("source_label"),
            pl.col("form"),
            pl.col("fiscal_period"),
            pl.col("fiscal_year"),
        ]
    )


def _empty_sec_frame() -> pl.DataFrame:
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
