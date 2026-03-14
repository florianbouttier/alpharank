from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET

import polars as pl
import requests

from alpharank.data.open_source.config import METRIC_SPECS
from alpharank.data.open_source.sec import (
    _derive_free_cash_flow,
    _empty_sec_frame,
    _select_best_facts,
)


@dataclass(frozen=True)
class FilingMetadata:
    accession_number: str
    filing_date: str
    report_date: str
    form: str
    primary_document: str


class SecFilingFactsClient:
    def __init__(
        self,
        user_agent: str,
        timeout: int = 30,
        cache_dir: Path | None = None,
        max_retries: int = 5,
        request_pause_seconds: float = 0.25,
    ) -> None:
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.request_pause_seconds = request_pause_seconds
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def extract_financials(self, ticker: str, cik: str | int, year: int) -> pl.DataFrame:
        filings = self._list_filings_for_year(cik, year)
        if not filings:
            return _empty_sec_frame()

        facts_payload = self._build_facts_payload(cik, filings)
        rows: list[dict[str, object]] = []
        for spec in METRIC_SPECS:
            if spec.metric == "free_cash_flow":
                continue
            selected = _select_best_facts(spec.statement, spec.sec_fact_roots, spec.sec_tags, facts_payload)
            if not selected:
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
                        "source": "sec_filing",
                        "source_label": fact["tag"],
                        "form": fact.get("form"),
                        "fiscal_period": fact.get("fp"),
                        "fiscal_year": fact.get("fy"),
                    }
                )

        frame = pl.DataFrame(rows) if rows else _empty_sec_frame()
        frame = _derive_missing_total_liabilities(frame)
        free_cash_flow = _derive_free_cash_flow(frame)
        if not free_cash_flow.is_empty():
            frame = pl.concat([frame, free_cash_flow.with_columns(pl.lit("sec_filing").alias("source"))], how="vertical")
        return frame.sort(["ticker", "statement", "metric", "date"])

    def _list_filings_for_year(self, cik: str | int, year: int) -> list[FilingMetadata]:
        payload = self._get_json(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
            cache_name=f"CIK{str(cik).zfill(10)}_submissions.json",
        )
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_documents = recent.get("primaryDocument", [])

        filings: list[FilingMetadata] = []
        for form, accession_number, filing_date, report_date, primary_document in zip(
            forms,
            accession_numbers,
            filing_dates,
            report_dates,
            primary_documents,
            strict=False,
        ):
            if form not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
                continue
            if not str(report_date).startswith(str(year)):
                continue
            filings.append(
                FilingMetadata(
                    accession_number=str(accession_number),
                    filing_date=str(filing_date),
                    report_date=str(report_date),
                    form=str(form),
                    primary_document=str(primary_document),
                )
            )
        return sorted(filings, key=lambda filing: (filing.report_date, filing.filing_date, filing.accession_number))

    def _build_facts_payload(self, cik: str | int, filings: list[FilingMetadata]) -> dict[str, Any]:
        payload: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}
        for filing in filings:
            for root_name, tag_name, unit_name, record in self._extract_filing_records(cik, filing):
                payload.setdefault(root_name, {}).setdefault(tag_name, {}).setdefault("units", {}).setdefault(unit_name, []).append(record)
        return payload

    def _extract_filing_records(self, cik: str | int, filing: FilingMetadata) -> list[tuple[str, str, str, dict[str, object]]]:
        instance_xml = self._get_instance_xml(cik, filing)
        root = ET.fromstring(instance_xml)
        contexts = _parse_contexts(root)
        units = _parse_units(root)
        filing_fy, filing_fp = _parse_document_focus(root)
        allowed_tags = {tag for spec in METRIC_SPECS for tag in spec.sec_tags}
        records: list[tuple[str, str, str, dict[str, object]]] = []
        for element in root.iter():
            context_ref = element.attrib.get("contextRef")
            if context_ref is None:
                continue
            local_name = _local_name(element.tag)
            namespace_uri = _namespace_uri(element.tag)
            fact_root = _fact_root_from_namespace(namespace_uri)
            if fact_root is None or local_name not in allowed_tags:
                continue
            value = _parse_numeric_value(element.text)
            if value is None:
                continue
            context = contexts.get(context_ref)
            if context is None:
                continue
            unit_name = units.get(element.attrib.get("unitRef"), "pure")
            if context["instant"] is not None:
                start = None
                end = context["instant"]
            else:
                start = context["start"]
                end = context["end"]
            if end is None:
                continue
            records.append(
                (
                    fact_root,
                    local_name,
                    unit_name,
                    {
                        "start": start,
                        "end": end,
                        "val": value,
                        "filed": filing.filing_date,
                        "form": filing.form,
                        "fy": filing_fy,
                        "fp": filing_fp,
                        "has_dimensions": context["has_dimensions"],
                    },
                )
            )
        return records

    def _get_instance_xml(self, cik: str | int, filing: FilingMetadata) -> str:
        accession_no_dashes = filing.accession_number.replace("-", "")
        folder = f"{int(cik):d}/{accession_no_dashes}"
        index_payload = self._get_json(
            f"https://www.sec.gov/Archives/edgar/data/{folder}/index.json",
            cache_name=f"{folder}_index.json".replace("/", "_"),
        )
        instance_name = _select_instance_name(index_payload)
        if instance_name is None:
            raise RuntimeError(f"No SEC XBRL instance found for {filing.accession_number}")
        return self._get_text(
            f"https://www.sec.gov/Archives/edgar/data/{folder}/{instance_name}",
            cache_name=f"{folder}_{instance_name}".replace("/", "_"),
        )

    def _get_json(self, url: str, cache_name: str) -> dict[str, Any]:
        cache_path = self.cache_dir / cache_name if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = self._request(url).json()
        if cache_path is not None:
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _get_text(self, url: str, cache_name: str) -> str:
        cache_path = self.cache_dir / cache_name if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        text = self._request(url).text
        if cache_path is not None:
            cache_path.write_text(text, encoding="utf-8")
        return text

    def _request(self, url: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_seconds = float(retry_after) if retry_after else min(30.0, 2.0 ** attempt)
                    time.sleep(sleep_seconds)
                    continue
                response.raise_for_status()
                time.sleep(self.request_pause_seconds)
                return response
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(30.0, 2.0 ** attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unable to fetch SEC payload from {url}")


def _parse_contexts(root: ET.Element) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    xbrli = "{http://www.xbrl.org/2003/instance}"
    xbrldi = "{http://xbrl.org/2006/xbrldi}"
    for context in root.findall(f"{xbrli}context"):
        context_id = context.attrib.get("id")
        if context_id is None:
            continue
        period = context.find(f"{xbrli}period")
        start = period.findtext(f"{xbrli}startDate") if period is not None else None
        end = period.findtext(f"{xbrli}endDate") if period is not None else None
        instant = period.findtext(f"{xbrli}instant") if period is not None else None
        has_dimensions = (
            context.find(f".//{xbrldi}explicitMember") is not None or context.find(f".//{xbrldi}typedMember") is not None
        )
        contexts[context_id] = {
            "start": start,
            "end": end,
            "instant": instant,
            "has_dimensions": has_dimensions,
        }
    return contexts


def _parse_units(root: ET.Element) -> dict[str, str]:
    units: dict[str, str] = {}
    xbrli = "{http://www.xbrl.org/2003/instance}"
    for unit in root.findall(f"{xbrli}unit"):
        unit_id = unit.attrib.get("id")
        if unit_id is None:
            continue
        measure = unit.findtext(f"{xbrli}measure")
        if measure is None:
            continue
        units[unit_id] = measure.rsplit(":", maxsplit=1)[-1]
    return units


def _parse_document_focus(root: ET.Element) -> tuple[int | None, str | None]:
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    for element in root:
        local_name = _local_name(element.tag)
        if local_name == "DocumentFiscalYearFocus":
            try:
                fiscal_year = int(str(element.text))
            except (TypeError, ValueError):
                fiscal_year = None
        elif local_name == "DocumentFiscalPeriodFocus":
            fiscal_period = str(element.text) if element.text is not None else None
    return fiscal_year, fiscal_period


def _select_instance_name(index_payload: dict[str, Any]) -> str | None:
    items = index_payload.get("directory", {}).get("item", [])
    preferred = [item.get("name") for item in items if str(item.get("name", "")).endswith("_htm.xml")]
    if preferred:
        return str(preferred[0])
    ignored_suffixes = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml")
    for item in items:
        name = str(item.get("name", ""))
        if not name.endswith(".xml"):
            continue
        if name == "FilingSummary.xml" or name.endswith(ignored_suffixes):
            continue
        return name
    return None


def _fact_root_from_namespace(namespace_uri: str | None) -> str | None:
    if namespace_uri is None:
        return None
    if "fasb.org/us-gaap" in namespace_uri:
        return "us-gaap"
    if "xbrl.sec.gov/dei" in namespace_uri:
        return "dei"
    if "xbrl.ifrs.org/taxonomy" in namespace_uri or "ifrs.org" in namespace_uri:
        return "ifrs-full"
    return None


def _parse_numeric_value(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _namespace_uri(tag: str) -> str | None:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return None


def _local_name(tag: str) -> str:
    return tag.split("}", maxsplit=1)[-1]


def _derive_missing_total_liabilities(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    base = frame.filter(pl.col("statement") == "balance_sheet")
    if base.is_empty():
        return frame

    liabilities = base.filter(pl.col("metric") == "total_liabilities")
    assets = base.filter(pl.col("metric") == "total_assets").rename(
        {"value": "total_assets", "source_label": "assets_source_label", "filing_date": "assets_filing_date"}
    )
    equity = base.filter(pl.col("metric") == "stockholders_equity").rename(
        {"value": "stockholders_equity", "source_label": "equity_source_label", "filing_date": "equity_filing_date"}
    )
    derived = (
        assets.join(
            equity.select(
                [
                    "ticker",
                    "statement",
                    "date",
                    "stockholders_equity",
                    "equity_source_label",
                    "equity_filing_date",
                    "form",
                    "fiscal_period",
                    "fiscal_year",
                ]
            ),
            on=["ticker", "statement", "date"],
            how="inner",
            suffix="_equity",
        )
        .join(liabilities.select(["ticker", "statement", "date"]).with_columns(pl.lit(True).alias("has_liabilities")), on=["ticker", "statement", "date"], how="left")
        .filter(pl.col("has_liabilities").fill_null(False).not_())
    )
    if derived.is_empty():
        return frame

    derived_rows = derived.select(
        [
            pl.col("ticker"),
            pl.col("statement"),
            pl.lit("total_liabilities").alias("metric"),
            pl.col("date"),
            pl.coalesce(["assets_filing_date", "equity_filing_date"]).alias("filing_date"),
            (pl.col("total_assets") - pl.col("stockholders_equity")).alias("value"),
            pl.lit("sec_filing").alias("source"),
            pl.lit("derived_from_assets_minus_stockholders_equity").alias("source_label"),
            pl.col("form"),
            pl.col("fiscal_period"),
            pl.col("fiscal_year"),
        ]
    )
    return pl.concat([frame, derived_rows], how="vertical").sort(["ticker", "statement", "metric", "date"])
