from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

import polars as pl
import requests

from alpharank.data.open_source.config import METRIC_SPECS


class SecCompanyFactsClient:
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

    def fetch_company_mapping(self) -> pl.DataFrame:
        payload = self._get_json("https://www.sec.gov/files/company_tickers_exchange.json", cache_name="company_tickers_exchange.json")
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
        return self._get_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json",
            cache_name=f"CIK{cik_str}.json",
        )

    def extract_financials(self, ticker: str, cik: str | int) -> pl.DataFrame:
        payload = self.fetch_company_facts(cik)
        facts_payload = payload.get("facts", {})
        rows: list[dict[str, object]] = []

        for spec in METRIC_SPECS:
            selected = _select_best_facts(spec.statement, spec.sec_fact_roots, spec.sec_tags, facts_payload)
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

    def _get_json(self, url: str, cache_name: str) -> dict[str, Any]:
        cache_path = self.cache_dir / cache_name if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

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
                payload = response.json()
                if cache_path is not None:
                    cache_path.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(self.request_pause_seconds)
                return payload
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(30.0, 2.0 ** attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unable to fetch SEC payload from {url}")


def _select_best_facts(
    statement: str,
    fact_roots: Iterable[str],
    tags: Iterable[str],
    facts_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for tag_priority, tag in enumerate(tags):
        for fact_root in fact_roots:
            unit_payload = facts_payload.get(fact_root, {}).get(tag, {}).get("units", {})
            records = _extract_unit_records(unit_payload)
            cleaned = [_clean_fact(statement, tag, record, tag_priority=tag_priority) for record in records]
            candidates.extend(record for record in cleaned if record is not None)

    if statement == "shares":
        return _select_share_facts(candidates)
    if statement in {"income_statement", "cash_flow"}:
        return _select_duration_facts(candidates)
    return _select_instant_facts(candidates)


def _clean_fact(statement: str, tag: str, fact: dict[str, Any], *, tag_priority: int) -> dict[str, Any] | None:
    end = fact.get("end")
    filed = fact.get("filed")
    value = fact.get("val")
    if end is None or filed is None or value is None:
        return None

    form = str(fact.get("form", ""))
    if form and form not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
        return None

    start = fact.get("start")
    frame = fact.get("frame")
    calendarized_end = _calendarized_date_from_frame(frame) if statement == "shares" else None
    duration_days = None
    if statement in {"income_statement", "cash_flow"}:
        if start is None:
            return None
        try:
            duration_days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        except ValueError:
            return None

    return {
        "tag": tag,
        "tag_priority": tag_priority,
        "start": start,
        "end": calendarized_end or end,
        "raw_end": end,
        "filed": filed,
        "val": value,
        "fy": fact.get("fy"),
        "fp": fact.get("fp"),
        "form": form,
        "duration_days": duration_days,
        "frame": frame,
        "has_dimensions": bool(fact.get("has_dimensions")) or fact.get("segment") is not None,
    }


def _extract_unit_records(unit_payload: dict[str, Any]) -> list[dict[str, Any]]:
    for unit in ("USD", "shares", "pure"):
        records = unit_payload.get(unit, [])
        if records:
            return records
    for records in unit_payload.values():
        if records:
            return records
    return []


def _calendarized_date_from_frame(frame: object) -> str | None:
    if frame is None:
        return None
    match = re.fullmatch(r"CY(\d{4})Q([1-4])I?", str(frame))
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    quarter_end = {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31",
    }
    return quarter_end[quarter]


def _select_share_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    share_candidates = [
        record
        for record in candidates
        if record.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
        or _calendarized_date_from_frame(record.get("frame")) is not None
        or (record.get("fp") == "FY" and str(record.get("tag")) == "CommonStockSharesOutstanding")
    ]
    for record in sorted(
        share_candidates,
        key=lambda item: (
            item["filed"] or "",
            item["end"],
            item["tag_priority"],
        ),
    ):
        key = str(record.get("filed") or record["end"])
        current = chosen.get(key)
        if current is None or _is_better_share_record(record, current):
            chosen[key] = record
    return list(chosen.values())


def _share_sort_key(record: dict[str, Any]) -> tuple[int, int, int, str, str]:
    preferred_tag_rank = 0 if str(record.get("tag")) == "CommonStockSharesOutstanding" else 1
    return (
        preferred_tag_rank,
        int(bool(record.get("has_dimensions"))),
        int(record["tag_priority"]),
        str(record.get("filed") or ""),
        str(record["end"]),
    )


def _is_better_share_record(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_key = _share_sort_key(candidate)
    current_key = _share_sort_key(current)
    if candidate_key[:2] != current_key[:2]:
        return candidate_key[:2] < current_key[:2]
    return str(candidate["end"]) > str(current["end"])


def _select_duration_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quarterlies = [
        record
        for record in candidates
        if record.get("duration_days") is not None and 70 <= int(record["duration_days"]) <= 110 and record.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
    ]
    annuals = [
        record
        for record in candidates
        if record.get("duration_days") is not None and int(record["duration_days"]) >= 300 and record.get("fp") == "FY"
    ]

    chosen_by_end = _dedupe_by_end(quarterlies)
    derived_q4 = _derive_q4_facts(chosen_by_end.values(), annuals)
    for record in derived_q4:
        if record["end"] not in chosen_by_end:
            chosen_by_end[record["end"]] = record
    return list(chosen_by_end.values())


def _select_instant_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(_dedupe_by_end(candidates).values())


def _dedupe_by_end(candidates: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for record in sorted(
        candidates,
        key=lambda item: (
            item["end"],
            item["tag_priority"],
            item.get("filed") or "",
        ),
    ):
        end = str(record["end"])
        current = chosen.get(end)
        if current is None or _record_sort_key(record) < _record_sort_key(current):
            chosen[end] = record
    return chosen


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return (int(bool(record.get("has_dimensions"))), int(record["tag_priority"]), str(record.get("filed") or ""))


def _derive_q4_facts(quarterlies: Iterable[dict[str, Any]], annuals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    quarterly_records = list(quarterlies)
    quarterly_by_fy_fp = {
        (record.get("fy"), record.get("fp")): record
        for record in quarterly_records
        if record.get("fy") is not None and record.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
    }
    derived: list[dict[str, Any]] = []
    for annual in annuals:
        fy = annual.get("fy")
        if fy is None:
            continue
        q1 = quarterly_by_fy_fp.get((fy, "Q1"))
        q2 = quarterly_by_fy_fp.get((fy, "Q2"))
        q3 = quarterly_by_fy_fp.get((fy, "Q3"))
        q4 = quarterly_by_fy_fp.get((fy, "Q4"))
        if q4 is not None or q1 is None or q2 is None or q3 is None:
            continue
        derived.append(
            {
                **annual,
                "fp": "Q4",
                "val": float(annual["val"]) - float(q1["val"]) - float(q2["val"]) - float(q3["val"]),
                "tag": f"{annual['tag']}_derived_q4",
                "tag_priority": -1,
                "duration_days": 90,
            }
        )
    return derived


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
