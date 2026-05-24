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

REVENUE_COMPONENT_INTEREST_TAGS: tuple[str, ...] = (
    "InterestAndDividendIncomeOperating",
    "InterestIncomeOperating",
    "InterestAndFeeIncomeLoansAndLeases",
)
REVENUE_COMPONENT_NONINTEREST_TAGS: tuple[str, ...] = (
    "NoninterestIncome",
    "NoninterestIncomeOtherOperatingIncome",
)


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

    def lookup_cik(self, ticker_root: str) -> int | None:
        """Look up CIK for a ticker, applying normalization if needed.

        The SEC ``company_tickers_exchange.json`` uses hyphens for share
        classes (e.g. ``BRK-B``) while many data vendors use dots
        (``BRK.B``). This method first tries the raw ticker, then
        applies known normalization overrides.
        """
        mapping = self.fetch_company_mapping()
        row = mapping.filter(pl.col("ticker") == ticker_root)
        if not row.is_empty():
            return row.select(pl.col("cik").cast(pl.Int64)).item()
        # Apply normalization overrides
        from alpharank.data.open_source.backfill import normalize_sec_ticker
        normalized = normalize_sec_ticker(ticker_root)
        if normalized != ticker_root:
            row = mapping.filter(pl.col("ticker") == normalized)
            if not row.is_empty():
                return row.select(pl.col("cik").cast(pl.Int64)).item()
        return None

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
            if spec.metric == "revenue":
                selected = _select_revenue_facts(spec.statement, spec.sec_fact_roots, spec.sec_tags, facts_payload)
            else:
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
                        "accession_number": fact.get("accession_number"),
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
    tag_list = tuple(tags)
    preferred_units: tuple[str, ...] = ()
    if statement == "shares":
        preferred_units = ("shares",)
    elif any("PerShare" in tag or "EarningsPerShare" in tag for tag in tag_list):
        preferred_units = ("USD/shares", "pure")

    for tag_priority, tag in enumerate(tag_list):
        for fact_root in fact_roots:
            unit_payload = facts_payload.get(fact_root, {}).get(tag, {}).get("units", {})
            records = _extract_unit_records(unit_payload, preferred_units=preferred_units)
            cleaned = [_clean_fact(statement, tag, record, tag_priority=tag_priority) for record in records]
            candidates.extend(record for record in cleaned if record is not None)

    if statement == "shares":
        return _select_share_facts(candidates)
    if statement in {"income_statement", "cash_flow"}:
        return _select_duration_facts(candidates)
    return _select_instant_facts(candidates)


def _select_revenue_facts(
    statement: str,
    fact_roots: Iterable[str],
    tags: Iterable[str],
    facts_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    direct = _select_best_facts(statement, fact_roots, tags, facts_payload)
    interest_component = _select_best_facts(
        statement,
        ("us-gaap",),
        REVENUE_COMPONENT_INTEREST_TAGS,
        facts_payload,
    )
    noninterest_component = _select_best_facts(
        statement,
        ("us-gaap",),
        REVENUE_COMPONENT_NONINTEREST_TAGS,
        facts_payload,
    )
    if not interest_component or not noninterest_component:
        return direct

    direct_by_end = {str(record["end"]): record for record in direct}
    interest_by_end = {str(record["end"]): record for record in interest_component}
    noninterest_by_end = {str(record["end"]): record for record in noninterest_component}
    derived: list[dict[str, Any]] = []
    for end, interest_record in interest_by_end.items():
        if end in direct_by_end:
            continue
        noninterest_record = noninterest_by_end.get(end)
        if noninterest_record is None:
            continue
        representative = interest_record
        derived.append(
            {
                **representative,
                "tag": f"{interest_record['tag']}_plus_{noninterest_record['tag']}_derived_revenue",
                "val": float(interest_record["val"]) + float(noninterest_record["val"]),
                "tag_priority": -2,
                "has_dimensions": bool(interest_record.get("has_dimensions")) or bool(noninterest_record.get("has_dimensions")),
            }
        )
    return sorted([*direct, *derived], key=lambda record: (str(record["end"]), int(record.get("tag_priority", 0))))


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
    frame_period = _quarter_from_frame(frame)
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
        "accession_number": fact.get("accession_number"),
        "val": value,
        "fy": fact.get("fy"),
        "fp": frame_period or fact.get("fp"),
        "form": form,
        "duration_days": duration_days,
        "frame": frame,
        "has_dimensions": bool(fact.get("has_dimensions")) or fact.get("segment") is not None,
        "dimensions": tuple(fact.get("dimensions", ())),
        "statement_class_member": fact.get("statement_class_member"),
    }


def _extract_unit_records(
    unit_payload: dict[str, Any],
    *,
    preferred_units: Iterable[str] = (),
) -> list[dict[str, Any]]:
    for unit in preferred_units:
        records = unit_payload.get(unit, [])
        if records:
            return records
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


def _quarter_from_frame(frame: object) -> str | None:
    if frame is None:
        return None
    match = re.fullmatch(r"CY(\d{4})Q([1-4])I?", str(frame))
    if not match:
        return None
    return f"Q{int(match.group(2))}"


def _select_share_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    share_candidates = [
        record
        for record in candidates
        if record.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
        or _calendarized_date_from_frame(record.get("frame")) is not None
        or (
            str(record.get("tag")) in {"EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"}
            and record.get("end") is not None
        )
    ]
    for record in share_candidates:
        key = (str(record.get("filed") or ""), str(record["end"]))
        grouped.setdefault(key, []).append(record)

    selected: list[dict[str, Any]] = []
    for _, records in sorted(grouped.items()):
        collapsed = _collapse_share_group(records)
        if collapsed is not None:
            selected.append(collapsed)
    return selected


def _collapse_share_group(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    total_candidates = [record for record in records if _is_total_share_record(record)]
    if total_candidates:
        return max(total_candidates, key=_share_record_priority)

    class_sum = _sum_share_class_records(records)
    if class_sum is not None:
        return class_sum

    return max(records, key=_share_record_priority) if records else None


def _is_total_share_record(record: dict[str, Any]) -> bool:
    dimensions = tuple(record.get("dimensions") or ())
    if not dimensions:
        return True
    non_class_dimensions = [
        (dimension, member)
        for dimension, member in dimensions
        if not str(dimension).endswith("StatementClassOfStockAxis")
    ]
    if not non_class_dimensions:
        return False
    return all(
        str(dimension).endswith("StatementEquityComponentsAxis") and str(member).endswith("CommonStockMember")
        for dimension, member in non_class_dimensions
    ) and record.get("statement_class_member") is None


def _sum_share_class_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    class_records = [record for record in records if _is_share_class_component(record)]
    if not class_records:
        return None

    best_by_member: dict[str, dict[str, Any]] = {}
    for record in class_records:
        member = _normalize_statement_class_member(record.get("statement_class_member"))
        current = best_by_member.get(member)
        if current is None or _share_record_priority(record) > _share_record_priority(current):
            best_by_member[member] = record

    if not best_by_member:
        return None

    representative = max(best_by_member.values(), key=_share_record_priority)
    summed_value = sum(float(record["val"]) for record in best_by_member.values())
    class_members = " | ".join(sorted(best_by_member))
    return {
        **representative,
        "tag": "SummedStatementClassOfStockAxisMembers",
        "val": summed_value,
        "has_dimensions": False,
        "dimensions": (),
        "statement_class_member": None,
        "class_member_count": len(best_by_member),
        "class_members": class_members,
    }


def _is_share_class_component(record: dict[str, Any]) -> bool:
    member = _normalize_statement_class_member(record.get("statement_class_member"))
    if member is None:
        return False
    dimensions = tuple(record.get("dimensions") or ())
    allowed_other_members = {"CommonStockMember"}
    for dimension, raw_member in dimensions:
        if str(dimension).endswith("StatementClassOfStockAxis"):
            continue
        member_name = _normalize_statement_class_member(raw_member)
        if str(dimension).endswith("StatementEquityComponentsAxis") and member_name in allowed_other_members:
            continue
        return False
    return True


def _normalize_statement_class_member(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    return raw.rsplit(":", maxsplit=1)[-1]


def _share_record_priority(record: dict[str, Any]) -> tuple[int, int, float, int, str, str]:
    return (
        int(_is_total_share_record(record)),
        int(record.get("tag") == "EntityCommonStockSharesOutstanding"),
        float(record.get("val") or 0.0),
        -int(bool(record.get("has_dimensions"))),
        str(record.get("filed") or ""),
        str(record.get("end") or ""),
    )


def _select_duration_facts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quarterly_direct = [
        record
        for record in candidates
        if record.get("duration_days") is not None and 70 <= int(record["duration_days"]) <= 130 and record.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
    ]
    cumulative_q2_q3 = [
        record
        for record in candidates
        if record.get("duration_days") is not None
        and (
            (record.get("fp") == "Q2" and 140 <= int(record["duration_days"]) <= 220)
            or (record.get("fp") == "Q3" and 230 <= int(record["duration_days"]) <= 320)
        )
    ]
    annuals = [
        record
        for record in candidates
        if record.get("duration_days") is not None and int(record["duration_days"]) >= 300 and record.get("fp") == "FY"
    ]

    direct_by_end = _dedupe_by_end(quarterly_direct)
    chosen_by_end: dict[str, dict[str, Any]] = dict(direct_by_end)

    direct_by_fy_fp = _dedupe_by_fy_fp(direct_by_end.values())
    cumulative_by_fy_fp = _dedupe_by_fy_fp(cumulative_q2_q3)
    annual_by_fy = _dedupe_annuals_by_fy(annuals)

    derived_from_cumulative = _derive_missing_quarters_from_cumulative(
        direct_by_fy_fp=direct_by_fy_fp,
        cumulative_by_fy_fp=cumulative_by_fy_fp,
    )
    for record in derived_from_cumulative:
        chosen_by_end.setdefault(str(record["end"]), record)

    q4_basis = {**direct_by_fy_fp, **_dedupe_by_fy_fp(derived_from_cumulative)}
    derived_q4 = _derive_q4_facts(q4_basis.values(), annual_by_fy.values())
    for record in derived_q4:
        chosen_by_end.setdefault(str(record["end"]), record)

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


def _dedupe_by_fy_fp(candidates: Iterable[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    chosen: dict[tuple[int, str], dict[str, Any]] = {}
    for record in sorted(
        candidates,
        key=lambda item: (
            int(item.get("fy") or -1),
            str(item.get("fp") or ""),
            _duration_bucket_priority(item),
            item.get("end") or "",
            item.get("filed") or "",
        ),
    ):
        fy = record.get("fy")
        fp = record.get("fp")
        if fy is None or fp not in {"Q1", "Q2", "Q3", "Q4"}:
            continue
        key = (int(fy), str(fp))
        current = chosen.get(key)
        if current is None or _fy_fp_record_sort_key(record) < _fy_fp_record_sort_key(current):
            chosen[key] = record
    return chosen


def _dedupe_annuals_by_fy(candidates: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    chosen: dict[int, dict[str, Any]] = {}
    for record in sorted(
        candidates,
        key=lambda item: (
            int(item.get("fy") or -1),
            _duration_bucket_priority(item),
            item.get("end") or "",
            item.get("filed") or "",
        ),
    ):
        fy = record.get("fy")
        if fy is None:
            continue
        key = int(fy)
        current = chosen.get(key)
        if current is None or _annual_record_sort_key(record) < _annual_record_sort_key(current):
            chosen[key] = record
    return chosen


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return (int(bool(record.get("has_dimensions"))), int(record["tag_priority"]), str(record.get("filed") or ""))


def _duration_bucket_priority(record: dict[str, Any]) -> tuple[int, int]:
    duration = int(record.get("duration_days") or 0)
    fp = str(record.get("fp") or "")
    if fp == "Q1":
        return (0, abs(duration - 90))
    if fp == "Q2":
        if duration <= 130:
            return (0, abs(duration - 91))
        return (1, abs(duration - 181))
    if fp == "Q3":
        if duration <= 130:
            return (0, abs(duration - 92))
        return (1, abs(duration - 273))
    if fp == "Q4":
        return (0, abs(duration - 91))
    return (9, duration)


def _fy_fp_record_sort_key(record: dict[str, Any]) -> tuple[int, int, int, str]:
    bucket_rank, bucket_distance = _duration_bucket_priority(record)
    return (
        bucket_rank,
        bucket_distance,
        int(bool(record.get("has_dimensions"))),
        str(record.get("filed") or ""),
    )


def _annual_record_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    duration = int(record.get("duration_days") or 0)
    return (abs(duration - 365), int(bool(record.get("has_dimensions"))), str(record.get("filed") or ""))


def _derive_missing_quarters_from_cumulative(
    *,
    direct_by_fy_fp: dict[tuple[int, str], dict[str, Any]],
    cumulative_by_fy_fp: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    fiscal_years = sorted({fy for fy, _ in direct_by_fy_fp} | {fy for fy, _ in cumulative_by_fy_fp})
    derived: list[dict[str, Any]] = []
    for fy in fiscal_years:
        quarter_records: dict[str, dict[str, Any] | None] = {
            "Q1": direct_by_fy_fp.get((fy, "Q1")),
            "Q2": direct_by_fy_fp.get((fy, "Q2")),
            "Q3": direct_by_fy_fp.get((fy, "Q3")),
        }
        q2_cumulative = cumulative_by_fy_fp.get((fy, "Q2"))
        q3_cumulative = cumulative_by_fy_fp.get((fy, "Q3"))
        changed = True
        while changed:
            changed = False

            if quarter_records["Q1"] is None:
                derived_q1 = _derive_q1_from_cumulative(
                    q2=quarter_records["Q2"],
                    q3=quarter_records["Q3"],
                    q2_cumulative=q2_cumulative,
                    q3_cumulative=q3_cumulative,
                )
                if derived_q1 is not None:
                    quarter_records["Q1"] = derived_q1
                    derived.append(derived_q1)
                    changed = True

            if quarter_records["Q2"] is None and quarter_records["Q1"] is not None and q2_cumulative is not None:
                derived_q2 = _derive_cumulative_quarter(
                    cumulative=q2_cumulative,
                    prior_quarters=[quarter_records["Q1"]],
                    quarter_label="Q2",
                )
                if derived_q2 is not None:
                    quarter_records["Q2"] = derived_q2
                    derived.append(derived_q2)
                    changed = True

            if (
                quarter_records["Q3"] is None
                and quarter_records["Q1"] is not None
                and quarter_records["Q2"] is not None
                and q3_cumulative is not None
            ):
                derived_q3 = _derive_cumulative_quarter(
                    cumulative=q3_cumulative,
                    prior_quarters=[quarter_records["Q1"], quarter_records["Q2"]],
                    quarter_label="Q3",
                )
                if derived_q3 is not None:
                    quarter_records["Q3"] = derived_q3
                    derived.append(derived_q3)
                    changed = True
    return derived


def _derive_q1_from_cumulative(
    *,
    q2: dict[str, Any] | None,
    q3: dict[str, Any] | None,
    q2_cumulative: dict[str, Any] | None,
    q3_cumulative: dict[str, Any] | None,
) -> dict[str, Any] | None:
    inferred_q1_end = _infer_prior_quarter_end(q2)
    if inferred_q1_end is None:
        return None
    if q2_cumulative is not None and q2 is not None and q2.get("val") is not None:
        return {
            **q2_cumulative,
            "fp": "Q1",
            "end": inferred_q1_end,
            "val": float(q2_cumulative["val"]) - float(q2["val"]),
            "tag": f"{q2_cumulative['tag']}_derived_q1",
            "tag_priority": -1,
            "duration_days": 90,
        }
    if q3_cumulative is not None and q2 is not None and q3 is not None:
        if q2.get("val") is None or q3.get("val") is None:
            return None
        return {
            **q3_cumulative,
            "fp": "Q1",
            "end": inferred_q1_end,
            "val": float(q3_cumulative["val"]) - float(q2["val"]) - float(q3["val"]),
            "tag": f"{q3_cumulative['tag']}_derived_q1",
            "tag_priority": -1,
            "duration_days": 90,
        }
    return None


def _infer_prior_quarter_end(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    start = record.get("start")
    if isinstance(start, str):
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d").date()
            return str(start_dt.fromordinal(start_dt.toordinal() - 1))
        except ValueError:
            pass
    end = record.get("end")
    if not isinstance(end, str):
        return None
    try:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return None
    month = end_dt.month - 3
    year = end_dt.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(end_dt.day, _days_in_month(year, month))
    return f"{year:04d}-{month:02d}-{day:02d}"


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1).date()
    else:
        next_month = datetime(year, month + 1, 1).date()
    this_month = datetime(year, month, 1).date()
    return (next_month - this_month).days


def _derive_cumulative_quarter(
    *,
    cumulative: dict[str, Any],
    prior_quarters: list[dict[str, Any]],
    quarter_label: str,
) -> dict[str, Any] | None:
    if any(record.get("val") is None for record in prior_quarters):
        return None
    derived_value = float(cumulative["val"]) - sum(float(record["val"]) for record in prior_quarters)
    return {
        **cumulative,
        "fp": quarter_label,
        "val": derived_value,
        "tag": f"{cumulative['tag']}_derived_{quarter_label.lower()}",
        "tag_priority": -1,
        "duration_days": 90,
    }


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
        inferred_end = _infer_next_quarter_end(q3)
        inferred_start = _infer_next_quarter_start(q3)
        derived.append(
            {
                **annual,
                "start": inferred_start or annual.get("start"),
                "end": inferred_end or annual.get("end"),
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
            pl.col("accession_number"),
            pl.col("form"),
            pl.col("fiscal_period"),
            pl.col("fiscal_year"),
        ]
    )


def _infer_next_quarter_end(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    end = record.get("end")
    if not isinstance(end, str):
        return None
    try:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return None
    month = end_dt.month + 3
    year = end_dt.year
    while month > 12:
        month -= 12
        year += 1
    source_month_end_day = _days_in_month(end_dt.year, end_dt.month)
    target_month_end_day = _days_in_month(year, month)
    if end_dt.day == source_month_end_day:
        day = target_month_end_day
    else:
        day = min(end_dt.day, target_month_end_day)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _infer_next_quarter_start(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    end = record.get("end")
    if not isinstance(end, str):
        return None
    try:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return None
    next_day = end_dt.fromordinal(end_dt.toordinal() + 1)
    return next_day.isoformat()


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
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )
