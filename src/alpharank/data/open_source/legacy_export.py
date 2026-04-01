from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from alpharank.data.open_source.config import GENERAL_COLUMNS, METRIC_SPECS
from alpharank.data.open_source.storage import coerce_schema


LEGACY_STATEMENT_FILES = {
    "income_statement": "US_Income_statement.parquet",
    "balance_sheet": "US_Balance_sheet.parquet",
    "cash_flow": "US_Cash_flow.parquet",
    "shares": "US_share.parquet",
}
LEGACY_SHARE_SEMANTICS_FILE = "legacy_share_semantics.parquet"
LEGACY_SHARE_ALIGNMENT_MIN_RATIO = 0.8
LEGACY_SHARE_ALIGNMENT_MAX_RATIO = 1.2


def export_legacy_compatible_outputs(
    *,
    clean_prices: pl.DataFrame,
    benchmark_prices: pl.DataFrame,
    general_reference: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame | None = None,
    earnings_frame: pl.DataFrame,
    reference_data_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    price_path = output_dir / "US_Finalprice.parquet"
    clean_prices.select(["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"]).sort(
        ["ticker", "date"]
    ).write_parquet(price_path)
    exported["US_Finalprice.parquet"] = price_path

    benchmark_path = output_dir / "SP500Price.parquet"
    benchmark_prices.select(["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"]).sort(
        ["ticker", "date"]
    ).write_parquet(benchmark_path)
    exported["SP500Price.parquet"] = benchmark_path

    general_path = output_dir / "US_General.parquet"
    _build_general_legacy_frame(general_reference, reference_data_dir).write_parquet(general_path)
    exported["US_General.parquet"] = general_path

    financial_frames: dict[str, pl.DataFrame] = {}
    for statement in LEGACY_STATEMENT_FILES:
        financial_frames[statement] = _build_financial_legacy_frame(
            statement=statement,
            consolidated_financials=consolidated_financials,
            consolidated_lineage=consolidated_lineage,
            general_reference=general_reference,
            reference_data_dir=reference_data_dir,
        )

    financial_frames["balance_sheet"] = _merge_balance_shares(
        balance_frame=financial_frames["balance_sheet"],
        shares_frame=financial_frames["shares"],
    )
    earnings_legacy = _build_earnings_legacy_frame(earnings_frame, reference_data_dir)
    financial_frames["balance_sheet"], legacy_share_semantics = _align_balance_shares_with_earnings_semantics(
        balance_frame=financial_frames["balance_sheet"],
        income_frame=financial_frames["income_statement"],
        earnings_legacy_frame=earnings_legacy,
        earnings_consolidated_frame=earnings_frame,
    )

    for statement, file_name in LEGACY_STATEMENT_FILES.items():
        frame = financial_frames[statement]
        path = output_dir / file_name
        frame.write_parquet(path)
        exported[file_name] = path

    earnings_path = output_dir / "US_Earnings.parquet"
    earnings_legacy.write_parquet(earnings_path)
    exported["US_Earnings.parquet"] = earnings_path

    lineage_dir = output_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    legacy_share_semantics.write_parquet(lineage_dir / LEGACY_SHARE_SEMANTICS_FILE)
    return exported


def _build_general_legacy_frame(general_reference: pl.DataFrame, reference_data_dir: Path) -> pl.DataFrame:
    reference_schema = pl.read_parquet(reference_data_dir / "US_General.parquet").schema
    if general_reference.is_empty():
        return pl.DataFrame(schema=reference_schema)
    prepared = _ensure_general_reference_columns(general_reference)
    frame = prepared.select(GENERAL_COLUMNS).with_columns(
        [
            pl.col("ticker").str.replace(r"\.US$", "").alias("Code"),
            pl.col("name").alias("Name"),
            pl.col("exchange").alias("Exchange"),
            pl.lit("USD").alias("CurrencyCode"),
            pl.lit("$").alias("CurrencySymbol"),
            pl.lit("United States").alias("CountryName"),
            pl.lit("US").alias("CountryISO"),
            pl.col("cik").alias("CIK"),
            pl.col("Sector").alias("Sector"),
            pl.col("industry").alias("Industry"),
        ]
    )
    return coerce_schema(frame, reference_schema).sort("Code")


def _build_financial_legacy_frame(
    *,
    statement: str,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame | None,
    general_reference: pl.DataFrame,
    reference_data_dir: Path,
) -> pl.DataFrame:
    reference_path = reference_data_dir / LEGACY_STATEMENT_FILES[statement]
    reference_schema = pl.read_parquet(reference_path).schema
    source_frame = (
        consolidated_lineage
        if consolidated_lineage is not None and not consolidated_lineage.is_empty()
        else consolidated_financials
    )
    statement_frame = _ensure_financial_legacy_columns(source_frame).filter(pl.col("statement") == statement)
    if statement_frame.is_empty():
        return pl.DataFrame(schema=reference_schema)

    metric_map = {
        spec.metric: spec.eodhd_column
        for spec in METRIC_SPECS
        if spec.statement == statement and spec.statement != "earnings"
    }
    sector_frame = _build_general_sector_frame(general_reference)
    filing_date_reference = _build_legacy_filing_date_reference(source_frame)
    keyed = (
        statement_frame.filter(pl.col("metric").is_in(list(metric_map)))
        .join(sector_frame, on="ticker", how="left")
        .with_columns(
            [
                pl.col("metric").replace_strict(metric_map, default=None).alias("legacy_metric"),
                _normalize_statement_date_for_legacy(pl.col("date").cast(pl.Utf8, strict=False)).alias("legacy_date"),
                pl.col("date").alias("source_date"),
            ]
        )
        .filter(pl.col("legacy_metric").is_not_null())
    )
    if keyed.is_empty():
        return pl.DataFrame(schema=reference_schema)

    keyed = _collapse_duplicate_filing_dates(keyed)
    chosen_rows = _select_financial_legacy_candidates(keyed, statement=statement)
    values = (
        chosen_rows.pivot(index=["ticker", "date"], on="legacy_metric", values="value", aggregate_function="first")
    )
    filing_dates = chosen_rows.group_by(["ticker", "date"]).agg(pl.col("filing_date").drop_nulls().sort().last().alias("filing_date"))
    frame = (
        values.join(filing_dates, on=["ticker", "date"], how="left", coalesce=True)
        .join(
            filing_date_reference.rename({"legacy_date": "date"}),
            on=["ticker", "date"],
            how="left",
            coalesce=True,
        )
        .with_columns(pl.coalesce([pl.col("filing_date"), pl.col("fallback_filing_date")]).alias("filing_date"))
        .drop("fallback_filing_date", strict=False)
    )

    if statement == "shares":
        frame = frame.with_columns(
            [
                pl.col("date").alias("dateFormatted"),
                (pl.col("shares") / 1_000_000.0).alias("sharesMln"),
            ]
        )
    else:
        # Legacy valuation logic uses filing dates to determine when a statement becomes usable.
        # Rows that cannot be tied to any filing date leak fundamentals too early and distort history.
        frame = frame.filter(pl.col("filing_date").is_not_null())

    return coerce_schema(frame, reference_schema).sort(["ticker", "date"])


def _build_earnings_legacy_frame(earnings_frame: pl.DataFrame, reference_data_dir: Path) -> pl.DataFrame:
    reference_schema = pl.read_parquet(reference_data_dir / "US_Earnings.parquet").schema
    if earnings_frame.is_empty():
        return pl.DataFrame(schema=reference_schema)

    legacy_date = _normalize_earnings_period_end_for_legacy(
        period_end=pl.col("period_end").cast(pl.Utf8, strict=False),
        report_date=pl.col("reportDate").cast(pl.Utf8, strict=False),
    )
    frame = (
        earnings_frame.select(
            [
                pl.col("ticker"),
                legacy_date.alias("date"),
                pl.col("reportDate"),
                pl.col("earningsDatetime"),
                pl.col("epsEstimate"),
                pl.col("epsActual"),
                (pl.col("epsActual") - pl.col("epsEstimate")).alias("epsDifference"),
                pl.col("surprisePercent"),
            ]
        )
        .with_columns(
            [
                _classify_before_after_market(pl.col("earningsDatetime")).alias("beforeAfterMarket"),
                pl.lit(None).cast(pl.Utf8).alias("currency"),
            ]
        )
        .sort(["ticker", "reportDate"])
        .unique(subset=["ticker", "reportDate"], keep="last", maintain_order=True)
    )
    return coerce_schema(frame, reference_schema)


def _merge_balance_shares(*, balance_frame: pl.DataFrame, shares_frame: pl.DataFrame) -> pl.DataFrame:
    if balance_frame.is_empty() or shares_frame.is_empty():
        return balance_frame
    if "commonStockSharesOutstanding" not in balance_frame.columns or "shares" not in shares_frame.columns:
        return balance_frame

    share_lookup = (
        shares_frame.select(
            [
                pl.col("ticker"),
                pl.col("dateFormatted").cast(pl.Utf8).alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares_from_legacy_share"),
            ]
        )
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
    )

    joined = balance_frame.join(share_lookup, on=["ticker", "date"], how="left")
    enriched = joined.with_columns(
        pl.coalesce(
            [
                pl.col("commonStockSharesOutstanding").cast(pl.Float64, strict=False),
                pl.col("shares_from_legacy_share"),
            ]
        ).alias("commonStockSharesOutstanding")
    )
    return enriched.drop("shares_from_legacy_share")


def _align_balance_shares_with_earnings_semantics(
    *,
    balance_frame: pl.DataFrame,
    income_frame: pl.DataFrame,
    earnings_legacy_frame: pl.DataFrame,
    earnings_consolidated_frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    lineage_schema = _empty_legacy_share_semantics_frame().schema
    required_balance_cols = {"ticker", "date", "commonStockSharesOutstanding"}
    required_income_cols = {"ticker", "date", "netIncome"}
    required_earnings_cols = {"ticker", "date", "epsActual"}

    if (
        balance_frame.is_empty()
        or income_frame.is_empty()
        or earnings_legacy_frame.is_empty()
        or not required_balance_cols.issubset(balance_frame.columns)
        or not required_income_cols.issubset(income_frame.columns)
        or not required_earnings_cols.issubset(earnings_legacy_frame.columns)
    ):
        return balance_frame, pl.DataFrame(schema=lineage_schema)

    balance_lookup = balance_frame.select(
        [
            pl.col("ticker"),
            pl.col("date"),
            pl.col("filing_date"),
            pl.col("commonStockSharesOutstanding").cast(pl.Float64, strict=False).alias("reported_commonStockSharesOutstanding"),
        ]
    )
    income_lookup = income_frame.select(
        [
            pl.col("ticker"),
            pl.col("date"),
            pl.col("netIncome").cast(pl.Float64, strict=False).alias("netIncome"),
        ]
    )
    earnings_lookup = earnings_legacy_frame.select(
        [
            pl.col("ticker"),
            pl.col("date"),
            pl.col("reportDate"),
            pl.col("epsActual").cast(pl.Float64, strict=False).alias("epsActual"),
        ]
    )

    earnings_source_columns = [
        column
        for column in [
            "selected_source",
            "actual_source",
            "estimate_source",
            "surprise_source",
            "source_label",
            "candidate_sources",
        ]
        if column in earnings_consolidated_frame.columns
    ]
    earnings_source_lookup = (
        earnings_consolidated_frame.select(
            [
                pl.col("ticker"),
                _normalize_earnings_period_end_for_legacy(
                    period_end=pl.col("period_end").cast(pl.Utf8, strict=False),
                    report_date=pl.col("reportDate").cast(pl.Utf8, strict=False),
                ).alias("date"),
                pl.col("reportDate"),
                *[pl.col(column) for column in earnings_source_columns],
            ]
        )
        .sort(["ticker", "date", "reportDate"])
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
    )

    semantics = (
        balance_lookup.join(income_lookup, on=["ticker", "date"], how="left")
        .join(earnings_lookup, on=["ticker", "date"], how="left", coalesce=True)
        .join(earnings_source_lookup, on=["ticker", "date", "reportDate"], how="left", coalesce=True)
        .with_columns(
            [
                pl.when(pl.col("netIncome").abs() > 0)
                .then(
                    pl.when(pl.col("epsActual").abs() > 1e-9)
                    .then(pl.col("netIncome").abs() / pl.col("epsActual").abs())
                    .otherwise(pl.lit(None).cast(pl.Float64))
                )
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("earnings_implied_commonStockSharesOutstanding"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("earnings_implied_commonStockSharesOutstanding") / pl.col("reported_commonStockSharesOutstanding")
                ).alias("implied_to_reported_ratio"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("reported_commonStockSharesOutstanding").is_not_null()
                    & (pl.col("reported_commonStockSharesOutstanding") > 0)
                    & pl.col("earnings_implied_commonStockSharesOutstanding").is_not_null()
                    & (pl.col("earnings_implied_commonStockSharesOutstanding") > 0)
                    & pl.col("implied_to_reported_ratio").is_between(
                        LEGACY_SHARE_ALIGNMENT_MIN_RATIO,
                        LEGACY_SHARE_ALIGNMENT_MAX_RATIO,
                        closed="both",
                    )
                )
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("use_earnings_implied"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("use_earnings_implied"))
                .then(pl.col("earnings_implied_commonStockSharesOutstanding"))
                .otherwise(pl.col("reported_commonStockSharesOutstanding"))
                .alias("exported_commonStockSharesOutstanding"),
                pl.when(pl.col("use_earnings_implied"))
                .then(pl.lit("earnings_implied"))
                .otherwise(pl.lit("reported_period_end"))
                .alias("selected_method"),
                pl.lit(
                    "use abs(netIncome / epsActual) when the implied shares stay within 0.8x-1.2x of the reported period-end shares"
                ).alias("selection_rule"),
            ]
        )
    )

    adjusted_balance = balance_frame.join(
        semantics.select(["ticker", "date", "exported_commonStockSharesOutstanding"]),
        on=["ticker", "date"],
        how="left",
    ).with_columns(
        pl.coalesce(
            [
                pl.col("exported_commonStockSharesOutstanding"),
                pl.col("commonStockSharesOutstanding").cast(pl.Float64, strict=False),
            ]
        ).alias("commonStockSharesOutstanding")
    ).drop("exported_commonStockSharesOutstanding", strict=False)

    lineage_columns = [
        "ticker",
        "date",
        "filing_date",
        "reportDate",
        "netIncome",
        "epsActual",
        "reported_commonStockSharesOutstanding",
        "earnings_implied_commonStockSharesOutstanding",
        "exported_commonStockSharesOutstanding",
        "implied_to_reported_ratio",
        "selected_method",
        "selection_rule",
    ]
    lineage_columns.extend(column for column in earnings_source_columns if column not in lineage_columns)
    lineage = semantics.select(lineage_columns)
    return adjusted_balance, coerce_schema(lineage, lineage_schema)


def _empty_legacy_share_semantics_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "date": pl.Utf8,
            "filing_date": pl.Utf8,
            "reportDate": pl.Utf8,
            "netIncome": pl.Float64,
            "epsActual": pl.Float64,
            "reported_commonStockSharesOutstanding": pl.Float64,
            "earnings_implied_commonStockSharesOutstanding": pl.Float64,
            "exported_commonStockSharesOutstanding": pl.Float64,
            "implied_to_reported_ratio": pl.Float64,
            "selected_method": pl.Utf8,
            "selection_rule": pl.Utf8,
            "selected_source": pl.Utf8,
            "actual_source": pl.Utf8,
            "estimate_source": pl.Utf8,
            "surprise_source": pl.Utf8,
            "source_label": pl.Utf8,
            "candidate_sources": pl.Utf8,
        }
    )


def _ensure_financial_legacy_columns(frame: pl.DataFrame) -> pl.DataFrame:
    desired_types: dict[str, pl.DataType] = {
        "source_priority": pl.Int64,
        "selected_fiscal_period": pl.Utf8,
        "selected_fiscal_year": pl.Int64,
        "selected_form": pl.Utf8,
    }
    expressions: list[pl.Expr] = []
    for column, dtype in desired_types.items():
        if column not in frame.columns:
            expressions.append(pl.lit(None).cast(dtype).alias(column))
    return frame.with_columns(expressions) if expressions else frame


def _build_general_sector_frame(general_reference: pl.DataFrame) -> pl.DataFrame:
    if general_reference.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "Sector": pl.Utf8,
                "industry": pl.Utf8,
            }
        )
    return _ensure_general_reference_columns(general_reference).select(["ticker", "Sector", "industry"])


def _build_legacy_filing_date_reference(source_frame: pl.DataFrame) -> pl.DataFrame:
    if source_frame.is_empty() or "filing_date" not in source_frame.columns:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "legacy_date": pl.Utf8, "fallback_filing_date": pl.Utf8})

    reference = (
        _ensure_financial_legacy_columns(source_frame)
        .filter(pl.col("filing_date").is_not_null())
        .with_columns(
            _normalize_statement_date_for_legacy(pl.col("date").cast(pl.Utf8, strict=False)).alias("legacy_date")
        )
        .filter(pl.col("legacy_date").is_not_null())
        .group_by(["ticker", "legacy_date"], maintain_order=True)
        .agg(pl.col("filing_date").drop_nulls().sort().last().alias("fallback_filing_date"))
    )
    if reference.is_empty():
        return pl.DataFrame(schema={"ticker": pl.Utf8, "legacy_date": pl.Utf8, "fallback_filing_date": pl.Utf8})
    return reference


def _collapse_duplicate_filing_dates(keyed: pl.DataFrame) -> pl.DataFrame:
    if keyed.is_empty() or "filing_date" not in keyed.columns:
        return keyed

    non_null_filing = keyed.filter(pl.col("filing_date").is_not_null())
    if non_null_filing.is_empty():
        return keyed

    scored_dates = (
        non_null_filing.group_by(["ticker", "legacy_metric", "filing_date", "legacy_date"], maintain_order=True)
        .agg(
            [
                pl.len().alias("row_count"),
                pl.col("source_priority").min().alias("best_source_priority"),
                pl.col("source").is_in(["yfinance", "simfin"]).sum().alias("vendor_count"),
                pl.col("source")
                .is_in(["sec_companyfacts", "sec_filing"])
                .sum()
                .alias("sec_count"),
                pl.col("legacy_metric").first().alias("legacy_metric_group"),
            ]
        )
        .with_columns(
            [
                pl.col("legacy_date").cast(pl.Date, strict=False).alias("legacy_date_as_date"),
                pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date_as_date"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("filing_date_as_date") - pl.col("legacy_date_as_date")
                )
                .dt.total_days()
                .alias("days_before_filing"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("days_before_filing") < 0)
                .then(pl.lit(10_000))
                .otherwise(pl.col("days_before_filing"))
                .alias("date_distance_score")
            ]
        )
        .sort(
            [
                "ticker",
                "legacy_metric",
                "filing_date",
                "vendor_count",
                "row_count",
                "best_source_priority",
                "date_distance_score",
                "legacy_date",
            ],
            descending=[False, False, False, True, True, False, False, True],
            nulls_last=True,
        )
        .group_by(["ticker", "legacy_metric", "filing_date"], maintain_order=True)
        .agg(pl.col("legacy_date").first().alias("selected_legacy_date"))
    )

    filtered = non_null_filing.join(scored_dates, on=["ticker", "legacy_metric", "filing_date"], how="left").filter(
        pl.col("legacy_date") == pl.col("selected_legacy_date")
    )
    null_filing = keyed.filter(pl.col("filing_date").is_null())
    combined = filtered.drop("selected_legacy_date")
    return pl.concat([combined, null_filing], how="diagonal_relaxed") if not null_filing.is_empty() else combined


def _select_financial_legacy_candidates(keyed: pl.DataFrame, *, statement: str) -> pl.DataFrame:
    selected_rows: list[dict[str, object]] = []
    candidate_map: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    series_keys: dict[tuple[str, str], list[tuple[str, str, str]]] = {}

    for _, group_frame in keyed.group_by(["ticker", "legacy_date", "legacy_metric"], maintain_order=True):
        ordered = group_frame.sort(
            ["source_priority", "filing_date", "selected_fiscal_year", "source_date"],
            descending=[False, False, False, False],
            nulls_last=True,
        )
        rows = ordered.to_dicts()
        if not rows:
            continue
        group_key = (
            str(rows[0]["ticker"]),
            str(rows[0]["legacy_date"]),
            str(rows[0]["legacy_metric"]),
        )
        candidate_map[group_key] = rows
        series_key = (group_key[0], group_key[2])
        series_keys.setdefault(series_key, []).append(group_key)
        selected_rows.append(_select_default_legacy_candidate(rows, statement=statement))

    selected_map = {
        (str(row["ticker"]), str(row["date"]), str(row["legacy_metric"])): row
        for row in selected_rows
    }

    for _, group_keys in series_keys.items():
        ordered_group_keys = sorted(group_keys, key=lambda key: key[1])
        for index, group_key in enumerate(ordered_group_keys):
            rows = candidate_map[group_key]
            if len(rows) <= 1:
                continue
            current = selected_map[group_key]
            replacement = _choose_legacy_sequence_replacement(
                statement=statement,
                current=current,
                candidates=rows,
                previous_selected=selected_map.get(ordered_group_keys[index - 1]) if index > 0 else None,
                next_selected=selected_map.get(ordered_group_keys[index + 1]) if index + 1 < len(ordered_group_keys) else None,
            )
            if replacement is not None:
                selected_map[group_key] = replacement

    return pl.DataFrame(list(selected_map.values())).sort(["ticker", "date", "legacy_metric"])


def _select_default_legacy_candidate(rows: list[dict[str, object]], *, statement: str) -> dict[str, object]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _sort_int(row.get("source_priority"), fallback=99),
            0 if row.get("filing_date") else 1,
            0 if row.get("selected_fiscal_period") else 1,
            0 if row.get("selected_fiscal_year") is not None else 1,
            str(row.get("source_date") or ""),
        ),
    )
    default = ordered[0]
    vendor = _best_vendor_candidate(ordered)
    sector = str(default.get("Sector") or "")
    legacy_metric = str(default.get("legacy_metric") or "")
    if statement == "shares":
        default_value = _coerce_positive_float(default.get("value"))
        vendor_value = _coerce_positive_float(vendor.get("value")) if vendor is not None else None
        if default_value is not None and vendor_value is not None:
            ratio = max(default_value, vendor_value) / min(default_value, vendor_value)
            if ratio >= 1.5 and default.get("source") != vendor.get("source"):
                return _legacy_candidate_row(vendor, rows=rows)
    if vendor is not None:
        default_value = _coerce_float(default.get("value"))
        vendor_value = _coerce_float(vendor.get("value"))
        ratio = _safe_ratio(default_value, vendor_value)
        if _should_prefer_vendor(
            statement=statement,
            legacy_metric=legacy_metric,
            sector=sector,
            default_source=str(default.get("source") or ""),
            default_value=default_value,
            vendor_value=vendor_value,
            ratio=ratio,
        ):
            return _legacy_candidate_row(vendor, rows=rows)
        if default_value is not None and vendor_value is not None and default.get("source", "").startswith("sec_"):
            if default_value <= 0 < vendor_value:
                return _legacy_candidate_row(vendor, rows=rows)
    return _legacy_candidate_row(default, rows=rows)


def _choose_legacy_sequence_replacement(
    *,
    statement: str,
    current: dict[str, object],
    candidates: list[dict[str, object]],
    previous_selected: dict[str, object] | None,
    next_selected: dict[str, object] | None,
) -> dict[str, object] | None:
    vendor = _best_vendor_candidate(candidates)
    if vendor is None:
        return None

    current_value = _coerce_float(current.get("value"))
    vendor_value = _coerce_float(vendor.get("value"))
    if current_value is None or vendor_value is None:
        return None

    sec_consensus = _has_sec_consensus(candidates)
    ratio = _safe_ratio(current_value, vendor_value)
    current_source = str(current.get("selected_source") or current.get("source") or "")
    fiscal_period = str(current.get("selected_fiscal_period") or "")

    if (
        statement in {"income_statement", "cash_flow"}
        and current_source.startswith("sec_")
        and fiscal_period == "Q4"
        and not sec_consensus
        and ratio is not None
        and ratio >= 1.25
    ):
        return _legacy_candidate_row(vendor, rows=candidates)

    expected = _expected_series_value(previous_selected=previous_selected, next_selected=next_selected)
    if expected is None or expected <= 0:
        if current_source.startswith("sec_") and current_value <= 0 < vendor_value:
            return _legacy_candidate_row(vendor, rows=candidates)
        return None

    best_candidate = min(
        (candidate for candidate in candidates if _coerce_positive_float(candidate.get("value")) is not None),
        key=lambda candidate: _log_distance(_coerce_positive_float(candidate.get("value")), expected),
        default=None,
    )
    if best_candidate is None:
        return None

    current_distance = _log_distance(_coerce_positive_float(current.get("value")), expected)
    best_distance = _log_distance(_coerce_positive_float(best_candidate.get("value")), expected)
    if best_distance >= current_distance:
        return None

    improvement_ratio = current_distance / max(best_distance, 1e-9)
    if current_source.startswith("sec_") and improvement_ratio >= 3.0 and current_distance >= 0.35:
        return _legacy_candidate_row(best_candidate, rows=candidates)
    return None


def _legacy_candidate_row(row: dict[str, object], *, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "ticker": row.get("ticker"),
        "date": row.get("legacy_date"),
        "legacy_metric": row.get("legacy_metric"),
        "filing_date": row.get("filing_date") or _best_available_field(rows, "filing_date"),
        "value": row.get("value"),
        "selected_source": row.get("source"),
        "selected_fiscal_period": row.get("selected_fiscal_period") or _best_available_field(rows, "selected_fiscal_period"),
        "selected_fiscal_year": row.get("selected_fiscal_year") or _best_available_field(rows, "selected_fiscal_year"),
        "selected_form": row.get("selected_form") or _best_available_field(rows, "selected_form"),
    }


def _best_vendor_candidate(rows: list[dict[str, object]]) -> dict[str, object] | None:
    vendor_rows = [row for row in rows if str(row.get("source") or "") in {"yfinance", "simfin"}]
    if not vendor_rows:
        return None
    return sorted(
        vendor_rows,
        key=lambda row: (
            0 if str(row.get("source") or "") == "yfinance" else 1,
            _sort_int(row.get("source_priority"), fallback=99),
            0 if row.get("filing_date") else 1,
        ),
    )[0]


def _should_prefer_vendor(
    *,
    statement: str,
    legacy_metric: str,
    sector: str,
    default_source: str,
    default_value: float | None,
    vendor_value: float | None,
    ratio: float | None,
) -> bool:
    if not default_source.startswith("sec_"):
        return False
    if default_value is None or vendor_value is None:
        return False
    if vendor_value <= 0:
        return False

    if legacy_metric == "totalRevenue" and sector in {"Real Estate", "Financial Services"}:
        return True
    if legacy_metric == "totalRevenue" and ratio is not None and ratio >= 2.0:
        return True

    if statement == "balance_sheet" and legacy_metric in {"totalLiab", "totalStockholderEquity"}:
        return ratio is not None and ratio >= 10.0

    return False


def _best_available_field(rows: list[dict[str, object]], field: str) -> object | None:
    for row in sorted(
        rows,
        key=lambda candidate: (
            0 if candidate.get(field) is not None else 1,
            _sort_int(candidate.get("source_priority"), fallback=99),
        ),
    ):
        value = row.get(field)
        if value is not None:
            return value
    return None


def _has_sec_consensus(rows: list[dict[str, object]]) -> bool:
    sec_values = [
        _coerce_positive_float(row.get("value"))
        for row in rows
        if str(row.get("source") or "").startswith("sec_")
    ]
    sec_values = [value for value in sec_values if value is not None]
    if len(sec_values) < 2:
        return False
    minimum = min(sec_values)
    maximum = max(sec_values)
    return minimum > 0 and (maximum / minimum) <= 1.01


def _expected_series_value(
    *,
    previous_selected: dict[str, object] | None,
    next_selected: dict[str, object] | None,
) -> float | None:
    neighbors = [
        _coerce_positive_float(previous_selected.get("value")) if previous_selected is not None else None,
        _coerce_positive_float(next_selected.get("value")) if next_selected is not None else None,
    ]
    numeric = [value for value in neighbors if value is not None]
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]
    return sum(numeric) / len(numeric)


def _log_distance(value: float | None, expected: float | None) -> float:
    if value is None or expected is None or value <= 0 or expected <= 0:
        return float("inf")
    ratio = max(value, expected) / min(value, expected)
    return abs(math.log(ratio))


def _safe_ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    return max(left, right) / min(left, right)


def _sort_int(value: object, *, fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_float(value: object) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def _ensure_general_reference_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in GENERAL_COLUMNS:
        if column not in frame.columns:
            expressions.append(pl.lit(None).cast(pl.Utf8).alias(column))
    return frame.with_columns(expressions) if expressions else frame


def _classify_before_after_market(column: pl.Expr) -> pl.Expr:
    hour = column.str.slice(11, 2).cast(pl.Int64, strict=False)
    return (
        pl.when(hour < 12)
        .then(pl.lit("BeforeMarket"))
        .when(hour >= 16)
        .then(pl.lit("AfterMarket"))
        .otherwise(pl.lit(None).cast(pl.Utf8))
    )


def _normalize_statement_date_for_legacy(column: pl.Expr) -> pl.Expr:
    date = column.cast(pl.Date, strict=False)
    month_end = date.dt.month_end()
    previous_month_end = (date.dt.truncate("1mo") - pl.duration(days=1)).cast(pl.Date)
    days_from_previous = date.dt.day().cast(pl.Int64)
    days_to_current = (month_end.dt.day() - date.dt.day()).cast(pl.Int64)
    normalized = (
        pl.when(date.is_null())
        .then(pl.lit(None).cast(pl.Date))
        .when(days_from_previous <= days_to_current)
        .then(previous_month_end)
        .otherwise(month_end)
    )
    return normalized.dt.strftime("%Y-%m-%d")


def _normalize_earnings_period_end_for_legacy(*, period_end: pl.Expr, report_date: pl.Expr) -> pl.Expr:
    period_date = period_end.cast(pl.Date, strict=False)
    report_date_date = report_date.cast(pl.Date, strict=False)
    same_month_end = period_date.dt.month_end()
    previous_month_end = (period_date.dt.truncate("1mo") - pl.duration(days=1)).cast(pl.Date)
    normalized_period_end = (
        pl.when(period_date.is_null())
        .then(pl.lit(None).cast(pl.Date))
        .when(period_date.dt.day() <= 7)
        .then(previous_month_end)
        .otherwise(same_month_end)
    )
    return pl.coalesce([normalized_period_end, report_date_date]).dt.strftime("%Y-%m-%d")
