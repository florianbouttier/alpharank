#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.data.ticker_integrity import (  # noqa: E402
    DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
    load_ticker_exclusion_registry,
)
from alpharank.portfolio.performance import legacy_report_statistics  # noqa: E402


DEFAULT_SNAPSHOT = Path(
    "outputs/2026-07-13/runs/20260713_201639/input_snapshot"
)
DEFAULT_LEGACY_DETAILED = Path(
    "outputs/2026-07-13/runs/20260713_201639/"
    "legacy_detailed_returns_polars.parquet"
)
DEFAULT_LEGACY_MONTHLY = Path(
    "outputs/2026-07-13/runs/20260713_201639/"
    "legacy_monthly_returns_polars.parquet"
)
DEFAULT_ML_HOLDINGS = Path(
    "outputs/multihorizon_boosting/"
    "legacy_ema_risk_overlay_long_history_clean_v2_20260726/"
    "allocation_holdings.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/data_quality/historical_ticker_price_audit_20260726"
)
DEFAULT_CORRECTED_RISK_OUTPUT = Path(
    "outputs/multihorizon_boosting/"
    "legacy_ema_risk_overlay_ticker_quarantine_v6_20260726"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_expr(column: str) -> pl.Expr:
    dtype = pl.col(column)
    return dtype.cast(pl.Date, strict=False)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _first_or_none(frame: pl.DataFrame, column: str) -> Any:
    if frame.is_empty() or column not in frame.columns:
        return None
    values = frame[column].drop_nulls()
    return values[0] if values.len() else None


def _price_audit_rows(
    entries: list[dict[str, Any]],
    *,
    prices: pl.DataFrame,
    general: pl.DataFrame,
    constituents: pl.DataFrame,
    legacy_holdings: pl.DataFrame,
    ml_holdings: pl.DataFrame,
) -> list[dict[str, Any]]:
    prices = prices.with_columns(_date_expr("date").alias("date"))
    constituents = constituents.with_columns(_date_expr("Date").alias("Date"))
    legacy_frequency = legacy_holdings.filter(
        (pl.col("portfolio_model") == "Combined_Frequency")
        & pl.col("weight_normalized").is_not_null()
    )
    ml_equal = ml_holdings.filter(pl.col("strategy") == "alpha_top5_equal")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        ticker = str(entry["ticker"])
        symbol = ticker.removesuffix(".US").replace("-", ".")
        ticker_prices = (
            prices.filter(
                (pl.col("ticker").cast(pl.Utf8).str.to_uppercase() == ticker)
                & pl.col("adjusted_close").is_not_null()
            )
            .sort("date")
            .with_columns(
                pl.col("adjusted_close").pct_change().alias("_daily_return")
            )
        )
        if {"open", "high", "low", "close"} <= set(ticker_prices.columns):
            ticker_prices = ticker_prices.with_columns(
                (
                    (pl.col("high") < pl.max_horizontal("open", "close", "low"))
                    | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
                    | (pl.col("high") < pl.col("low"))
                    | (pl.col("open") <= 0)
                    | (pl.col("close") <= 0)
                )
                .fill_null(True)
                .alias("_ohlc_violation")
            )
        else:
            ticker_prices = ticker_prices.with_columns(
                pl.lit(False).alias("_ohlc_violation")
            )
        extreme = (
            ticker_prices.filter(pl.col("_daily_return").is_not_null())
            .sort(pl.col("_daily_return").abs(), descending=True)
            .head(1)
        )
        constituent = constituents.filter(
            pl.col("Ticker").cast(pl.Utf8).str.to_uppercase() == symbol
        ).sort("Date")
        reference = general.filter(
            pl.col("ticker").cast(pl.Utf8).str.to_uppercase() == ticker
        )
        terminal = entry.get("official_terminal_date")
        official_start = entry.get("official_start_date")
        post_terminal = (
            ticker_prices.filter(pl.col("date") > pl.lit(terminal).str.to_date())
            if terminal
            else ticker_prices.head(0)
        )
        pre_official = (
            ticker_prices.filter(pl.col("date") < pl.lit(official_start).str.to_date())
            if official_start
            else ticker_prices.head(0)
        )
        legacy_exposure = legacy_frequency.filter(pl.col("ticker") == ticker)
        ml_exposure = ml_equal.filter(pl.col("ticker") == ticker)
        constituent_names = (
            constituent["Name"].drop_nulls().unique(maintain_order=True).to_list()
            if not constituent.is_empty()
            else []
        )
        rows.append(
            {
                "ticker": ticker,
                "decision": entry["decision"],
                "reason_codes": ", ".join(entry["reason_codes"]),
                "constituent_identity": entry["constituent_identity"],
                "dataset_general_identity": _first_or_none(reference, "Name")
                or entry["dataset_general_identity"],
                "constituent_names_in_file": " | ".join(map(str, constituent_names)),
                "constituent_start": _iso(
                    constituent["Date"].min() if not constituent.is_empty() else None
                ),
                "constituent_end": _iso(
                    constituent["Date"].max() if not constituent.is_empty() else None
                ),
                "constituent_months": constituent.height,
                "official_start_date": official_start,
                "official_terminal_date": terminal,
                "price_start": _iso(
                    ticker_prices["date"].min() if not ticker_prices.is_empty() else None
                ),
                "price_end": _iso(
                    ticker_prices["date"].max() if not ticker_prices.is_empty() else None
                ),
                "non_null_price_rows": ticker_prices.height,
                "post_terminal_price_rows": post_terminal.height,
                "pre_official_start_price_rows": pre_official.height,
                "adjusted_close_min": _safe_float(
                    ticker_prices["adjusted_close"].min()
                    if not ticker_prices.is_empty()
                    else None
                ),
                "adjusted_close_max": _safe_float(
                    ticker_prices["adjusted_close"].max()
                    if not ticker_prices.is_empty()
                    else None
                ),
                "ohlc_violation_rows": int(
                    ticker_prices["_ohlc_violation"].sum() or 0
                ),
                "max_abs_daily_return": _safe_float(
                    abs(_first_or_none(extreme, "_daily_return") or np.nan)
                ),
                "max_abs_daily_return_date": _iso(
                    _first_or_none(extreme, "date")
                ),
                "median_daily_dollar_volume": _safe_float(
                    (
                        ticker_prices["close"] * ticker_prices["volume"]
                    ).median()
                    if {"close", "volume"} <= set(ticker_prices.columns)
                    else None
                ),
                "legacy_frequency_holding_months": legacy_exposure[
                    "year_month"
                ].n_unique(),
                "legacy_frequency_rows": legacy_exposure.height,
                "ml_v2_equal_holding_months": ml_exposure[
                    "holding_month"
                ].n_unique(),
                "ml_v2_equal_rows": ml_exposure.height,
                "rationale": entry["rationale"],
                "sources": entry["sources"],
            }
        )
    return rows


def _screen_held_universe(
    *,
    prices: pl.DataFrame,
    general: pl.DataFrame,
    constituents: pl.DataFrame,
    legacy_holdings: pl.DataFrame,
    ml_holdings: pl.DataFrame,
    excluded_tickers: Iterable[str],
) -> list[dict[str, Any]]:
    """Screen every held ticker and leave ambiguous cases in a review queue."""

    excluded = set(excluded_tickers)
    prices = prices.with_columns(_date_expr("date").alias("date"))
    constituents = constituents.with_columns(
        _date_expr("Date").alias("Date"),
        (
            pl.col("Ticker")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .str.replace_all(r"\.", "-")
            + pl.lit(".US")
        ).alias("_ticker"),
    )
    legacy_tickers = legacy_holdings.filter(
        (pl.col("portfolio_model") == "Combined_Frequency")
        & pl.col("weight_normalized").is_not_null()
    )["ticker"].unique()
    ml_tickers = ml_holdings.filter(
        pl.col("strategy") == "alpha_top5_equal"
    )["ticker"].unique()
    held_tickers = sorted(
        set(legacy_tickers.to_list()) | set(ml_tickers.to_list())
    )
    bounds = constituents.group_by("_ticker").agg(
        pl.col("Date").min().alias("_membership_start"),
        pl.col("Date").max().alias("_membership_end"),
        pl.col("Name").last().alias("_constituent_name"),
    )
    rows: list[dict[str, Any]] = []
    for ticker in held_tickers:
        bound = bounds.filter(pl.col("_ticker") == ticker)
        if bound.is_empty():
            continue
        start = bound["_membership_start"][0]
        end = bound["_membership_end"][0]
        ticker_prices = (
            prices.filter(
                (pl.col("ticker") == ticker)
                & (pl.col("date") >= start)
                & (pl.col("date") <= end)
                & pl.col("adjusted_close").is_not_null()
            )
            .sort("date")
            .with_columns(
                pl.col("adjusted_close").pct_change().alias("_daily_return"),
                (
                    (pl.col("high") < pl.max_horizontal("open", "close", "low"))
                    | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
                    | (pl.col("high") < pl.col("low"))
                )
                .fill_null(True)
                .alias("_ohlc_violation"),
            )
        )
        reference = general.filter(pl.col("ticker") == ticker)
        constituent_name = str(bound["_constituent_name"][0] or "")
        general_name = str(_first_or_none(reference, "Name") or "")
        name_similarity = SequenceMatcher(
            None,
            constituent_name.lower(),
            general_name.lower(),
        ).ratio()
        max_abs_daily_return = _safe_float(
            ticker_prices["_daily_return"].abs().max()
            if not ticker_prices.is_empty()
            else None
        )
        median_dollar_volume = _safe_float(
            (ticker_prices["close"] * ticker_prices["volume"]).median()
            if not ticker_prices.is_empty()
            else None
        )
        ohlc_violations = (
            int(ticker_prices["_ohlc_violation"].sum() or 0)
            if not ticker_prices.is_empty()
            else 0
        )
        flags: list[str] = []
        if max_abs_daily_return is not None and max_abs_daily_return > 1.0:
            flags.append("daily_return_above_100pct")
        if ohlc_violations:
            flags.append("ohlc_violation")
        if median_dollar_volume is None or median_dollar_volume < 1_000_000:
            flags.append("median_dollar_volume_below_1m")
        if name_similarity < 0.35:
            flags.append("low_name_similarity")
        status = (
            "excluded"
            if ticker in excluded
            else "review"
            if flags
            else "screen_pass"
        )
        rows.append(
            {
                "ticker": ticker,
                "status": status,
                "flags": ", ".join(flags),
                "constituent_name": constituent_name,
                "general_name": general_name,
                "name_similarity": name_similarity,
                "membership_start": _iso(start),
                "membership_end": _iso(end),
                "price_rows_during_membership": ticker_prices.height,
                "max_abs_daily_return": max_abs_daily_return,
                "ohlc_violation_rows": ohlc_violations,
                "median_daily_dollar_volume": median_dollar_volume,
            }
        )
    return rows


def _legacy_post_selection_sensitivity(
    detailed: pl.DataFrame,
    excluded_tickers: Iterable[str],
) -> pl.DataFrame:
    excluded = set(excluded_tickers)
    combined = detailed.filter(
        (pl.col("portfolio_model") == "Combined_Frequency")
        & pl.col("weight_normalized").is_not_null()
        & pl.col("dr").is_not_null()
    )
    rows: list[dict[str, Any]] = []
    for month in combined.partition_by("year_month", maintain_order=True):
        kept = month.filter(~pl.col("ticker").is_in(excluded))
        if kept.is_empty():
            filtered_return = 0.0
            kept_weight = 0.0
        else:
            raw_weight = kept["weight"].cast(pl.Float64)
            kept_weight = float(raw_weight.sum())
            filtered_return = float(
                (kept["dr"].cast(pl.Float64) * raw_weight / kept_weight).sum()
            )
        rows.append(
            {
                "holding_month": month["year_month"][0],
                "filtered_return": filtered_return,
                "kept_raw_weight": kept_weight,
                "removed_positions": month.height - kept.height,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("holding_month").cast(pl.Date)
    ).sort("holding_month")


def _ml_post_selection_sensitivity(
    holdings: pl.DataFrame,
    excluded_tickers: Iterable[str],
) -> pl.DataFrame:
    excluded = set(excluded_tickers)
    equal = holdings.filter(
        (pl.col("strategy") == "alpha_top5_equal")
        & pl.col("future_return_1m").is_not_null()
    )
    rows: list[dict[str, Any]] = []
    for month in equal.partition_by("holding_month", maintain_order=True):
        kept = month.filter(~pl.col("ticker").is_in(excluded))
        if kept.is_empty():
            filtered_return = 0.0
        else:
            filtered_return = float(kept["future_return_1m"].mean())
        rows.append(
            {
                "holding_month": month["holding_month"][0],
                "published_return": float(
                    (
                        month["future_return_1m"].cast(pl.Float64)
                        * month["portfolio_weight"].cast(pl.Float64)
                    ).sum()
                ),
                "filtered_return": filtered_return,
                "removed_positions": month.height - kept.height,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("holding_month").cast(pl.Date)
    ).sort("holding_month")


def _spy_monthly(prices: pl.DataFrame) -> pl.DataFrame:
    return (
        prices.filter(
            (pl.col("ticker") == "SPY.US")
            & pl.col("adjusted_close").is_not_null()
        )
        .with_columns(_date_expr("date").alias("date"))
        .sort("date")
        .with_columns(pl.col("date").dt.truncate("1mo").alias("holding_month"))
        .group_by("holding_month")
        .agg(pl.col("adjusted_close").last().alias("_close"))
        .sort("holding_month")
        .with_columns(
            (pl.col("_close") / pl.col("_close").shift(1) - 1.0).alias("return")
        )
        .drop_nulls("return")
        .select("holding_month", "return")
    )


def _metric_row(
    *,
    comparison: str,
    series: str,
    frame: pl.DataFrame,
    return_column: str,
) -> dict[str, Any]:
    clean = frame.filter(pl.col(return_column).is_not_null()).sort("holding_month")
    stats = legacy_report_statistics(
        clean[return_column].to_numpy(),
        holding_months=clean["holding_month"].to_numpy(),
    )
    return {
        "comparison": comparison,
        "series": series,
        "start": _iso(clean["holding_month"].min()),
        "end": _iso(clean["holding_month"].max()),
        "months": clean.height,
        "cagr": stats["cagr"],
        "sharpe_legacy": stats["sharpe"],
        "max_drawdown": stats["max_drawdown"],
        "worst_full_year": stats["worst_full_calendar_year"],
        "worst_full_year_return": stats["worst_full_calendar_year_return"],
    }


def _performance_rows(
    *,
    legacy_monthly: pl.DataFrame,
    legacy_sensitivity: pl.DataFrame,
    ml_sensitivity: pl.DataFrame,
    spy: pl.DataFrame,
) -> list[dict[str, Any]]:
    legacy = (
        legacy_monthly.filter(pl.col("model") == "Combined_Frequency")
        .select(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.col("monthly_return").alias("published_return"),
        )
        .sort("holding_month")
    )
    full = (
        legacy.join(legacy_sensitivity, on="holding_month", how="inner")
        .join(spy.rename({"return": "spy_return"}), on="holding_month", how="inner")
    )
    ml_start = ml_sensitivity["holding_month"].min()
    ml_end = ml_sensitivity["holding_month"].max()
    ml_common = (
        ml_sensitivity.join(
            legacy.rename({"published_return": "legacy_return"}),
            on="holding_month",
            how="inner",
        )
        .join(spy.rename({"return": "spy_return"}), on="holding_month", how="inner")
        .filter(
            (pl.col("holding_month") >= ml_start)
            & (pl.col("holding_month") <= ml_end)
        )
    )
    rows = [
        _metric_row(
            comparison="Legacy full common window",
            series="Legacy published",
            frame=full,
            return_column="published_return",
        ),
        _metric_row(
            comparison="Legacy full common window",
            series="Legacy holdings sensitivity — all audited tickers removed",
            frame=full,
            return_column="filtered_return",
        ),
        _metric_row(
            comparison="Legacy full common window",
            series="SPY total return",
            frame=full,
            return_column="spy_return",
        ),
        _metric_row(
            comparison="ML v2 common window",
            series="ML v2 equal weight published",
            frame=ml_common,
            return_column="published_return",
        ),
        _metric_row(
            comparison="ML v2 common window",
            series="ML v2 holdings sensitivity — all audited tickers removed",
            frame=ml_common,
            return_column="filtered_return",
        ),
        _metric_row(
            comparison="ML v2 common window",
            series="Legacy published",
            frame=ml_common,
            return_column="legacy_return",
        ),
        _metric_row(
            comparison="ML v2 common window",
            series="SPY total return",
            frame=ml_common,
            return_column="spy_return",
        ),
    ]
    return rows


def _percent(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], *, wide: bool = False) -> str:
    table_class = "wide" if wide else ""
    return (
        f'<div class="table-wrap"><table class="{table_class}"><thead><tr>'
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _render_html(
    *,
    registry: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
    screening_rows: list[dict[str, Any]],
    corrected_performance: list[dict[str, Any]],
    output_path: Path,
) -> None:
    impact_table = _table(
        ["Fenêtre", "Série", "Période", "Mois", "CAGR", "Sharpe", "Max DD", "Pire année"],
        [
            [
                html.escape(row["comparison"]),
                html.escape(row["series"]),
                f'{html.escape(row["start"])} → {html.escape(row["end"])}',
                str(row["months"]),
                _percent(row["cagr"]),
                _number(row["sharpe_legacy"], 3),
                _percent(row["max_drawdown"]),
                (
                    f'{row["worst_full_year"]} · '
                    f'{_percent(row["worst_full_year_return"])}'
                    if row["worst_full_year"] != -1
                    else "—"
                ),
            ]
            for row in performance_rows
        ],
        wide=True,
    )
    audit_table = _table(
        [
            "Ticker",
            "Identité constituants",
            "Identité référence",
            "Prix dataset",
            "Anomalies",
            "Exposition",
            "Décision",
        ],
        [
            [
                f"<strong>{html.escape(row['ticker'])}</strong>",
                (
                    f"{html.escape(row['constituent_identity'])}<br>"
                    f"<small>{html.escape(str(row['constituent_start']))} → "
                    f"{html.escape(str(row['constituent_end']))}</small>"
                ),
                html.escape(str(row["dataset_general_identity"])),
                (
                    f"{_number(row['adjusted_close_min'])} → "
                    f"{_number(row['adjusted_close_max'])}<br>"
                    f"<small>{html.escape(str(row['price_start']))} → "
                    f"{html.escape(str(row['price_end']))}</small>"
                ),
                (
                    f"OHLC: {row['ohlc_violation_rows']}<br>"
                    f"après fin officielle: {row['post_terminal_price_rows']}<br>"
                    f"avant cotation officielle: {row['pre_official_start_price_rows']}"
                ),
                (
                    f"Legacy: {row['legacy_frequency_holding_months']} mois<br>"
                    f"ML v2: {row['ml_v2_equal_holding_months']} mois"
                ),
                '<span class="decision">Exclusion complète</span>',
            ]
            for row in audit_rows
        ],
        wide=True,
    )
    corrected_table = _table(
        ["Série", "Période", "CAGR", "Sharpe", "Max DD", "Pire année"],
        [
            [
                html.escape(str(row["series"])),
                (
                    f'{html.escape(str(row["start_holding_month"]))} → '
                    f'{html.escape(str(row["end_holding_month"]))}'
                ),
                _percent(row["cagr"]),
                _number(row["sharpe"], 3),
                _percent(row["max_drawdown"]),
                (
                    f'{row["worst_full_calendar_year"]} · '
                    f'{_percent(row["worst_full_calendar_year_return"])}'
                ),
            ]
            for row in corrected_performance
        ],
        wide=True,
    )
    reviews = [
        row for row in screening_rows if row["status"] == "review"
    ]
    review_table = _table(
        [
            "Ticker",
            "Drapeaux automatiques",
            "Identité constituants",
            "Identité référence",
            "Décision",
        ],
        [
            [
                f"<strong>{html.escape(row['ticker'])}</strong>",
                html.escape(row["flags"]),
                html.escape(row["constituent_name"]),
                html.escape(row["general_name"]),
                '<span class="decision review">Revue manuelle</span>',
            ]
            for row in reviews
        ],
        wide=True,
    )
    detail_cards = []
    for row in audit_rows:
        source_links = "".join(
            (
                f'<li><a href="{html.escape(source["url"], quote=True)}">'
                f'{html.escape(source["title"])}</a>'
                f'<span>{html.escape(source["claim"])}</span></li>'
            )
            for source in row["sources"]
        )
        detail_cards.append(
            f"""
            <article class="case">
              <div class="case-head">
                <h3>{html.escape(row["ticker"])}</h3>
                <span class="decision">exclude_all_dates</span>
              </div>
              <p>{html.escape(row["rationale"])}</p>
              <p class="codes">{html.escape(row["reason_codes"])}</p>
              <ul class="sources">{source_links}</ul>
            </article>
            """
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Audit des incohérences de prix et d'identité</title>
  <style>
    :root {{ --ink:#17212b; --muted:#62707d; --line:#dbe2e8; --paper:#f5f7f8;
      --card:#fff; --red:#a82b2b; --red-bg:#fff0ee; --blue:#215b7d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--paper);
      font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:48px 24px 80px; }}
    .eyebrow {{ color:var(--blue); font-size:12px; font-weight:800; letter-spacing:.13em;
      text-transform:uppercase; }}
    h1 {{ font-size:clamp(34px,6vw,64px); line-height:1.02; letter-spacing:-.045em;
      margin:10px 0 20px; max-width:930px; }}
    h2 {{ margin:54px 0 14px; font-size:28px; letter-spacing:-.025em; }}
    h3 {{ margin:0; font-size:21px; }}
    .lede {{ max-width:800px; color:var(--muted); font-size:18px; }}
    .verdict {{ margin:30px 0; padding:22px 24px; border-left:5px solid var(--red);
      background:var(--red-bg); border-radius:8px; font-size:17px; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:26px 0; }}
    .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }}
    .kpi strong {{ display:block; font-size:31px; letter-spacing:-.03em; }}
    .kpi span, small {{ color:var(--muted); }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px;
      background:var(--card); }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    table.wide {{ min-width:1050px; }}
    th,td {{ text-align:left; vertical-align:top; padding:13px 14px;
      border-bottom:1px solid var(--line); }}
    th {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
      background:#f9fafb; }}
    tr:last-child td {{ border-bottom:0; }}
    .decision {{ display:inline-block; padding:4px 8px; border-radius:999px;
      background:var(--red-bg); color:var(--red); font-size:12px; font-weight:800; }}
    .cases {{ display:grid; grid-template-columns:repeat(2,1fr); gap:15px; }}
    .case {{ padding:22px; background:var(--card); border:1px solid var(--line); border-radius:12px; }}
    .case-head {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
    .codes {{ color:var(--muted); font:12px/1.5 ui-monospace, SFMono-Regular, monospace; }}
    .sources {{ margin:14px 0 0; padding-left:18px; }}
    .sources li {{ margin:10px 0; }}
    .sources a {{ color:var(--blue); font-weight:700; }}
    .sources span {{ display:block; color:var(--muted); font-size:13px; }}
    .method {{ padding:22px; background:#edf4f7; border-radius:12px; }}
    footer {{ margin-top:60px; color:var(--muted); font-size:12px; }}
    @media(max-width:760px) {{
      main {{ padding:32px 16px 60px; }} .grid,.cases {{ grid-template-columns:1fr; }}
      h2 {{ margin-top:42px; }} .case-head {{ align-items:flex-start; }}
    }}
  </style>
</head>
<body><main>
  <p class="eyebrow">AlphaRank · intégrité des données · {html.escape(registry["registry_id"])}</p>
  <h1>Audit des incohérences de prix et d'identité</h1>
  <p class="lede">Chaque exclusion est appuyée par une anomalie mesurée dans le
  snapshot et par des sources externes officielles. La décision porte sur toute
  la trajectoire du symbole dans ce dataset, jamais seulement sur un mauvais mois.</p>
  <div class="verdict"><strong>Verdict.</strong> {len(audit_rows)} symboles sont mis en quarantaine
  complète. Cette correction retire les collisions identifiées, mais ne répare pas
  encore le biais plus large du fichier historique de constituants.</div>
  <div class="grid">
    <div class="kpi"><strong>{len(audit_rows)}</strong><span>tickers exclus sur toutes les dates</span></div>
    <div class="kpi"><strong>{len(screening_rows)}</strong><span>tickers détenus passés au crible</span></div>
    <div class="kpi"><strong>{len(reviews)}</strong><span>cas laissés en revue, non supprimés</span></div>
  </div>

  <h2>Règle appliquée</h2>
  <div class="method"><strong>Exclusion complète en amont.</strong> Le ticker est
  retiré des prix, fondamentaux, constituants, rangs cross-sectionnels,
  entraînements, sélections et rendements réalisés. Une sensibilité sur les
  holdings publiés est présentée ci-dessous, mais seul un réentraînement/rerun
  complet constitue le résultat corrigé.</div>

  <h2>Impact immédiat sur les holdings publiés</h2>
  <p class="lede">Ces chiffres isolent l'exposition directe en filtrant puis
  renormalisant les portefeuilles déjà sélectionnés. Ils ne sont pas un rerun
  causal complet, car les rangs et modèles peuvent changer après exclusion.</p>
  {impact_table}

  <h2>Rerun ML complet après quarantaine</h2>
  <p class="lede">Ici les dix tickers sont retirés avant les EMA, les rangs,
  l'entraînement et les sélections. Legacy reste la série publiée tant que son
  rerun Optuna complet n'est pas arrivé à terme.</p>
  {corrected_table}

  <h2>Inventaire quantitatif</h2>
  {audit_table}

  <h2>File de revue — aucune exclusion automatique</h2>
  <p class="lede">Les drapeaux servent à orienter une recherche humaine. Ils ne
  suffisent jamais seuls à supprimer un titre. Les abréviations de sociétés,
  véritables krachs ou erreurs OHLC isolées peuvent produire des faux positifs.</p>
  {review_table}

  <h2>Dossiers de preuve</h2>
  <div class="cases">{''.join(detail_cards)}</div>

  <h2>Limite résiduelle</h2>
  <p class="lede">La quarantaine est volontairement conservatrice. Elle ne rend
  pas l'univers point-in-time valide à elle seule : les dates d'entrée/sortie et
  les identifiants société stables doivent encore être reconstruits. Un ticker
  ne sera réintégré qu'après résolution CIK/CUSIP/FIGI et validation de sa série
  de prix auprès d'au moins deux sources indépendantes.</p>

  <footer>Généré depuis le snapshot 20260713_201639 · sources consultées le
  2026-07-26 · SPY en adjusted close · Sharpe selon la convention Legacy.</footer>
</main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit historical ticker identity and price integrity."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--legacy-detailed", type=Path, default=DEFAULT_LEGACY_DETAILED
    )
    parser.add_argument(
        "--legacy-monthly", type=Path, default=DEFAULT_LEGACY_MONTHLY
    )
    parser.add_argument("--ml-holdings", type=Path, default=DEFAULT_ML_HOLDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--corrected-risk-output",
        type=Path,
        default=DEFAULT_CORRECTED_RISK_OUTPUT,
    )
    args = parser.parse_args()

    registry = load_ticker_exclusion_registry(args.registry)
    prices = pl.read_parquet(args.snapshot / "US_Finalprice.parquet")
    general = pl.read_parquet(args.snapshot / "US_General.parquet")
    constituents = pl.read_csv(args.snapshot / "SP500_Constituents.csv")
    legacy_detailed = pl.read_parquet(args.legacy_detailed)
    legacy_monthly = pl.read_parquet(args.legacy_monthly)
    ml_holdings = pl.read_parquet(args.ml_holdings)

    audit_rows = _price_audit_rows(
        registry.payload["entries"],
        prices=prices,
        general=general,
        constituents=constituents,
        legacy_holdings=legacy_detailed,
        ml_holdings=ml_holdings,
    )
    screening_rows = _screen_held_universe(
        prices=prices,
        general=general,
        constituents=constituents,
        legacy_holdings=legacy_detailed,
        ml_holdings=ml_holdings,
        excluded_tickers=registry.excluded_tickers,
    )
    legacy_sensitivity = _legacy_post_selection_sensitivity(
        legacy_detailed,
        registry.excluded_tickers,
    )
    ml_sensitivity = _ml_post_selection_sensitivity(
        ml_holdings,
        registry.excluded_tickers,
    )
    performance_rows = _performance_rows(
        legacy_monthly=legacy_monthly,
        legacy_sensitivity=legacy_sensitivity,
        ml_sensitivity=ml_sensitivity,
        spy=_spy_monthly(
            pl.read_parquet(args.snapshot / "SP500Price.parquet")
        ),
    )
    corrected_performance_path = (
        args.corrected_risk_output
        / "allocation_performance_legacy_convention.csv"
    )
    corrected_performance = (
        pl.read_csv(corrected_performance_path)
        .filter(
            pl.col("series").is_in(
                [
                    "alpha_top5_equal",
                    "alpha_top5_inverse_vol_h3",
                    "Legacy",
                    "SPY total return",
                ]
            )
        )
        .to_dicts()
        if corrected_performance_path.exists()
        else []
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_audit = [
        {key: value for key, value in row.items() if key != "sources"}
        for row in audit_rows
    ]
    pl.DataFrame(serializable_audit).write_csv(
        args.output_dir / "ticker_identity_price_audit.csv"
    )
    pl.DataFrame(performance_rows).write_csv(
        args.output_dir / "portfolio_impact_sensitivity.csv"
    )
    pl.DataFrame(screening_rows).write_csv(
        args.output_dir / "holding_universe_screen.csv"
    )
    legacy_sensitivity.write_csv(
        args.output_dir / "legacy_monthly_exclusion_sensitivity.csv"
    )
    ml_sensitivity.write_csv(
        args.output_dir / "ml_v2_monthly_exclusion_sensitivity.csv"
    )
    manifest = {
        "registry_id": registry.registry_id,
        "registry_path": str(registry.path),
        "registry_sha256": _sha256(registry.path),
        "excluded_tickers": list(registry.excluded_tickers),
        "snapshot": str(args.snapshot.resolve()),
        "legacy_detailed": str(args.legacy_detailed.resolve()),
        "legacy_monthly": str(args.legacy_monthly.resolve()),
        "ml_holdings": str(args.ml_holdings.resolve()),
        "corrected_risk_output": str(args.corrected_risk_output.resolve()),
        "input_sha256": {
            "prices": _sha256(args.snapshot / "US_Finalprice.parquet"),
            "general": _sha256(args.snapshot / "US_General.parquet"),
            "constituents": _sha256(args.snapshot / "SP500_Constituents.csv"),
            "legacy_detailed": _sha256(args.legacy_detailed),
            "legacy_monthly": _sha256(args.legacy_monthly),
            "ml_holdings": _sha256(args.ml_holdings),
        },
        "outputs": [
            "ticker_identity_price_audit.csv",
            "portfolio_impact_sensitivity.csv",
            "holding_universe_screen.csv",
            "legacy_monthly_exclusion_sensitivity.csv",
            "ml_v2_monthly_exclusion_sensitivity.csv",
            "price_identity_audit.html",
        ],
        "methodology": {
            "exclusion_scope": "all dates and all pipeline stages",
            "performance_impact": "post-selection sensitivity; not a complete rerun",
            "benchmark": "SPY adjusted close",
            "sharpe": "(CAGR - 2%) / annualized volatility",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _render_html(
        registry=registry.payload,
        audit_rows=audit_rows,
        performance_rows=performance_rows,
        screening_rows=screening_rows,
        corrected_performance=corrected_performance,
        output_path=args.output_dir / "price_identity_audit.html",
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
