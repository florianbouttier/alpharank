from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import polars as pl

from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    PRICE_LINEAGE_COLUMNS,
    PRICE_VALUE_COLUMNS,
    PriceGatePolicy,
    PRODUCTION_PRICE_GATE_POLICY,
)


@dataclass(frozen=True)
class HybridPriceResult:
    prices: pl.DataFrame
    lineage: pl.DataFrame
    composition_report: dict[str, object]


def compose_hybrid_price_history(
    *,
    eodhd_seed: pl.DataFrame,
    active_yahoo_vintage: pl.DataFrame,
    retained_open_history: pl.DataFrame | None,
    active_tickers: Sequence[str],
    policy: PriceGatePolicy = PRODUCTION_PRICE_GATE_POLICY,
) -> HybridPriceResult:
    """Compose one Yahoo vintage for active names over an immutable EODHD base."""

    active = {_normalize_ticker(ticker) for ticker in active_tickers}
    seed = _ensure_lineage(eodhd_seed)
    yahoo = _ensure_lineage(active_yahoo_vintage).filter(
        pl.col("ticker").is_in(sorted(active))
    )
    retained = (
        _ensure_lineage(retained_open_history)
        if retained_open_history is not None and not retained_open_history.is_empty()
        else pl.DataFrame(schema=seed.schema)
    )

    inactive_seed = seed.filter(~pl.col("ticker").is_in(sorted(active)))
    active_yahoo_tickers = (
        set(yahoo.get_column("ticker").unique().to_list())
        if not yahoo.is_empty()
        else set()
    )
    missing_active = sorted(active - active_yahoo_tickers)

    # Active names must use exactly the fresh full Yahoo vintage. EODHD is not
    # allowed to fill holes because that would reintroduce a mixed adjustment basis.
    selected_frames = [yahoo] if not yahoo.is_empty() else []
    unresolved_tails: list[dict[str, object]] = []
    bridge_rows: list[dict[str, object]] = []
    inactive_selected: list[pl.DataFrame] = []
    retained_inactive = retained.filter(~pl.col("ticker").is_in(sorted(active)))
    retained_by_ticker = {
        key[0] if isinstance(key, tuple) else key: frame
        for key, frame in retained_inactive.partition_by("ticker", as_dict=True).items()
    }

    for key, ticker_seed in inactive_seed.partition_by("ticker", as_dict=True).items():
        ticker = key[0] if isinstance(key, tuple) else key
        open_history = retained_by_ticker.get(ticker)
        if open_history is None or open_history.is_empty():
            inactive_selected.append(ticker_seed)
            continue
        seed_last = ticker_seed.select(pl.col("date").max()).item()
        tail_candidates = open_history.filter(pl.col("date") > seed_last)
        if tail_candidates.is_empty():
            inactive_selected.append(ticker_seed)
            continue
        extension, bridge = _build_return_ledger_extension(
            ticker_seed=ticker_seed,
            open_history=open_history,
            seed_last=seed_last,
            policy=policy,
        )
        if extension is None:
            unresolved_tails.append({"ticker": ticker, **bridge})
            inactive_selected.append(ticker_seed)
            continue
        inactive_selected.extend([ticker_seed, extension])
        bridge_rows.append({"ticker": ticker, **bridge})

    if inactive_selected:
        selected_frames.extend(inactive_selected)
    lineage = (
        pl.concat(selected_frames, how="diagonal_relaxed")
        .sort(["ticker", "date", "source", "source_vintage_id"])
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
        .sort(["ticker", "date"])
        .select(PRICE_LINEAGE_COLUMNS)
    )
    prices = lineage.select(PRICE_VALUE_COLUMNS)
    report = {
        "contract_version": 1,
        "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
        "active_ticker_count": len(active),
        "active_yahoo_ticker_count": len(active_yahoo_tickers),
        "missing_active_yahoo_tickers": missing_active,
        "eodhd_seed_rows_selected": lineage.filter(
            pl.col("source") == "eodhd_frozen_history"
        ).height,
        "eodhd_seed_tickers_selected": lineage.filter(
            pl.col("source") == "eodhd_frozen_history"
        )
        .select(pl.col("ticker").n_unique())
        .item(),
        "bridged_inactive_tickers": len(bridge_rows),
        "bridges": bridge_rows,
        "unresolved_inactive_tails": unresolved_tails,
    }
    return HybridPriceResult(prices=prices, lineage=lineage, composition_report=report)


def roll_forward_validated_price_history(
    *,
    previous_validated_lineage: pl.DataFrame,
    active_yahoo_vintage: pl.DataFrame,
    active_tickers: Sequence[str],
    preserved_terminal_tickers: Sequence[str] = (),
) -> HybridPriceResult:
    """Keep validated inactive history unchanged and replace every active name."""

    active = {_normalize_ticker(ticker) for ticker in active_tickers}
    terminal = {_normalize_ticker(ticker) for ticker in preserved_terminal_tickers}
    invalid_terminal = sorted(terminal - active)
    if invalid_terminal:
        raise RuntimeError(
            f"Terminal preservation exceptions are not in the active snapshot: {invalid_terminal}"
        )
    refreshable_active = active - terminal
    previous = _ensure_lineage(previous_validated_lineage)
    yahoo = _ensure_lineage(active_yahoo_vintage).filter(
        pl.col("ticker").is_in(sorted(refreshable_active))
    )
    yahoo_tickers = set(yahoo.get_column("ticker").unique().to_list())
    missing_active = sorted(refreshable_active - yahoo_tickers)
    if missing_active:
        raise RuntimeError(
            "Fresh Yahoo vintage does not cover every active ticker: "
            f"{missing_active[:20]}"
        )
    vintages = yahoo.select(pl.col("source_vintage_id").drop_nulls().unique())
    if vintages.height != 1:
        raise RuntimeError(
            "Fresh active universe must use exactly one Yahoo vintage; "
            f"found={vintages.height}"
        )
    non_yahoo = yahoo.filter(pl.col("source") != "yfinance").height
    if non_yahoo:
        raise RuntimeError(f"Fresh active universe contains {non_yahoo} non-Yahoo rows")

    preserved = previous.filter(~pl.col("ticker").is_in(sorted(refreshable_active)))
    preserved_eodhd_tickers = (
        preserved.filter(pl.col("source") == "eodhd_frozen_history")
        .get_column("ticker")
        .unique()
        .to_list()
    )
    preserved_open_source_only = preserved.filter(
        ~pl.col("ticker").is_in(preserved_eodhd_tickers)
    )
    lineage = (
        pl.concat([preserved, yahoo], how="diagonal_relaxed")
        .sort(["ticker", "date"])
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
        .select(PRICE_LINEAGE_COLUMNS)
    )
    expected_preserved = preserved.select(PRICE_LINEAGE_COLUMNS).sort(["ticker", "date"])
    observed_preserved = lineage.filter(
        ~pl.col("ticker").is_in(sorted(refreshable_active))
    ).sort(["ticker", "date"])
    if not expected_preserved.equals(observed_preserved, null_equal=True):
        raise RuntimeError("Preserved validated price history changed during roll-forward")

    return HybridPriceResult(
        prices=lineage.select(PRICE_VALUE_COLUMNS),
        lineage=lineage,
        composition_report={
            "contract_version": 1,
            "mode": "validated_snapshot_roll_forward",
            "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
            "previous_rows": previous.height,
            "preserved_history_rows": preserved.height,
            "preserved_history_tickers": preserved.select(
                pl.col("ticker").n_unique()
            ).item(),
            "preserved_open_source_only_rows": preserved_open_source_only.height,
            "preserved_open_source_only_tickers": preserved_open_source_only.select(
                pl.col("ticker").n_unique()
            ).item(),
            "active_ticker_count": len(active),
            "refreshable_active_ticker_count": len(refreshable_active),
            "preserved_terminal_tickers": sorted(terminal),
            "active_yahoo_rows": yahoo.height,
            "active_yahoo_ticker_count": len(yahoo_tickers),
            "active_yahoo_vintage_id": vintages.item(),
            "candidate_rows": lineage.height,
            "candidate_tickers": lineage.select(pl.col("ticker").n_unique()).item(),
            "missing_active_yahoo_tickers": [],
        },
    )


def _build_return_ledger_extension(
    *,
    ticker_seed: pl.DataFrame,
    open_history: pl.DataFrame,
    seed_last: str,
    policy: PriceGatePolicy,
) -> tuple[pl.DataFrame | None, dict[str, object]]:
    tail_dates = (
        open_history.filter(pl.col("date") > seed_last)
        .select("date")
        .unique()
        .sort("date")
    )
    first_tail = tail_dates.select(pl.col("date").min()).item()
    gap_days = (date.fromisoformat(first_tail) - date.fromisoformat(seed_last)).days
    if gap_days > policy.maximum_bridge_gap_calendar_days:
        return None, {
            "reason": "tail_starts_after_symbol_gap",
            "seed_last_date": seed_last,
            "tail_first_date": first_tail,
            "gap_calendar_days": gap_days,
        }

    vintage_rows = (
        open_history.sort(
            ["source", "source_vintage_id", "date", "ingested_at"]
        )
        .unique(
            subset=["source", "source_vintage_id", "date"],
            keep="last",
            maintain_order=True,
        )
        .sort(["source", "source_vintage_id", "date"])
        .with_columns(
            pl.col("adjusted_close")
            .pct_change()
            .over(["source", "source_vintage_id"])
            .alias("source_daily_return")
        )
    )
    return_rows = (
        vintage_rows.filter(
            (pl.col("date") > seed_last)
            & pl.col("source_daily_return").is_finite()
            & (pl.col("source_daily_return") > -1.0)
        )
        .sort(["date", "ingested_at", "source", "source_vintage_id"])
        .unique(subset=["date"], keep="last", maintain_order=True)
        .sort("date")
    )
    missing_return_dates = tail_dates.join(
        return_rows.select("date"), on="date", how="anti"
    )
    if missing_return_dates.height:
        return None, {
            "reason": "incomplete_return_ledger",
            "seed_last_date": seed_last,
            "tail_first_date": first_tail,
            "tail_date_count": tail_dates.height,
            "missing_return_date_count": missing_return_dates.height,
            "missing_return_date_examples": missing_return_dates.head(10)
            .get_column("date")
            .to_list(),
        }

    anchor = ticker_seed.sort("date").tail(1).row(0, named=True)
    anchor_adjusted = float(anchor["adjusted_close"])
    anchor_close = float(anchor["close"])
    anchor_adjustment_factor = (
        anchor_adjusted / anchor_close if anchor_close else 1.0
    )
    extension = return_rows.with_columns(
        (pl.lit(anchor_adjusted) * (pl.col("source_daily_return") + 1.0).cum_prod())
        .alias("synthetic_adjusted_close"),
        pl.col("source_vintage_id").alias("underlying_return_vintage"),
    ).with_columns(
        (pl.col("synthetic_adjusted_close") / pl.col("adjusted_close"))
        .alias("adjustment_bridge_factor"),
        (pl.col("synthetic_adjusted_close") / anchor_adjustment_factor)
        .alias("synthetic_close"),
    )
    for column in ("open", "high", "low"):
        extension = extension.with_columns(
            pl.when(pl.col("close").is_not_null() & (pl.col("close") != 0))
            .then(pl.col(column) / pl.col("close") * pl.col("synthetic_close"))
            .otherwise(None)
            .alias(column)
        )
    ledger_id = "eodhd_yahoo_return_ledger_v1"
    extension = extension.with_columns(
        pl.col("synthetic_close").alias("close"),
        pl.col("synthetic_adjusted_close").alias("adjusted_close"),
        pl.lit("yfinance_return_ledger").alias("source"),
        pl.lit("prices_yfinance_return_ledger").alias("dataset"),
        pl.lit(ledger_id).alias("source_vintage_id"),
        pl.col("underlying_return_vintage").alias("return_source_vintage_id"),
        pl.lit(ADJUSTMENT_POLICY_VERSION).alias("adjustment_policy_version"),
        pl.lit(anchor["eodhd_seed_sha256"]).alias("eodhd_seed_sha256"),
    ).select(PRICE_LINEAGE_COLUMNS)
    return extension, {
        "method": "same_vintage_daily_return_ledger",
        "source_vintage_id": ledger_id,
        "seed_last_date": seed_last,
        "tail_first_date": first_tail,
        "tail_last_date": extension.select(pl.col("date").max()).item(),
        "tail_rows": extension.height,
        "underlying_vintage_count": extension.select(
            pl.col("return_source_vintage_id").n_unique()
        ).item(),
    }


def _ensure_lineage(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={column: pl.String for column in PRICE_LINEAGE_COLUMNS}
        )
    date_expr = (
        pl.col("date").str.to_date(strict=False)
        if frame.schema.get("date") == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    normalized = frame.with_columns(
        date_expr.dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("adjusted_close").cast(pl.Float64, strict=False),
    ).filter(
        pl.col("adjusted_close").is_not_null() & (pl.col("adjusted_close") > 0)
    )
    defaults: dict[str, pl.Expr] = {
        "source_vintage_id": (
            pl.col("ingestion_run_id")
            if "ingestion_run_id" in normalized.columns
            else pl.lit("unknown")
        ),
        "return_source_vintage_id": (
            pl.col("ingestion_run_id")
            if "ingestion_run_id" in normalized.columns
            else pl.lit("unknown")
        ),
        "adjustment_policy_version": pl.lit(ADJUSTMENT_POLICY_VERSION),
        "adjustment_bridge_factor": pl.lit(1.0),
        "eodhd_seed_sha256": pl.lit(None).cast(pl.String),
        "correction_overlay_id": pl.lit(None).cast(pl.String),
    }
    for column, expression in defaults.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(expression.alias(column))
    missing = set(PRICE_LINEAGE_COLUMNS) - set(normalized.columns)
    if missing:
        raise ValueError(f"Price lineage frame is missing columns: {sorted(missing)}")
    return normalized.select(PRICE_LINEAGE_COLUMNS)


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).upper()
    return value if value.endswith(".US") else f"{value}.US"
