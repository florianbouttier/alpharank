from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class FinancialSourceInput:
    source_name: str
    frame: pl.DataFrame
    priority: int


def consolidate_financial_sources(
    sources: Iterable[FinancialSourceInput],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source_list = sorted((source for source in sources if not source.frame.is_empty()), key=lambda item: item.priority)
    if not source_list:
        empty = _empty_consolidated_frame()
        return empty, _empty_lineage_frame(), _empty_source_summary()

    candidates = _build_candidate_frame(source_list)
    selection_key = ["ticker", "statement", "metric", "date"]
    selected = (
        candidates.sort(selection_key + ["source_priority", "filing_date"], descending=[False, False, False, False, False, False])
        .unique(subset=selection_key, keep="first", maintain_order=True)
        .sort(selection_key)
    )

    lineage_rollup = candidates.group_by(selection_key).agg(
        [
            pl.len().alias("candidate_source_count"),
            pl.col("source").sort_by("source_priority").implode().list.join(" | ").alias("candidate_sources"),
            pl.col("source_label").sort_by("source_priority").implode().list.join(" | ").alias("candidate_source_labels"),
        ]
    )
    consolidated = (
        selected.join(lineage_rollup, on=selection_key, how="left", coalesce=True)
        .with_columns(
            [
                pl.lit("open_source_consolidated").alias("source"),
                (pl.col("source_priority") > pl.lit(source_list[0].priority)).alias("fallback_used"),
            ]
        )
        .select(_consolidated_columns())
        .sort(selection_key)
    )

    source_summary = (
        consolidated.group_by(["statement", "selected_source"])
        .agg(
            [
                pl.len().alias("selected_rows"),
                pl.col("fallback_used").sum().cast(pl.Int64).alias("fallback_rows"),
                pl.col("ticker").n_unique().alias("ticker_count"),
                pl.col("metric").n_unique().alias("metric_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("selected_rows") > 0)
            .then((pl.col("fallback_rows") / pl.col("selected_rows")) * 100.0)
            .otherwise(0.0)
            .alias("fallback_rate_pct")
        )
        .sort(["statement", "selected_rows", "selected_source"], descending=[False, True, False])
    )

    return consolidated, candidates, source_summary


def split_consolidated_by_statement(consolidated: pl.DataFrame) -> dict[str, pl.DataFrame]:
    statement_frames: dict[str, pl.DataFrame] = {}
    if consolidated.is_empty():
        for statement in ("income_statement", "balance_sheet", "cash_flow", "shares"):
            statement_frames[statement] = consolidated.clone()
        return statement_frames
    for statement in ("income_statement", "balance_sheet", "cash_flow", "shares"):
        statement_frames[statement] = consolidated.filter(pl.col("statement") == statement).sort(["ticker", "metric", "date"])
    return statement_frames


def _build_candidate_frame(sources: list[FinancialSourceInput]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for source in sources:
        prepared = _ensure_lineage_columns(source.frame)
        frames.append(
            prepared.with_columns(
                [
                    pl.lit(source.priority).alias("source_priority"),
                    pl.col("source").alias("selected_source"),
                    pl.col("source_label").alias("selected_source_label"),
                    pl.col("form").alias("selected_form"),
                    pl.col("fiscal_period").alias("selected_fiscal_period"),
                    pl.col("fiscal_year").alias("selected_fiscal_year"),
                ]
            ).select(_lineage_columns())
        )
    return pl.concat(frames, how="vertical").sort(["ticker", "statement", "metric", "date", "source_priority"])


def _consolidated_columns() -> list[str]:
    return [
        "ticker",
        "statement",
        "metric",
        "date",
        "filing_date",
        "value",
        "source",
        "source_label",
        "selected_source",
        "selected_source_label",
        "selected_form",
        "selected_fiscal_period",
        "selected_fiscal_year",
        "source_priority",
        "fallback_used",
        "candidate_source_count",
        "candidate_sources",
        "candidate_source_labels",
    ]


def _lineage_columns() -> list[str]:
    return [
        "ticker",
        "statement",
        "metric",
        "date",
        "filing_date",
        "value",
        "source",
        "source_label",
        "selected_source",
        "selected_source_label",
        "selected_form",
        "selected_fiscal_period",
        "selected_fiscal_year",
        "source_priority",
    ]


def _empty_consolidated_frame() -> pl.DataFrame:
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
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
            "fallback_used": pl.Boolean,
            "candidate_source_count": pl.Int64,
            "candidate_sources": pl.String,
            "candidate_source_labels": pl.String,
        }
    )


def _empty_lineage_frame() -> pl.DataFrame:
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
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )


def _empty_source_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "statement": pl.String,
            "selected_source": pl.String,
            "selected_rows": pl.Int64,
            "fallback_rows": pl.Int64,
            "ticker_count": pl.Int64,
            "metric_count": pl.Int64,
            "fallback_rate_pct": pl.Float64,
        }
    )


def _ensure_lineage_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    if "form" not in frame.columns:
        expressions.append(pl.lit(None).cast(pl.Utf8).alias("form"))
    if "fiscal_period" not in frame.columns:
        expressions.append(pl.lit(None).cast(pl.Utf8).alias("fiscal_period"))
    if "fiscal_year" not in frame.columns:
        expressions.append(pl.lit(None).cast(pl.Int64).alias("fiscal_year"))
    return frame.with_columns(expressions) if expressions else frame
