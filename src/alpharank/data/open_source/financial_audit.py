from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from alpharank.data.open_source.benchmark import resolve_eodhd_output_dir
from alpharank.data.open_source.config import METRIC_SPECS

AUDITED_FINANCIAL_STATEMENTS: tuple[str, ...] = (
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "shares",
)

HISTOGRAM_BINS: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class FinancialAuditResult:
    output_dir: Path
    dashboard_path: Path
    summary_md_path: Path
    summary_json_path: Path
    alignment_path: Path
    issue_details_path: Path
    statement_summary_path: Path
    metric_summary_path: Path
    ticker_summary_path: Path
    quarter_summary_path: Path
    ticker_quarter_summary_path: Path
    total_rows: int
    matched_rows: int
    error_rows: int
    missing_open_rows: int
    extra_open_rows: int
    reference_rows: int
    error_ticker_quarters: int
    missing_open_ticker_quarters: int
    total_reference_ticker_quarters: int


def build_financial_statement_audit_dashboard(
    *,
    eodhd_dir: Path,
    open_source_dir: Path,
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    threshold_pct: float = 1.0,
    statements: Sequence[str] = AUDITED_FINANCIAL_STATEMENTS,
) -> FinancialAuditResult:
    end_date = end_date or "9999-12-31"
    normalized_eodhd = normalize_output_financials_between(
        eodhd_dir,
        source_name="eodhd_output",
        start_date=start_date,
        end_date=end_date,
        statements=statements,
    )
    normalized_open = normalize_output_financials_between(
        open_source_dir,
        source_name="open_source_output",
        start_date=start_date,
        end_date=end_date,
        statements=statements,
    )
    return build_financial_statement_audit_dashboard_from_frames(
        eodhd_financials=normalized_eodhd,
        open_financials=normalized_open,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        threshold_pct=threshold_pct,
    )


def build_financial_statement_audit_dashboard_from_frames(
    *,
    eodhd_financials: pl.DataFrame,
    open_financials: pl.DataFrame,
    output_dir: Path,
    start_date: str,
    end_date: str | None = None,
    threshold_pct: float = 1.0,
) -> FinancialAuditResult:
    end_date = end_date or "9999-12-31"
    alignment = build_financial_alignment_with_metadata(
        eodhd_frame=eodhd_financials,
        open_frame=open_financials,
        open_source="open_source_output",
        tolerance_days=10,
    ).with_columns(
        [
            _issue_kind_expr(threshold_pct).alias("issue_kind"),
            _abs_diff_expr().alias("abs_diff_pct"),
            _quarter_label_expr("date").alias("quarter_label"),
            _quarter_label_expr("open_date").alias("open_quarter_label"),
            pl.concat_str(
                [
                    pl.col("ticker").fill_null(""),
                    pl.lit("|"),
                    pl.col("date").fill_null(""),
                ]
            ).alias("ticker_quarter_key"),
        ]
    )

    statement_summary = _aggregate_alignment(alignment, by=["statement"], threshold_pct=threshold_pct)
    metric_summary = _aggregate_alignment(alignment, by=["statement", "metric"], threshold_pct=threshold_pct)
    ticker_summary = _aggregate_alignment(alignment, by=["ticker"], threshold_pct=threshold_pct)
    quarter_summary = _aggregate_alignment(alignment, by=["quarter_label"], threshold_pct=threshold_pct)
    ticker_quarter_summary = _build_ticker_quarter_summary(alignment, threshold_pct=threshold_pct)
    issue_details = alignment.filter(pl.col("issue_kind") != "within_threshold").sort(
        ["statement", "metric", "ticker", "date"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = output_dir / "financial_alignment_since_2020.parquet"
    issue_details_path = output_dir / "financial_issue_details_since_2020.parquet"
    statement_summary_path = output_dir / "statement_summary_since_2020.parquet"
    metric_summary_path = output_dir / "metric_summary_since_2020.parquet"
    ticker_summary_path = output_dir / "ticker_summary_since_2020.parquet"
    quarter_summary_path = output_dir / "quarter_summary_since_2020.parquet"
    ticker_quarter_summary_path = output_dir / "ticker_quarter_summary_since_2020.parquet"
    dashboard_path = output_dir / "dashboard.html"
    summary_md_path = output_dir / "summary.md"
    summary_json_path = output_dir / "summary.json"

    alignment.write_parquet(alignment_path)
    issue_details.write_parquet(issue_details_path)
    statement_summary.write_parquet(statement_summary_path)
    metric_summary.write_parquet(metric_summary_path)
    ticker_summary.write_parquet(ticker_summary_path)
    quarter_summary.write_parquet(quarter_summary_path)
    ticker_quarter_summary.write_parquet(ticker_quarter_summary_path)

    summary = _build_summary_payload(
        alignment=alignment,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
        ticker_summary=ticker_summary,
        ticker_quarter_summary=ticker_quarter_summary,
        threshold_pct=threshold_pct,
        start_date=start_date,
        end_date=end_date,
    )
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    dashboard_path.write_text(
        _render_dashboard_html(
            summary=summary,
            alignment=alignment,
            statement_summary=statement_summary,
            metric_summary=metric_summary,
            ticker_summary=ticker_summary,
            quarter_summary=quarter_summary,
            ticker_quarter_summary=ticker_quarter_summary,
            threshold_pct=threshold_pct,
            start_date=start_date,
            end_date=end_date,
        ),
        encoding="utf-8",
    )

    return FinancialAuditResult(
        output_dir=output_dir,
        dashboard_path=dashboard_path,
        summary_md_path=summary_md_path,
        summary_json_path=summary_json_path,
        alignment_path=alignment_path,
        issue_details_path=issue_details_path,
        statement_summary_path=statement_summary_path,
        metric_summary_path=metric_summary_path,
        ticker_summary_path=ticker_summary_path,
        quarter_summary_path=quarter_summary_path,
        ticker_quarter_summary_path=ticker_quarter_summary_path,
        total_rows=int(summary["global"]["total_rows"]),
        matched_rows=int(summary["global"]["matched_rows"]),
        error_rows=int(summary["global"]["error_rows"]),
        missing_open_rows=int(summary["global"]["missing_open_rows"]),
        extra_open_rows=int(summary["global"]["extra_open_rows"]),
        reference_rows=int(summary["global"]["reference_rows"]),
        error_ticker_quarters=int(summary["global"]["error_ticker_quarters"]),
        missing_open_ticker_quarters=int(summary["global"]["missing_open_ticker_quarters"]),
        total_reference_ticker_quarters=int(summary["global"]["reference_ticker_quarters"]),
    )


def normalize_output_financials_between(
    data_dir: Path,
    *,
    source_name: str,
    start_date: str,
    end_date: str,
    statements: Sequence[str] = AUDITED_FINANCIAL_STATEMENTS,
    tickers: Iterable[str] | None = None,
) -> pl.DataFrame:
    resolved_dir = resolve_eodhd_output_dir(data_dir)
    ticker_filter = None
    if tickers is not None:
        ticker_filter = [ticker if ticker.endswith(".US") else f"{ticker}.US" for ticker in tickers]

    lineage = _load_optional_open_lineage(resolved_dir) if source_name != "eodhd_output" else None

    frames: list[pl.DataFrame] = []
    statement_set = set(statements)
    unique_paths = sorted({spec.eodhd_path for spec in METRIC_SPECS if spec.statement in statement_set})
    for parquet_name in unique_paths:
        frame_path = resolved_dir / parquet_name
        if not frame_path.exists():
            continue
        frame = pl.read_parquet(frame_path)
        date_col = "dateFormatted" if parquet_name == "US_share.parquet" else "date"
        filing_col = "dateFormatted" if parquet_name == "US_share.parquet" else "filing_date"
        if date_col not in frame.columns:
            continue
        if ticker_filter is not None and "ticker" in frame.columns:
            frame = frame.filter(pl.col("ticker").is_in(ticker_filter))
        frame = frame.filter(
            (pl.col(date_col).cast(pl.Utf8, strict=False) >= pl.lit(start_date))
            & (pl.col(date_col).cast(pl.Utf8, strict=False) <= pl.lit(end_date))
        )
        if frame.is_empty():
            continue
        for spec in [item for item in METRIC_SPECS if item.eodhd_path == parquet_name and item.statement in statement_set]:
            if spec.eodhd_column not in frame.columns:
                continue
            frames.append(
                frame.select(
                    [
                        pl.col("ticker").cast(pl.Utf8, strict=False),
                        pl.lit(spec.statement).alias("statement"),
                        pl.lit(spec.metric).alias("metric"),
                        pl.col(date_col).cast(pl.Utf8, strict=False).alias("date"),
                        pl.col(filing_col).cast(pl.Utf8, strict=False).alias("filing_date"),
                        pl.col(spec.eodhd_column).cast(pl.Float64, strict=False).alias("value"),
                        pl.lit(source_name).alias("source"),
                        pl.lit(spec.eodhd_column).alias("source_label"),
                    ]
                ).filter(pl.col("value").is_not_null())
            )
    normalized = pl.concat(frames, how="vertical") if frames else _empty_financial_frame()
    normalized = normalized.sort(["ticker", "statement", "metric", "date"])
    if lineage is None or normalized.is_empty():
        return normalized.with_columns(
            [
                pl.lit(source_name).alias("selected_source"),
                pl.col("source_label").alias("selected_source_label"),
                pl.lit(source_name).alias("candidate_sources"),
                pl.lit(None).cast(pl.Utf8).alias("selected_fiscal_period"),
                pl.lit(None).cast(pl.Int64).alias("selected_fiscal_year"),
            ]
        )
    return (
        normalized.join(lineage, on=["ticker", "statement", "metric", "date"], how="left", coalesce=True)
        .with_columns(
            [
                pl.col("selected_source").fill_null(source_name),
                pl.col("selected_source_label").fill_null(pl.col("source_label")),
                pl.col("candidate_sources").fill_null(pl.col("selected_source")),
            ]
        )
    )


def build_financial_alignment_with_metadata(
    *,
    eodhd_frame: pl.DataFrame,
    open_frame: pl.DataFrame,
    open_source: str,
    tolerance_days: int,
) -> pl.DataFrame:
    key_cols = ["ticker", "statement", "metric"]
    eod = (
        eodhd_frame.select(key_cols + ["date", "filing_date", "value", "source_label"])
        .rename(
            {
                "date": "eodhd_date",
                "filing_date": "eodhd_filing_date",
                "value": "eodhd_value",
                "source_label": "eodhd_source_label",
            }
        )
        .with_row_index("eodhd_row_id")
        .with_columns(_parsed_date_expr("eodhd_date").alias("eodhd_date_dt"))
    )
    opn = (
        open_frame.select(
            key_cols
            + [
                "date",
                "filing_date",
                "value",
                "source_label",
                "selected_source",
                "selected_source_label",
                "candidate_sources",
                "selected_fiscal_period",
                "selected_fiscal_year",
            ]
        )
        .rename(
            {
                "date": "open_date",
                "filing_date": "open_filing_date",
                "value": "open_value",
                "source_label": "open_source_label",
            }
        )
        .with_row_index("open_row_id")
        .with_columns(_parsed_date_expr("open_date").alias("open_date_dt"))
    )

    candidate_matches = (
        eod.join(opn, on=key_cols, how="inner")
        .with_columns((pl.col("open_date_dt") - pl.col("eodhd_date_dt")).dt.total_days().alias("date_diff_days"))
        .with_columns(pl.col("date_diff_days").abs().alias("abs_date_diff_days"))
        .filter(pl.col("abs_date_diff_days") <= tolerance_days)
        .sort(["abs_date_diff_days", "eodhd_row_id", "open_row_id"])
    )

    matches: list[dict[str, object]] = []
    used_eodhd: set[int] = set()
    used_open: set[int] = set()
    for row in candidate_matches.iter_rows(named=True):
        eodhd_row_id = int(row["eodhd_row_id"])
        open_row_id = int(row["open_row_id"])
        if eodhd_row_id in used_eodhd or open_row_id in used_open:
            continue
        used_eodhd.add(eodhd_row_id)
        used_open.add(open_row_id)
        matches.append(row)

    matched = pl.DataFrame(matches) if matches else pl.DataFrame(schema=_alignment_schema())
    matched_rows = (
        matched.select(
            key_cols
            + [
                pl.col("eodhd_date").alias("date"),
                pl.col("eodhd_date"),
                pl.col("open_date"),
                pl.col("eodhd_filing_date"),
                pl.col("open_filing_date"),
                pl.col("eodhd_value"),
                pl.col("open_value"),
                pl.col("eodhd_source_label"),
                pl.col("open_source_label"),
                pl.col("selected_source"),
                pl.col("selected_source_label"),
                pl.col("candidate_sources"),
                pl.col("selected_fiscal_period"),
                pl.col("selected_fiscal_year"),
                pl.col("date_diff_days"),
            ]
        )
        if not matched.is_empty()
        else _empty_alignment_frame()
    )

    matched_rows = matched_rows.with_columns(
        [
            pl.lit(open_source).alias("source"),
            pl.lit("matched").alias("match_status"),
            (pl.col("open_value") - pl.col("eodhd_value")).alias("value_diff"),
        ]
    ).with_columns(
        pl.when(pl.col("eodhd_value").abs() > 0)
        .then((pl.col("value_diff") / pl.col("eodhd_value")) * 100.0)
        .otherwise(None)
        .alias("diff_pct")
    )

    eodhd_only = (
        eod.filter(~pl.col("eodhd_row_id").is_in(list(used_eodhd)))
        .select(
            key_cols
            + [
                pl.col("eodhd_date").alias("date"),
                pl.col("eodhd_date"),
                pl.lit(None).cast(pl.Utf8).alias("open_date"),
                pl.col("eodhd_filing_date"),
                pl.lit(None).cast(pl.Utf8).alias("open_filing_date"),
                pl.col("eodhd_value"),
                pl.lit(None).cast(pl.Float64).alias("open_value"),
                pl.col("eodhd_source_label"),
                pl.lit(None).cast(pl.Utf8).alias("open_source_label"),
                pl.lit(None).cast(pl.Utf8).alias("selected_source"),
                pl.lit(None).cast(pl.Utf8).alias("selected_source_label"),
                pl.lit(None).cast(pl.Utf8).alias("candidate_sources"),
                pl.lit(None).cast(pl.Utf8).alias("selected_fiscal_period"),
                pl.lit(None).cast(pl.Int64).alias("selected_fiscal_year"),
                pl.lit(None).cast(pl.Int64).alias("date_diff_days"),
            ]
        )
        .with_columns(
            [
                pl.lit(open_source).alias("source"),
                pl.lit("missing_in_open_source").alias("match_status"),
                pl.lit(None).cast(pl.Float64).alias("value_diff"),
                pl.lit(None).cast(pl.Float64).alias("diff_pct"),
            ]
        )
    )

    open_only = (
        opn.filter(~pl.col("open_row_id").is_in(list(used_open)))
        .select(
            key_cols
            + [
                pl.col("open_date").alias("date"),
                pl.lit(None).cast(pl.Utf8).alias("eodhd_date"),
                pl.col("open_date"),
                pl.lit(None).cast(pl.Utf8).alias("eodhd_filing_date"),
                pl.col("open_filing_date"),
                pl.lit(None).cast(pl.Float64).alias("eodhd_value"),
                pl.col("open_value"),
                pl.lit(None).cast(pl.Utf8).alias("eodhd_source_label"),
                pl.col("open_source_label"),
                pl.col("selected_source"),
                pl.col("selected_source_label"),
                pl.col("candidate_sources"),
                pl.col("selected_fiscal_period"),
                pl.col("selected_fiscal_year"),
                pl.lit(None).cast(pl.Int64).alias("date_diff_days"),
            ]
        )
        .with_columns(
            [
                pl.lit(open_source).alias("source"),
                pl.lit("extra_in_open_source").alias("match_status"),
                pl.lit(None).cast(pl.Float64).alias("value_diff"),
                pl.lit(None).cast(pl.Float64).alias("diff_pct"),
            ]
        )
    )

    return pl.concat([matched_rows, eodhd_only, open_only], how="vertical").sort(key_cols + ["date"])


def _load_optional_open_lineage(resolved_output_dir: Path) -> pl.DataFrame | None:
    lineage_path = resolved_output_dir / "lineage" / "financials_open_source_consolidated.parquet"
    if not lineage_path.exists():
        return None
    lineage = pl.read_parquet(lineage_path)
    if lineage.is_empty():
        return None
    return lineage.select(
        [
            "ticker",
            "statement",
            "metric",
            "date",
            "selected_source",
            "selected_source_label",
            "candidate_sources",
            "selected_fiscal_period",
            "selected_fiscal_year",
        ]
    ).unique(subset=["ticker", "statement", "metric", "date"], keep="last")


def _aggregate_alignment(df: pl.DataFrame, *, by: list[str], threshold_pct: float) -> pl.DataFrame:
    return (
        df.group_by(by)
        .agg(
            [
                _count_expr(pl.col("match_status") == "matched").alias("matched_rows"),
                _count_expr((pl.col("match_status") == "matched") & (pl.col("diff_pct").abs() > threshold_pct)).alias(
                    "error_rows"
                ),
                _count_expr(pl.col("match_status") == "missing_in_open_source").alias("missing_open_rows"),
                _count_expr(pl.col("match_status") == "extra_in_open_source").alias("extra_open_rows"),
                pl.col("abs_diff_pct").max().alias("max_abs_diff_pct"),
            ]
        )
        .with_columns(
            [
                (pl.col("matched_rows") + pl.col("missing_open_rows")).alias("reference_rows"),
                (pl.col("matched_rows") + pl.col("missing_open_rows") + pl.col("extra_open_rows")).alias("total_rows"),
            ]
        )
        .with_columns(
            [
                _safe_pct_expr("matched_rows", "error_rows").alias("error_rate_pct"),
                _safe_pct_expr("reference_rows", "missing_open_rows").alias("missing_open_rate_pct"),
                _safe_pct_expr("total_rows", "extra_open_rows").alias("extra_open_rate_pct"),
            ]
        )
        .sort(by)
    )


def _build_ticker_quarter_summary(df: pl.DataFrame, *, threshold_pct: float) -> pl.DataFrame:
    return (
        df.group_by(["ticker", "quarter_label", "date"])
        .agg(
            [
                pl.len().alias("kpi_rows"),
                _count_expr(pl.col("match_status") == "matched").alias("matched_kpis"),
                _count_expr((pl.col("match_status") == "matched") & (pl.col("diff_pct").abs() > threshold_pct)).alias(
                    "error_kpis"
                ),
                _count_expr(pl.col("match_status") == "missing_in_open_source").alias("missing_open_kpis"),
                _count_expr(pl.col("match_status") == "extra_in_open_source").alias("extra_open_kpis"),
                pl.col("abs_diff_pct").max().alias("worst_abs_diff_pct"),
                pl.col("statement").n_unique().alias("statement_count"),
                pl.col("metric").n_unique().alias("metric_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("matched_kpis") + pl.col("missing_open_kpis")).alias("reference_kpis"),
            ]
        )
        .with_columns(
            [
                _safe_pct_expr("matched_kpis", "error_kpis").alias("error_rate_pct"),
                _safe_pct_expr("reference_kpis", "missing_open_kpis").alias("missing_open_rate_pct"),
                (
                    (pl.col("error_kpis") > 0)
                    | (pl.col("missing_open_kpis") > 0)
                    | (pl.col("extra_open_kpis") > 0)
                ).alias("has_issue"),
            ]
        )
        .sort(["has_issue", "error_rate_pct", "missing_open_rate_pct", "ticker", "date"], descending=[True, True, True, False, False])
    )


def _build_summary_payload(
    *,
    alignment: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    ticker_summary: pl.DataFrame,
    ticker_quarter_summary: pl.DataFrame,
    threshold_pct: float,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    global_row = alignment.select(
        [
            pl.len().alias("total_rows"),
            _count_expr(pl.col("match_status") == "matched").alias("matched_rows"),
            _count_expr((pl.col("match_status") == "matched") & (pl.col("diff_pct").abs() > threshold_pct)).alias(
                "error_rows"
            ),
            _count_expr(pl.col("match_status") == "missing_in_open_source").alias("missing_open_rows"),
            _count_expr(pl.col("match_status") == "extra_in_open_source").alias("extra_open_rows"),
        ]
    ).to_dicts()[0]
    reference_rows = int(global_row["matched_rows"]) + int(global_row["missing_open_rows"])
    total_reference_ticker_quarters = int(
        ticker_quarter_summary.filter(pl.col("reference_kpis") > 0).select(pl.len()).item()
        if not ticker_quarter_summary.is_empty()
        else 0
    )
    error_ticker_quarters = int(
        ticker_quarter_summary.filter(pl.col("error_kpis") > 0).select(pl.len()).item()
        if not ticker_quarter_summary.is_empty()
        else 0
    )
    missing_open_ticker_quarters = int(
        ticker_quarter_summary.filter(pl.col("missing_open_kpis") > 0).select(pl.len()).item()
        if not ticker_quarter_summary.is_empty()
        else 0
    )
    top_metrics = metric_summary.sort(
        ["missing_open_rate_pct", "error_rate_pct", "reference_rows", "metric"],
        descending=[True, True, True, False],
    ).head(15)
    top_tickers = ticker_summary.sort(
        ["missing_open_rate_pct", "error_rate_pct", "reference_rows", "ticker"],
        descending=[True, True, True, False],
    ).head(15)
    top_ticker_quarters = ticker_quarter_summary.sort(
        ["missing_open_rate_pct", "error_rate_pct", "worst_abs_diff_pct", "ticker", "date"],
        descending=[True, True, True, False, False],
    ).head(20)
    return {
        "scope": {
            "start_date": start_date,
            "end_date": end_date,
            "threshold_pct": threshold_pct,
            "statements": list(AUDITED_FINANCIAL_STATEMENTS),
        },
        "global": {
            "total_rows": int(global_row["total_rows"]),
            "matched_rows": int(global_row["matched_rows"]),
            "error_rows": int(global_row["error_rows"]),
            "missing_open_rows": int(global_row["missing_open_rows"]),
            "extra_open_rows": int(global_row["extra_open_rows"]),
            "reference_rows": reference_rows,
            "error_rate_pct": _safe_pct(int(global_row["matched_rows"]), int(global_row["error_rows"])),
            "missing_open_rate_pct": _safe_pct(reference_rows, int(global_row["missing_open_rows"])),
            "extra_open_rate_pct": _safe_pct(int(global_row["total_rows"]), int(global_row["extra_open_rows"])),
            "reference_ticker_quarters": total_reference_ticker_quarters,
            "error_ticker_quarters": error_ticker_quarters,
            "missing_open_ticker_quarters": missing_open_ticker_quarters,
            "issue_ticker_quarters": int(
                ticker_quarter_summary.filter(pl.col("has_issue")).select(pl.len()).item()
                if not ticker_quarter_summary.is_empty()
                else 0
            ),
        },
        "top_metrics": top_metrics.to_dicts(),
        "top_tickers": top_tickers.to_dicts(),
        "top_ticker_quarters": top_ticker_quarters.to_dicts(),
    }


def _render_summary_markdown(summary: dict[str, object]) -> str:
    scope = summary["scope"]
    global_row = summary["global"]
    top_metrics = summary["top_metrics"]
    top_tickers = summary["top_tickers"]
    lines = [
        "# Financial Statement Audit Since 2020",
        "",
        f"- Scope: `{scope['start_date']}` -> `{scope['end_date']}`",
        f"- Threshold: `{scope['threshold_pct']:.2f}%`",
        f"- Audited statements: `{', '.join(scope['statements'])}`",
        "",
        "## Headline",
        "",
        f"- KPI rows compared: `{global_row['reference_rows']}` reference rows, `{global_row['matched_rows']}` matched rows",
        f"- Rows above threshold: `{global_row['error_rows']}` (`{global_row['error_rate_pct']:.2f}%` of matched rows)",
        f"- Missing in open-source: `{global_row['missing_open_rows']}` (`{global_row['missing_open_rate_pct']:.2f}%` of reference rows)",
        f"- Extra in open-source: `{global_row['extra_open_rows']}` (`{global_row['extra_open_rate_pct']:.2f}%` of all rows)",
        f"- Ticker-quarter combos with any > threshold KPI: `{global_row['error_ticker_quarters']}` / `{global_row['reference_ticker_quarters']}`",
        f"- Ticker-quarter combos with at least one missing KPI in open-source: `{global_row['missing_open_ticker_quarters']}` / `{global_row['reference_ticker_quarters']}`",
        "",
        "## Worst KPIs",
        "",
        "| Statement | KPI | Error rows | Error rate % | Missing rows | Missing rate % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in top_metrics[:10]:
        lines.append(
            f"| {row['statement']} | {row['metric']} | {int(row['error_rows'])} | {float(row['error_rate_pct']):.2f} | {int(row['missing_open_rows'])} | {float(row['missing_open_rate_pct']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Worst Tickers",
            "",
            "| Ticker | Error rows | Error rate % | Missing rows | Missing rate % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_tickers[:10]:
        lines.append(
            f"| {row['ticker']} | {int(row['error_rows'])} | {float(row['error_rate_pct']):.2f} | {int(row['missing_open_rows'])} | {float(row['missing_open_rate_pct']):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_dashboard_html(
    *,
    summary: dict[str, object],
    alignment: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    ticker_summary: pl.DataFrame,
    quarter_summary: pl.DataFrame,
    ticker_quarter_summary: pl.DataFrame,
    threshold_pct: float,
    start_date: str,
    end_date: str,
) -> str:
    threshold_label = f"{threshold_pct:.2f}"
    detail_columns = [
        "ticker",
        "statement",
        "metric",
        "date",
        "quarter_label",
        "match_status",
        "issue_kind",
        "diff_pct",
        "abs_diff_pct",
        "eodhd_value",
        "open_value",
        "eodhd_filing_date",
        "open_filing_date",
        "date_diff_days",
        "selected_source",
        "selected_source_label",
        "candidate_sources",
    ]
    detail_rows = alignment.select(detail_columns).to_dicts()
    histogram_labels, histogram_counts = _build_histogram_counts(alignment)
    filters = {
        "statements": sorted(alignment.select("statement").unique().to_series().to_list()),
        "metrics": sorted(alignment.select("metric").unique().to_series().to_list()),
        "tickers": sorted(alignment.select("ticker").unique().to_series().to_list()),
        "quarters": sorted(alignment.select("quarter_label").drop_nulls().unique().to_series().to_list()),
    }
    statement_rows = statement_summary.to_dicts()
    metric_rows = metric_summary.to_dicts()
    ticker_rows = ticker_summary.sort(
        ["missing_open_rate_pct", "error_rate_pct", "reference_rows", "ticker"],
        descending=[True, True, True, False],
    ).head(200).to_dicts()
    quarter_rows = quarter_summary.sort("quarter_label").to_dicts()
    ticker_quarter_rows = ticker_quarter_summary.head(400).to_dicts()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Financial statement audit since 2020</title>
  <style>
    :root {{
      --bg: #f3f7fb;
      --card: #ffffff;
      --line: #d8e2ee;
      --text: #17212b;
      --muted: #5a6b7c;
      --accent: #005bbb;
      --accent-soft: #dbeeff;
      --danger: #c0392b;
      --warn: #d97706;
      --good: #127a52;
    }}
    body {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: var(--bg); color: var(--text); }}
    .page {{ max-width: 1500px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    p {{ margin: 0; }}
    .hero {{ background: linear-gradient(135deg, #eaf4ff 0%, #ffffff 100%); border: 1px solid var(--line); border-radius: 18px; padding: 22px; box-shadow: 0 10px 30px rgba(18, 35, 58, 0.06); }}
    .hero .meta {{ color: var(--muted); margin-top: 8px; }}
    .grid {{ display: grid; gap: 14px; }}
    .cards {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 18px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .card .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .controls {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 22px; }}
    .control {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
    .control label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .05em; }}
    .control select, .control input {{ width: 100%; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; font-size: 14px; background: white; }}
    .section {{ margin-top: 22px; }}
    .two-col {{ display:grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
    .chart {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
    .bar-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(44px, 1fr)); align-items: end; gap: 8px; min-height: 220px; margin-top: 12px; }}
    .bar-wrap {{ display:flex; flex-direction:column; gap:6px; align-items:center; }}
    .bar {{ width: 100%; max-width: 40px; border-radius: 8px 8px 0 0; background: linear-gradient(180deg, #62a7ff 0%, var(--accent) 100%); min-height: 2px; }}
    .bar-label {{ font-size: 11px; color: var(--muted); text-align: center; }}
    .bar-value {{ font-size: 11px; font-weight: 600; color: var(--text); }}
    .table-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #ecf1f6; padding: 8px 10px; text-align: left; font-size: 12px; vertical-align: top; }}
    th {{ color: var(--muted); background: #f9fbfd; position: sticky; top: 0; }}
    .scroll {{ max-height: 480px; overflow: auto; border: 1px solid #edf2f7; border-radius: 10px; }}
    .pill {{ display:inline-block; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
    .pill.error_gt_threshold {{ background:#fff1ef; color: var(--danger); }}
    .pill.missing_in_open_source {{ background:#fff8e8; color: var(--warn); }}
    .pill.extra_in_open_source {{ background:#eef7ff; color: var(--accent); }}
    .pill.within_threshold {{ background:#edf9f2; color: var(--good); }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    @media (max-width: 1100px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Financial Statement Audit Since 2020</h1>
      <p class="meta">Scope: <strong>{escape(start_date)}</strong> to <strong>{escape(end_date)}</strong> | Threshold: <strong>{threshold_pct:.2f}%</strong> | KPI set: <strong>income, balance, cash flow, shares</strong></p>
    </div>

    <div id="cards" class="grid cards"></div>

    <div class="grid controls">
      <div class="control">
        <label for="statementFilter">Statement</label>
        <select id="statementFilter"></select>
      </div>
      <div class="control">
        <label for="metricFilter">KPI</label>
        <select id="metricFilter"></select>
      </div>
      <div class="control">
        <label for="tickerFilter">Ticker</label>
        <input id="tickerFilter" list="tickerOptions" placeholder="AAPL.US or substring">
        <datalist id="tickerOptions"></datalist>
      </div>
      <div class="control">
        <label for="quarterFilter">Quarter</label>
        <select id="quarterFilter"></select>
      </div>
      <div class="control">
        <label for="issueFilter">Issue Type</label>
        <select id="issueFilter">
          <option value="all">All rows</option>
          <option value="any_issue">Any issue</option>
          <option value="error_gt_threshold">Error &gt; threshold</option>
          <option value="missing_in_open_source">Missing in open-source</option>
          <option value="extra_in_open_source">Extra in open-source</option>
          <option value="within_threshold">Within threshold</option>
        </select>
      </div>
    </div>

    <div class="section two-col">
      <div class="chart">
        <h2>Abs Diff % Histogram</h2>
        <p class="muted small">Matched KPI rows only. Filters above apply live.</p>
        <div id="histogram" class="bar-grid"></div>
      </div>
      <div class="chart">
        <h2>Issue Mix</h2>
        <p class="muted small">Rows in current filtered slice.</p>
        <div id="issueMix" class="bar-grid"></div>
      </div>
    </div>

    <div class="section two-col">
      <div class="table-card">
        <h2>Worst KPI Summary</h2>
        <div class="scroll"><table id="metricTable"></table></div>
      </div>
      <div class="table-card">
        <h2>Worst Ticker Summary</h2>
        <div class="scroll"><table id="tickerTable"></table></div>
      </div>
    </div>

    <div class="section two-col">
      <div class="table-card">
        <h2>Ticker x Quarter</h2>
        <p class="muted small">How often a ticker-quarter has at least one KPI in error or missing.</p>
        <div class="scroll"><table id="tickerQuarterTable"></table></div>
      </div>
      <div class="table-card">
        <h2>Quarter Summary</h2>
        <div class="scroll"><table id="quarterTable"></table></div>
      </div>
    </div>

    <div class="section table-card">
      <h2>Detailed Rows</h2>
      <p class="muted small">Showing up to 250 rows after filtering. This is the one to use for ticker / quarter / KPI deep dives.</p>
      <div class="scroll"><table id="detailTable"></table></div>
    </div>
  </div>

  <script>
    const SUMMARY = {json.dumps(summary)};
    const DETAIL_ROWS = {json.dumps(detail_rows)};
    const STATEMENT_ROWS = {json.dumps(statement_rows)};
    const METRIC_ROWS = {json.dumps(metric_rows)};
    const TICKER_ROWS = {json.dumps(ticker_rows)};
    const QUARTER_ROWS = {json.dumps(quarter_rows)};
    const TICKER_QUARTER_ROWS = {json.dumps(ticker_quarter_rows)};
    const FILTERS = {json.dumps(filters)};
    const HISTOGRAM_LABELS = {json.dumps(histogram_labels)};
    const HISTOGRAM_COUNTS = {json.dumps(histogram_counts)};

    function qs(id) {{
      return document.getElementById(id);
    }}

    function setOptions(select, values, allLabel) {{
      select.innerHTML = '';
      const allOption = document.createElement('option');
      allOption.value = 'all';
      allOption.textContent = allLabel;
      select.appendChild(allOption);
      values.forEach((value) => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }});
    }}

    function fmtNumber(value, digits = 0) {{
      if (value === null || value === undefined || Number.isNaN(value)) return '';
      return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: digits, minimumFractionDigits: digits }});
    }}

    function fmtPct(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(value)) return '';
      return `${{fmtNumber(value, digits)}}%`;
    }}

    function pill(value) {{
      return `<span class="pill ${{value}}">${{value}}</span>`;
    }}

    function getFilteredRows() {{
      const statement = qs('statementFilter').value;
      const metric = qs('metricFilter').value;
      const quarter = qs('quarterFilter').value;
      const issue = qs('issueFilter').value;
      const ticker = qs('tickerFilter').value.trim().toUpperCase();
      return DETAIL_ROWS.filter((row) => {{
        if (statement !== 'all' && row.statement !== statement) return false;
        if (metric !== 'all' && row.metric !== metric) return false;
        if (quarter !== 'all' && row.quarter_label !== quarter) return false;
        if (ticker && !row.ticker.toUpperCase().includes(ticker)) return false;
        if (issue === 'all') return true;
        if (issue === 'any_issue') return row.issue_kind !== 'within_threshold';
        return row.issue_kind === issue;
      }});
    }}

    function computeCards(rows) {{
      const matched = rows.filter((row) => row.match_status === 'matched');
      const errors = rows.filter((row) => row.issue_kind === 'error_gt_threshold');
      const missingOpen = rows.filter((row) => row.issue_kind === 'missing_in_open_source');
      const extraOpen = rows.filter((row) => row.issue_kind === 'extra_in_open_source');
      const referenceRows = rows.filter((row) => row.match_status !== 'extra_in_open_source').length;
      const tqMap = new Map();
      rows.forEach((row) => {{
        const key = `${{row.ticker}}|${{row.date}}`;
        const item = tqMap.get(key) || {{ reference: false, error: false, missingOpen: false }};
        if (row.match_status !== 'extra_in_open_source') item.reference = true;
        if (row.issue_kind === 'error_gt_threshold') item.error = true;
        if (row.issue_kind === 'missing_in_open_source') item.missingOpen = true;
        tqMap.set(key, item);
      }});
      const referenceTickerQuarters = Array.from(tqMap.values()).filter((item) => item.reference).length;
      const errorTickerQuarters = Array.from(tqMap.values()).filter((item) => item.reference && item.error).length;
      const missingTickerQuarters = Array.from(tqMap.values()).filter((item) => item.reference && item.missingOpen).length;
      return [
        ['Reference KPI Rows', referenceRows, ''],
        ['Matched KPI Rows', matched.length, ''],
        ['Rows > Threshold', errors.length, fmtPct(matched.length ? (errors.length / matched.length) * 100 : 0)],
        ['Missing In Open', missingOpen.length, fmtPct(referenceRows ? (missingOpen.length / referenceRows) * 100 : 0)],
        ['Extra In Open', extraOpen.length, fmtPct(rows.length ? (extraOpen.length / rows.length) * 100 : 0)],
        ['Ticker-Quarter With Error', errorTickerQuarters, `${{referenceTickerQuarters ? fmtPct((errorTickerQuarters / referenceTickerQuarters) * 100) : '0.00%'}}`],
        ['Ticker-Quarter Missing', missingTickerQuarters, `${{referenceTickerQuarters ? fmtPct((missingTickerQuarters / referenceTickerQuarters) * 100) : '0.00%'}}`],
      ];
    }}

    function renderCards(rows) {{
      const cards = computeCards(rows);
      qs('cards').innerHTML = cards.map(([label, value, meta]) => `
        <div class="card">
          <div class="label">${{label}}</div>
          <div class="value">${{fmtNumber(value)}}</div>
          <div class="muted small">${{meta || '&nbsp;'}}</div>
        </div>
      `).join('');
    }}

    function renderBars(containerId, labels, counts) {{
      const max = Math.max(...counts, 1);
      const container = qs(containerId);
      container.innerHTML = labels.map((label, index) => {{
        const count = counts[index];
        const height = Math.max(2, (count / max) * 180);
        return `
          <div class="bar-wrap">
            <div class="bar-value">${{fmtNumber(count)}}</div>
            <div class="bar" style="height:${{height}}px"></div>
            <div class="bar-label">${{label}}</div>
          </div>
        `;
      }}).join('');
    }}

    function renderHistogram(rows) {{
      const bins = Array(HISTOGRAM_LABELS.length).fill(0);
      rows.forEach((row) => {{
        if (row.match_status !== 'matched' || row.abs_diff_pct === null || row.abs_diff_pct === undefined) return;
        const value = Number(row.abs_diff_pct);
        for (let i = 0; i < HISTOGRAM_LABELS.length; i += 1) {{
          const label = HISTOGRAM_LABELS[i];
          if (label.startsWith('>')) {{
            bins[i] += 1;
            break;
          }}
          const upper = parseFloat(label.split('–')[1]);
          if (value <= upper) {{
            bins[i] += 1;
            break;
          }}
        }}
      }});
      renderBars('histogram', HISTOGRAM_LABELS, bins);
    }}

    function renderIssueMix(rows) {{
      const labels = ['within_threshold', 'error_gt_threshold', 'missing_in_open_source', 'extra_in_open_source'];
      const counts = labels.map((label) => rows.filter((row) => row.issue_kind === label).length);
      renderBars('issueMix', labels, counts);
    }}

    function aggregate(rows, keys) {{
      const map = new Map();
      rows.forEach((row) => {{
        const key = keys.map((keyName) => row[keyName] ?? '').join('|');
        let item = map.get(key);
        if (!item) {{
          item = Object.fromEntries(keys.map((keyName) => [keyName, row[keyName]]));
          item.matched_rows = 0;
          item.error_rows = 0;
          item.missing_open_rows = 0;
          item.extra_open_rows = 0;
          item.reference_rows = 0;
          item.max_abs_diff_pct = 0;
          map.set(key, item);
        }}
        if (row.match_status === 'matched') {{
          item.matched_rows += 1;
          if (row.issue_kind === 'error_gt_threshold') item.error_rows += 1;
        }}
        if (row.match_status === 'missing_in_open_source') item.missing_open_rows += 1;
        if (row.match_status === 'extra_in_open_source') item.extra_open_rows += 1;
        if (row.match_status !== 'extra_in_open_source') item.reference_rows += 1;
        if (row.abs_diff_pct !== null && row.abs_diff_pct !== undefined) {{
          item.max_abs_diff_pct = Math.max(item.max_abs_diff_pct, Number(row.abs_diff_pct));
        }}
      }});
      return Array.from(map.values()).map((item) => ({{
        ...item,
        error_rate_pct: item.matched_rows ? (item.error_rows / item.matched_rows) * 100 : 0,
        missing_open_rate_pct: item.reference_rows ? (item.missing_open_rows / item.reference_rows) * 100 : 0,
      }}));
    }}

    function renderTable(tableId, rows, columns, limit = 50) {{
      const table = qs(tableId);
      const safeRows = rows.slice(0, limit);
      table.innerHTML = `
        <thead><tr>${{columns.map((column) => `<th>${{column.label}}</th>`).join('')}}</tr></thead>
        <tbody>
          ${{safeRows.map((row) => `<tr>${{columns.map((column) => `<td>${{column.render ? column.render(row[column.key], row) : (row[column.key] ?? '')}}</td>`).join('')}}</tr>`).join('')}}
        </tbody>
      `;
    }}

    function renderTables(rows) {{
      const metricRows = aggregate(rows, ['statement', 'metric'])
        .sort((a, b) => (b.missing_open_rate_pct - a.missing_open_rate_pct) || (b.error_rate_pct - a.error_rate_pct) || (b.reference_rows - a.reference_rows));
      const tickerRows = aggregate(rows, ['ticker'])
        .sort((a, b) => (b.missing_open_rate_pct - a.missing_open_rate_pct) || (b.error_rate_pct - a.error_rate_pct) || (b.reference_rows - a.reference_rows));
      const quarterRows = aggregate(rows, ['quarter_label'])
        .sort((a, b) => String(a.quarter_label).localeCompare(String(b.quarter_label)));

      const tqMap = new Map();
      rows.forEach((row) => {{
        const key = `${{row.ticker}}|${{row.date}}`;
        let item = tqMap.get(key);
        if (!item) {{
          item = {{
            ticker: row.ticker,
            quarter_label: row.quarter_label,
            date: row.date,
            matched_kpis: 0,
            error_kpis: 0,
            missing_open_kpis: 0,
            extra_open_kpis: 0,
            reference_kpis: 0,
            worst_abs_diff_pct: 0,
          }};
          tqMap.set(key, item);
        }}
        if (row.match_status === 'matched') {{
          item.matched_kpis += 1;
          if (row.issue_kind === 'error_gt_threshold') item.error_kpis += 1;
        }}
        if (row.issue_kind === 'missing_in_open_source') item.missing_open_kpis += 1;
        if (row.issue_kind === 'extra_in_open_source') item.extra_open_kpis += 1;
        if (row.match_status !== 'extra_in_open_source') item.reference_kpis += 1;
        if (row.abs_diff_pct !== null && row.abs_diff_pct !== undefined) {{
          item.worst_abs_diff_pct = Math.max(item.worst_abs_diff_pct, Number(row.abs_diff_pct));
        }}
      }});
      const tickerQuarterRows = Array.from(tqMap.values()).map((row) => ({{
        ...row,
        error_rate_pct: row.matched_kpis ? (row.error_kpis / row.matched_kpis) * 100 : 0,
        missing_open_rate_pct: row.reference_kpis ? (row.missing_open_kpis / row.reference_kpis) * 100 : 0,
      }})).sort((a, b) => (b.missing_open_rate_pct - a.missing_open_rate_pct) || (b.error_rate_pct - a.error_rate_pct) || (b.worst_abs_diff_pct - a.worst_abs_diff_pct));

      renderTable('metricTable', metricRows, [
        {{ key: 'statement', label: 'Statement' }},
        {{ key: 'metric', label: 'KPI' }},
        {{ key: 'reference_rows', label: 'Ref Rows', render: (v) => fmtNumber(v) }},
        {{ key: 'error_rows', label: `> {threshold_label}%`, render: (v) => fmtNumber(v) }},
        {{ key: 'error_rate_pct', label: 'Error %', render: (v) => fmtPct(v) }},
        {{ key: 'missing_open_rows', label: 'Missing', render: (v) => fmtNumber(v) }},
        {{ key: 'missing_open_rate_pct', label: 'Missing %', render: (v) => fmtPct(v) }},
      ], 80);

      renderTable('tickerTable', tickerRows, [
        {{ key: 'ticker', label: 'Ticker' }},
        {{ key: 'reference_rows', label: 'Ref Rows', render: (v) => fmtNumber(v) }},
        {{ key: 'error_rows', label: `> {threshold_label}%`, render: (v) => fmtNumber(v) }},
        {{ key: 'error_rate_pct', label: 'Error %', render: (v) => fmtPct(v) }},
        {{ key: 'missing_open_rows', label: 'Missing', render: (v) => fmtNumber(v) }},
        {{ key: 'missing_open_rate_pct', label: 'Missing %', render: (v) => fmtPct(v) }},
      ], 120);

      renderTable('quarterTable', quarterRows, [
        {{ key: 'quarter_label', label: 'Quarter' }},
        {{ key: 'reference_rows', label: 'Ref Rows', render: (v) => fmtNumber(v) }},
        {{ key: 'error_rows', label: `> {threshold_label}%`, render: (v) => fmtNumber(v) }},
        {{ key: 'error_rate_pct', label: 'Error %', render: (v) => fmtPct(v) }},
        {{ key: 'missing_open_rows', label: 'Missing', render: (v) => fmtNumber(v) }},
        {{ key: 'missing_open_rate_pct', label: 'Missing %', render: (v) => fmtPct(v) }},
      ], 60);

      renderTable('tickerQuarterTable', tickerQuarterRows, [
        {{ key: 'ticker', label: 'Ticker' }},
        {{ key: 'quarter_label', label: 'Quarter' }},
        {{ key: 'reference_kpis', label: 'Ref KPIs', render: (v) => fmtNumber(v) }},
        {{ key: 'error_kpis', label: `> {threshold_label}%`, render: (v) => fmtNumber(v) }},
        {{ key: 'missing_open_kpis', label: 'Missing', render: (v) => fmtNumber(v) }},
        {{ key: 'error_rate_pct', label: 'Error %', render: (v) => fmtPct(v) }},
        {{ key: 'missing_open_rate_pct', label: 'Missing %', render: (v) => fmtPct(v) }},
        {{ key: 'worst_abs_diff_pct', label: 'Worst Diff %', render: (v) => fmtPct(v) }},
      ], 150);

      renderTable('detailTable', rows, [
        {{ key: 'ticker', label: 'Ticker' }},
        {{ key: 'quarter_label', label: 'Quarter' }},
        {{ key: 'statement', label: 'Statement' }},
        {{ key: 'metric', label: 'KPI' }},
        {{ key: 'issue_kind', label: 'Issue', render: (v) => pill(v) }},
        {{ key: 'diff_pct', label: 'Diff %', render: (v) => fmtPct(v) }},
        {{ key: 'eodhd_value', label: 'EODHD', render: (v) => fmtNumber(v, 2) }},
        {{ key: 'open_value', label: 'Open', render: (v) => fmtNumber(v, 2) }},
        {{ key: 'date_diff_days', label: 'Date Δ days', render: (v) => fmtNumber(v) }},
        {{ key: 'selected_source', label: 'Selected Source' }},
        {{ key: 'selected_source_label', label: 'Source Label' }},
      ], 250);
    }}

    function refresh() {{
      const rows = getFilteredRows();
      renderCards(rows);
      renderHistogram(rows);
      renderIssueMix(rows);
      renderTables(rows);
    }}

    setOptions(qs('statementFilter'), FILTERS.statements, 'All statements');
    setOptions(qs('metricFilter'), FILTERS.metrics, 'All KPIs');
    setOptions(qs('quarterFilter'), FILTERS.quarters, 'All quarters');
    qs('tickerOptions').innerHTML = FILTERS.tickers.map((ticker) => `<option value="${{ticker}}"></option>`).join('');
    ['statementFilter', 'metricFilter', 'quarterFilter', 'issueFilter', 'tickerFilter'].forEach((id) => {{
      qs(id).addEventListener('input', refresh);
      qs(id).addEventListener('change', refresh);
    }});
    refresh();
  </script>
</body>
</html>"""


def _build_histogram_counts(df: pl.DataFrame) -> tuple[list[str], list[int]]:
    labels: list[str] = []
    counts: list[int] = []
    matched = df.filter(pl.col("match_status") == "matched")
    for lower, upper in zip(HISTOGRAM_BINS[:-1], HISTOGRAM_BINS[1:]):
        labels.append(f"{lower:g}–{upper:g}")
        counts.append(
            matched.filter((pl.col("abs_diff_pct") > lower) & (pl.col("abs_diff_pct") <= upper)).height
        )
    labels.append(f">{HISTOGRAM_BINS[-1]:g}")
    counts.append(matched.filter(pl.col("abs_diff_pct") > HISTOGRAM_BINS[-1]).height)
    return labels, counts


def _issue_kind_expr(threshold_pct: float) -> pl.Expr:
    return (
        pl.when(pl.col("match_status") == "missing_in_open_source")
        .then(pl.lit("missing_in_open_source"))
        .when(pl.col("match_status") == "extra_in_open_source")
        .then(pl.lit("extra_in_open_source"))
        .when(pl.col("diff_pct").abs() > threshold_pct)
        .then(pl.lit("error_gt_threshold"))
        .otherwise(pl.lit("within_threshold"))
    )


def _quarter_label_expr(column: str) -> pl.Expr:
    date_expr = _parsed_date_expr(column)
    return (
        pl.when(date_expr.is_not_null())
        .then(
            pl.format(
                "{}-Q{}",
                date_expr.dt.year(),
                (((date_expr.dt.month() - 1) // 3) + 1),
            )
        )
        .otherwise(pl.lit(None).cast(pl.Utf8))
    )


def _abs_diff_expr() -> pl.Expr:
    return pl.when(pl.col("diff_pct").is_not_null()).then(pl.col("diff_pct").abs()).otherwise(None)


def _safe_pct(denominator: int, numerator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _safe_pct_expr(denominator_col: str, numerator_col: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator_col) > 0)
        .then((pl.col(numerator_col) / pl.col(denominator_col)) * 100.0)
        .otherwise(0.0)
    )


def _count_expr(predicate: pl.Expr) -> pl.Expr:
    return pl.when(predicate).then(pl.lit(1)).otherwise(pl.lit(0)).sum()


def _parsed_date_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        pl.col(column).cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col(column).cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-%m", strict=False),
        pl.col(column).cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-Q%q", strict=False),
    )


def _alignment_schema() -> dict[str, pl.DataType]:
    return {
        "ticker": pl.String,
        "statement": pl.String,
        "metric": pl.String,
        "eodhd_date": pl.String,
        "open_date": pl.String,
        "eodhd_filing_date": pl.String,
        "open_filing_date": pl.String,
        "eodhd_value": pl.Float64,
        "open_value": pl.Float64,
        "eodhd_source_label": pl.String,
        "open_source_label": pl.String,
        "selected_source": pl.String,
        "selected_source_label": pl.String,
        "candidate_sources": pl.String,
        "selected_fiscal_period": pl.String,
        "selected_fiscal_year": pl.Int64,
        "date_diff_days": pl.Int64,
    }


def _empty_financial_frame() -> pl.DataFrame:
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
            "candidate_sources": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
        }
    )


def _empty_alignment_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "eodhd_date": pl.String,
            "open_date": pl.String,
            "eodhd_filing_date": pl.String,
            "open_filing_date": pl.String,
            "eodhd_value": pl.Float64,
            "open_value": pl.Float64,
            "eodhd_source_label": pl.String,
            "open_source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "candidate_sources": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "date_diff_days": pl.Int64,
            "source": pl.String,
            "match_status": pl.String,
            "value_diff": pl.Float64,
            "diff_pct": pl.Float64,
        }
    )
