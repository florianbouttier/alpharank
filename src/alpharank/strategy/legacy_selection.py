"""Point-in-time sector and monthly holding selection for Legacy."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from alpharank.data.contracts.sector_history import (
    SECTOR_HISTORY_LINEAGE_COLUMNS,
    resolve_point_in_time_sectors,
)
from alpharank.data.terminal_eligibility import (
    TERMINAL_ENTRY_POLICY_ID,
    apply_terminal_entry_gate,
)
from alpharank.portfolio.terminal_event_registry import load_terminal_event_registry
from alpharank.utils.frame_backend import to_polars

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


def attach_legacy_sector_policy(candidates: "pl.DataFrame", sector: pd.DataFrame) -> "pl.DataFrame":
    """Attach only sectors that were known before the Legacy order boundary."""

    decision_candidates = candidates.with_columns(
        pl.col("year_month").cast(pl.Datetime).dt.replace_time_zone("UTC").alias("decision_at")
    )
    terminal_registry = load_terminal_event_registry()
    terminal_gate = apply_terminal_entry_gate(
        decision_candidates,
        terminal_registry.terminal_entry_blocks(),
    )
    blocked_by_month = (
        terminal_gate.blocked.group_by("year_month").agg(
            pl.len().alias("terminal_entry_blocked_candidates")
        )
        if terminal_gate.blocked.height
        else pl.DataFrame(
            schema={
                "year_month": decision_candidates.schema["year_month"],
                "terminal_entry_blocked_candidates": pl.UInt32,
            }
        )
    )
    decision_candidates = terminal_gate.eligible.join(
        blocked_by_month,
        on="year_month",
        how="left",
    ).with_columns(
        pl.col("terminal_entry_blocked_candidates").fill_null(0).cast(pl.UInt32),
        pl.lit(TERMINAL_ENTRY_POLICY_ID).alias("terminal_entry_policy_id"),
        pl.lit(terminal_registry.payload["registry_id"]).alias("terminal_entry_registry_id"),
        pl.lit(terminal_registry.sha256).alias("terminal_entry_registry_sha256"),
    )
    required_history = {
        "ticker",
        "Sector",
        *SECTOR_HISTORY_LINEAGE_COLUMNS,
    }
    if required_history.issubset(sector.columns):
        decisions = decision_candidates.select("ticker", "decision_at").unique()
        resolved = resolve_point_in_time_sectors(
            decisions,
            to_polars(sector),
        ).select(
            "ticker",
            "decision_at",
            "Sector",
            "sector_constraint_enabled",
            "sector_constraint_reason",
            "sector_known_at_selected",
            "classification_id",
        )
        return decision_candidates.join(
            resolved,
            on=["ticker", "decision_at"],
            how="left",
        )

    return decision_candidates.with_columns(
        pl.lit(None).cast(pl.String).alias("Sector"),
        pl.lit(False).alias("sector_constraint_enabled"),
        pl.lit("disabled_no_point_in_time_sector_history").alias("sector_constraint_reason"),
        pl.lit(None).cast(pl.Datetime(time_zone="UTC")).alias("sector_known_at_selected"),
        pl.lit(None).cast(pl.String).alias("classification_id"),
    )


def get_portfolio_at_month(
    portfolio_output: Dict[str, Any],
    month: Optional[pd.Period] = None,
) -> pd.DataFrame:
    """Return normalized Legacy holdings for one month."""

    if "detailed" in portfolio_output:
        df = portfolio_output["detailed"].copy()
    elif "detailled" in portfolio_output:
        df = portfolio_output["detailled"].copy()
    else:
        raise ValueError("portfolio_output must contain 'detailed' or 'detailled' key")

    if df.empty:
        raise ValueError("Portfolio is empty")
    if month is None:
        month = df["year_month"].max()

    df_month = df[df["year_month"] == month].copy()
    if df_month.empty:
        available = df["year_month"].unique()[:5]
        raise ValueError(f"No data for month {month}. Available months: {list(available)}...")

    if "weight" not in df_month.columns:
        df_month["weight"] = 1.0
    total_weight = df_month["weight"].sum()
    if total_weight > 0:
        df_month["weight_normalized"] = df_month["weight"] / total_weight
    else:
        df_month["weight_normalized"] = 1.0 / len(df_month)

    if "dr" in df_month.columns:
        return_col = "dr"
    elif "monthly_return" in df_month.columns:
        return_col = "monthly_return"
    else:
        return_col = None

    output_cols = ["ticker"]
    if "Sector" in df_month.columns:
        output_cols.append("Sector")
    output_cols.extend(["weight", "weight_normalized"])
    if return_col:
        df_month["monthly_return"] = df_month[return_col]
        output_cols.append("monthly_return")
    if "n_models" in df_month.columns:
        output_cols.append("n_models")

    result = df_month[output_cols].copy()
    result = result.sort_values("weight_normalized", ascending=False).reset_index(drop=True)
    result.attrs["month"] = month
    result.attrs["total_stocks"] = len(result)
    return result
