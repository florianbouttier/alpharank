"""Point-in-time Legacy valuation eligibility for controlled comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import polars as pl

from alpharank.data.processing import FundamentalProcessor, PricesDataPreprocessor
from alpharank.utils.frame_backend import to_pandas, to_polars

LEGACY_MINIMUM_PE = 0.0
LEGACY_MAXIMUM_PE = 100.0
LEGACY_PE_MARKET_CAP_POLICY_ID = "legacy_pe_market_cap_v1"
NO_SEC_FUNDAMENTALS_POLICY_ID = "no_sec_fundamentals_v1"
SUPPORTED_FUNDAMENTAL_ELIGIBILITY_POLICY_IDS = frozenset(
    {LEGACY_PE_MARKET_CAP_POLICY_ID, NO_SEC_FUNDAMENTALS_POLICY_ID}
)


@dataclass(frozen=True, slots=True)
class LegacyValuationInputs:
    """Immutable snapshot files used by Legacy's valuation gate."""

    final_price: Path
    balance: Path
    cash_flow: Path
    earnings: Path
    income: Path


def build_legacy_selection_universe(
    *,
    policy_id: str,
    monthly_return: pl.DataFrame,
    historical_membership: pl.DataFrame,
    balance: pl.DataFrame | None = None,
    cash_flow: pl.DataFrame | None = None,
    earnings: pl.DataFrame | None = None,
    income: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build Legacy candidate keys under one explicit fundamental policy.

    ``no_sec_fundamentals_v1`` uses only observed monthly prices and historical
    index membership. It neither reads nor computes any fundamental value.
    """

    _require_supported_policy(policy_id)
    membership = _normalized_membership(historical_membership)
    if policy_id == NO_SEC_FUNDAMENTALS_POLICY_ID:
        return (
            monthly_return.select(
                pl.col("ticker").cast(pl.String),
                pl.col("year_month").cast(pl.Date, strict=False),
            )
            .unique()
            .join(membership, on=["ticker", "year_month"], how="inner")
            .sort(["year_month", "ticker"])
        )

    fundamental_frames = {
        "balance": balance,
        "cash_flow": cash_flow,
        "earnings": earnings,
        "income": income,
    }
    missing = sorted(name for name, frame in fundamental_frames.items() if frame is None)
    if missing:
        raise ValueError("Legacy PE policy requires fundamental frames: " + ", ".join(missing))
    valuation = to_polars(
        FundamentalProcessor.calculate_pe_ratios(
            balance=balance,
            earnings=earnings,
            cashflow=cash_flow,
            income=income,
            earning_choice="netincome_rolling",
            monthly_return=to_pandas(monthly_return),
            list_date_to_maximise=["filing_date_income", "filing_date_balance"],
            backend="polars",
        )
    )
    _ = FundamentalProcessor.calculate_all_ratios(
        balance_sheet=balance,
        income_statement=income,
        cash_flow=cash_flow,
        earnings=earnings,
        monthly_return=to_pandas(monthly_return),
        backend="polars",
    )
    return (
        valuation.with_columns(pl.col("year_month").cast(pl.Date, strict=False))
        .filter(
            (pl.col("pe") < LEGACY_MAXIMUM_PE)
            & (pl.col("pe") > LEGACY_MINIMUM_PE)
            & pl.col("pe").is_not_null()
            & pl.col("market_cap").is_not_null()
        )
        .join(membership, on=["ticker", "year_month"], how="inner")
    )


def build_legacy_valuation_registry(
    *,
    snapshot_dir: Path,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Classify candidate ticker-months with Legacy's filing-aware PE gate."""

    _require_candidate_columns(candidates)
    inputs = _resolve_inputs(snapshot_dir)
    final_price = pl.read_parquet(inputs.final_price)
    monthly_return = to_polars(
        PricesDataPreprocessor.calculate_monthly_returns(
            df=final_price,
            column_close="adjusted_close",
            column_date="date",
            backend="polars",
        )
    )
    source_frames = _read_fundamental_inputs(inputs)
    valuation = _calculate_legacy_valuation(source_frames, monthly_return)
    sec_tickers = {
        str(ticker)
        for frame in source_frames.values()
        if "ticker" in frame.columns
        for ticker in frame["ticker"].drop_nulls().to_list()
    }
    return classify_legacy_valuation_eligibility(
        candidates=candidates,
        valuation=valuation,
        sec_tickers=sec_tickers,
    )


def classify_legacy_valuation_eligibility(
    *,
    candidates: pl.DataFrame,
    valuation: pl.DataFrame,
    sec_tickers: set[str],
) -> pl.DataFrame:
    """Attach one mutually exclusive Legacy valuation reason per candidate."""

    _require_candidate_columns(candidates)
    required_valuation = {"ticker", "decision_month", "pe", "market_cap"}
    missing = sorted(required_valuation - set(valuation.columns))
    if missing:
        raise ValueError(f"Valuation is missing columns: {', '.join(missing)}")
    _require_unique_ticker_months(valuation, label="Legacy valuation")
    candidate_keys = candidates.select(
        pl.col("ticker").cast(pl.String),
        pl.col("decision_month").cast(pl.Date),
    ).unique()
    return (
        candidate_keys.join(
            valuation.select("ticker", "decision_month", "pe", "market_cap"),
            on=["ticker", "decision_month"],
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("ticker").is_in(sorted(sec_tickers)).alias("has_sec_source_rows"))
        .with_columns(_eligibility_reason_expression())
        .with_columns(
            (pl.col("eligibility_reason") == "eligible").alias("legacy_valuation_eligible")
        )
        .sort(["decision_month", "ticker"])
    )


def filter_predictions_to_legacy_valuation_universe(
    predictions: pl.DataFrame,
    registry: pl.DataFrame,
) -> pl.DataFrame:
    """Apply Legacy's point-in-time valuation gate before Boosting ranking."""

    _require_candidate_columns(predictions)
    _require_unique_ticker_months(registry, label="Legacy valuation registry")
    eligible = registry.filter(pl.col("legacy_valuation_eligible")).select(
        "ticker", "decision_month"
    )
    return predictions.join(
        eligible,
        on=["ticker", "decision_month"],
        how="inner",
        validate="m:1",
    )


def _resolve_inputs(snapshot_dir: Path) -> LegacyValuationInputs:
    inputs = LegacyValuationInputs(
        final_price=snapshot_dir / "US_Finalprice.parquet",
        balance=snapshot_dir / "US_Balance_sheet.parquet",
        cash_flow=snapshot_dir / "US_Cash_flow.parquet",
        earnings=snapshot_dir / "US_Earnings.parquet",
        income=snapshot_dir / "US_Income_statement.parquet",
    )
    paths = (
        inputs.final_price,
        inputs.balance,
        inputs.cash_flow,
        inputs.earnings,
        inputs.income,
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing snapshot inputs:\n" + "\n".join(map(str, missing)))
    return inputs


def _normalized_membership(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"ticker", "year_month"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Historical membership is missing columns: {', '.join(missing)}")
    return frame.select(
        pl.col("ticker").cast(pl.String),
        pl.col("year_month").cast(pl.Date, strict=False),
    ).unique()


def _require_supported_policy(policy_id: str) -> None:
    if policy_id not in SUPPORTED_FUNDAMENTAL_ELIGIBILITY_POLICY_IDS:
        supported = ", ".join(sorted(SUPPORTED_FUNDAMENTAL_ELIGIBILITY_POLICY_IDS))
        raise ValueError(
            f"Unsupported Legacy fundamental eligibility policy {policy_id!r}; "
            f"expected one of: {supported}."
        )


def _read_fundamental_inputs(inputs: LegacyValuationInputs) -> dict[str, pl.DataFrame]:
    return {
        "balance": pl.read_parquet(inputs.balance),
        "cash_flow": pl.read_parquet(inputs.cash_flow),
        "earnings": pl.read_parquet(inputs.earnings),
        "income": pl.read_parquet(inputs.income),
    }


def _calculate_legacy_valuation(
    source_frames: Mapping[str, pl.DataFrame],
    monthly_return: pl.DataFrame,
) -> pl.DataFrame:
    valuation = to_polars(
        FundamentalProcessor.calculate_pe_ratios(
            balance=to_pandas(source_frames["balance"]),
            earnings=to_pandas(source_frames["earnings"]),
            cashflow=to_pandas(source_frames["cash_flow"]),
            income=to_pandas(source_frames["income"]),
            earning_choice="netincome_rolling",
            monthly_return=to_pandas(monthly_return),
            list_date_to_maximise=["filing_date_income", "filing_date_balance"],
            backend="polars",
        )
    )
    return valuation.with_columns(
        pl.col("year_month")
        .cast(pl.String)
        .str.slice(0, 7)
        .str.to_date("%Y-%m")
        .alias("decision_month")
    )


def _eligibility_reason_expression() -> pl.Expr:
    return (
        pl.when(~pl.col("has_sec_source_rows"))
        .then(pl.lit("no_sec_source_rows"))
        .when(pl.col("pe").is_null())
        .then(pl.lit("missing_point_in_time_pe"))
        .when(pl.col("market_cap").is_null())
        .then(pl.lit("missing_market_cap"))
        .when(pl.col("pe") <= LEGACY_MINIMUM_PE)
        .then(pl.lit("pe_nonpositive"))
        .when(pl.col("pe") >= LEGACY_MAXIMUM_PE)
        .then(pl.lit("pe_at_least_100"))
        .otherwise(pl.lit("eligible"))
        .alias("eligibility_reason")
    )


def _require_candidate_columns(frame: pl.DataFrame) -> None:
    required = {"ticker", "decision_month"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidates are missing columns: {', '.join(missing)}")


def _require_unique_ticker_months(frame: pl.DataFrame, *, label: str) -> None:
    required = {"ticker", "decision_month"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
    duplicate_count = (
        frame.group_by("ticker", "decision_month").len().filter(pl.col("len") > 1).height
    )
    if duplicate_count:
        raise ValueError(f"{label} has {duplicate_count} duplicate ticker-month keys.")
