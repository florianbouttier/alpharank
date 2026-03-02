"""Backend and parity helpers for pandas/polars interoperability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None

Backend = Literal["pandas", "polars"]
FrameLike = Union[pd.DataFrame, "pl.DataFrame"]  # type: ignore[name-defined]


@dataclass
class FrameComparison:
    ok: bool
    message: str
    mismatched_columns: Optional[List[str]] = None


def is_polars_available() -> bool:
    return pl is not None


def require_polars() -> None:
    if pl is None:
        raise ImportError(
            "Polars backend requested but polars is not installed. "
            "Install dependencies with `pip install polars pyarrow`."
        )


def ensure_backend_name(backend: Optional[str], default: Backend = "polars") -> Backend:
    if backend is None:
        return default
    backend = backend.lower()
    if backend not in {"pandas", "polars"}:
        raise ValueError(f"Unknown backend '{backend}'. Expected 'pandas' or 'polars'.")
    return backend  # type: ignore[return-value]


def to_pandas(df: FrameLike) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def to_polars(df: FrameLike) -> "pl.DataFrame":  # type: ignore[name-defined]
    require_polars()
    if isinstance(df, pd.DataFrame):
        pdf = df.copy()
        # Polars does not natively preserve pandas Period extension dtype.
        # Convert Period columns to month-start timestamps before crossing the boundary.
        period_cols = [
            c for c in pdf.columns
            if isinstance(getattr(pdf[c], "dtype", None), pd.PeriodDtype)
        ]
        for col in period_cols:
            pdf[col] = pdf[col].dt.to_timestamp(how="start")
        return pl.from_pandas(pdf)  # type: ignore[union-attr]
    if isinstance(df, pl.DataFrame):  # type: ignore[union-attr]
        return df.clone()
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def normalize_year_month_to_timestamp(df: pd.DataFrame, col: str = "year_month") -> pd.DataFrame:
    """Normalize a monthly period-like column to month-start timestamps."""
    if col not in df.columns:
        return df
    out = df.copy()
    ser = out[col]
    if isinstance(getattr(ser, "dtype", None), pd.PeriodDtype):
        out[col] = ser.dt.to_timestamp(how="start")
        return out
    if pd.api.types.is_datetime64_any_dtype(ser):
        out[col] = pd.to_datetime(ser).dt.to_period("M").dt.to_timestamp(how="start")
        return out
    out[col] = pd.to_datetime(ser, errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    return out


def normalize_year_month_to_period(df: pd.DataFrame, col: str = "year_month") -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    ser = out[col]
    if isinstance(getattr(ser, "dtype", None), pd.PeriodDtype):
        return out
    if pd.api.types.is_datetime64_any_dtype(ser):
        out[col] = pd.to_datetime(ser, errors="coerce").dt.to_period("M")
        return out
    if pd.api.types.is_numeric_dtype(ser):
        numeric = pd.to_numeric(ser, errors="coerce")
        valid = numeric.dropna()
        if len(valid) > 0:
            # Handle common YYYYMM integer encoding.
            if valid.between(100001, 999912).all():
                years = (numeric // 100).astype("Int64")
                months = (numeric % 100).astype("Int64")
                ts = pd.to_datetime(
                    {
                        "year": years,
                        "month": months,
                        "day": pd.Series(1, index=out.index, dtype="Int64"),
                    },
                    errors="coerce",
                )
                out[col] = ts.dt.to_period("M")
                return out
            # Handle pandas Period ordinal storage (e.g. 588 -> 2019-01 for freq='M').
            ordinals = numeric.round()
            if np.allclose(valid.to_numpy(), ordinals.dropna().to_numpy(), equal_nan=True):
                period_obj = pd.Series(pd.NaT, index=out.index, dtype="object")
                valid_ord = ordinals.dropna().astype("int64")
                if len(valid_ord) > 0:
                    period_obj.loc[valid_ord.index] = pd.PeriodIndex.from_ordinals(
                        valid_ord.to_numpy(),
                        freq="M",
                    ).astype("object")
                out[col] = pd.PeriodIndex(period_obj, freq="M")
                return out
    out[col] = pd.to_datetime(ser, errors="coerce").dt.to_period("M")
    return out


def normalize_for_compare(
    df: pd.DataFrame,
    sort_by: Optional[Sequence[str]] = None,
    year_month_cols: Sequence[str] = ("year_month",),
) -> pd.DataFrame:
    out = df.copy()
    for col in year_month_cols:
        if col in out.columns:
            out = normalize_year_month_to_period(out, col=col)

    if sort_by:
        valid = [c for c in sort_by if c in out.columns]
        if valid:
            out = out.sort_values(valid, kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    sort_by: Optional[Sequence[str]] = None,
    strict_columns: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> FrameComparison:
    ldf = normalize_for_compare(left, sort_by=sort_by)
    rdf = normalize_for_compare(right, sort_by=sort_by)

    if strict_columns and list(ldf.columns) != list(rdf.columns):
        return FrameComparison(
            ok=False,
            message=f"Column mismatch: left={list(ldf.columns)} right={list(rdf.columns)}",
        )

    common_cols = [c for c in ldf.columns if c in rdf.columns]
    if ldf.shape[0] != rdf.shape[0]:
        return FrameComparison(
            ok=False,
            message=f"Row count mismatch: left={ldf.shape[0]} right={rdf.shape[0]}",
        )

    numeric = [c for c in _numeric_columns(ldf[common_cols]) if c in rdf.columns]
    non_numeric = [c for c in common_cols if c not in numeric]

    mismatched: List[str] = []
    for col in non_numeric:
        lvals = ldf[col].astype("string").fillna("<NA>")
        rvals = rdf[col].astype("string").fillna("<NA>")
        if not (lvals == rvals).all():
            mismatched.append(col)

    for col in numeric:
        lvals = ldf[col].to_numpy(dtype=float)
        rvals = rdf[col].to_numpy(dtype=float)
        both_nan = np.isnan(lvals) & np.isnan(rvals)
        mask = ~both_nan
        if not np.allclose(lvals[mask], rvals[mask], rtol=rtol, atol=atol, equal_nan=True):
            mismatched.append(col)

    if mismatched:
        return FrameComparison(
            ok=False,
            message=f"Mismatched columns: {mismatched}",
            mismatched_columns=mismatched,
        )

    return FrameComparison(ok=True, message="OK")
